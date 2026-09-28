"""
blocking.py — Memory-efficient FAISS blocking for entity resolution at scale.

Memory budget (25 GB total RAM, 4 GB available after data loading):
  Problem: 10.3M DB × 256 dims × float32 = 10.6 GB → too big to hold all at once.

  Solution — process ONE COUNTRY AT A TIME, never holding full DB in memory:
    1. Fit TF-IDF + SVD on sampled texts (500K sample, cached)
    2. For each country partition:
       a. Embed ONLY that country's DB records (e.g., US: 6.2M × 128 × 2 = 1.6 GB)
       b. Build FAISS index for that country
       c. Embed ONLY that country's S1 records, search, store results
       d. Free everything, GC
    3. Merge per-country results

  With 128 SVD dims + float16 storage:
    US DB:    6.2M × 128 × 2 = 1.6 GB  ✓
    India DB: 4.1M × 128 × 2 = 1.1 GB  ✓
    Peak per country: ~2 GB  ✓

  Total pipeline RAM budget: ~6 GB peak (well within 25 GB)
"""

import gc
import sys
import time
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from tqdm import tqdm
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from normalize import normalize_business_name

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
TOP_K              = 25       # Candidates per S1 entity
TFIDF_MAX_FEATURES = 60_000   # Reduced vocab to lower sparse matrix RAM
TFIDF_NGRAM_RANGE  = (2, 4)   # Char n-gram range
SVD_COMPONENTS     = 128      # 128 dims: 2× less RAM vs 256, minimal quality loss
EMBED_BATCH        = 100_000  # Rows per embedding batch (controls RAM spike)
FAISS_BATCH        = 50_000   # Query batch size for FAISS

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────
def load_source(path: Path) -> pd.DataFrame:
    log.info(f"Loading {path.name} ...")
    df = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8", na_filter=True)
    df = df.fillna("")
    log.info(f"  → {len(df):,} rows")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Text features
# ──────────────────────────────────────────────────────────────────────────────
def build_text_features(df: pd.DataFrame) -> list:
    """Vectorized: country + normalized business_name."""
    names     = df["business_name"].fillna("").apply(normalize_business_name)
    countries = df["country"].fillna("unknown").str.lower().str.strip()
    return (countries + " " + names).tolist()


def build_text_features_from_arrays(
    names: "pd.Series", countries: "pd.Series"
) -> list:
    norm_names = names.fillna("").apply(normalize_business_name)
    norm_ctry  = countries.fillna("unknown").str.lower().str.strip()
    return (norm_ctry + " " + norm_names).tolist()


# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF + SVD model (fit once, cached)
# ──────────────────────────────────────────────────────────────────────────────
def fit_tfidf_svd(
    sample_texts: list,
    cache_prefix: str,
) -> tuple:
    """Fit TF-IDF + TruncatedSVD on a sample. Cached to disk."""
    model_cache = CACHE_DIR / f"tfidf_svd_{cache_prefix}.pkl"
    if model_cache.exists():
        log.info(f"Loading TF-IDF+SVD from cache: {model_cache}")
        with open(model_cache, "rb") as f:
            return pickle.load(f)

    n = len(sample_texts)
    log.info(f"Fitting TF-IDF on {n:,} sample texts ...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        min_df=3,
        dtype=np.float32,
    )
    vectorizer.fit(sample_texts)
    log.info(f"  Vocab: {len(vectorizer.vocabulary_):,}  ({time.time()-t0:.1f}s)")

    log.info(f"Fitting TruncatedSVD ({SVD_COMPONENTS} dims) ...")
    t0 = time.time()
    X_sample = vectorizer.transform(sample_texts)
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42, n_iter=5)
    svd.fit(X_sample)
    explained = svd.explained_variance_ratio_.sum()
    log.info(f"  Explained variance: {explained:.3f}  ({time.time()-t0:.1f}s)")
    del X_sample; gc.collect()

    with open(model_cache, "wb") as f:
        pickle.dump((vectorizer, svd), f, protocol=4)
    return vectorizer, svd


# ──────────────────────────────────────────────────────────────────────────────
# Batched embedding (low RAM)
# ──────────────────────────────────────────────────────────────────────────────
def embed_texts(
    texts: list,
    vectorizer: TfidfVectorizer,
    svd: TruncatedSVD,
    batch_size: int = EMBED_BATCH,
    dtype=np.float16,
) -> np.ndarray:
    """
    Transform a list of texts → L2-normalised dense embeddings.
    Uses float16 to halve RAM. Processes in batches.
    """
    n = len(texts)
    chunks = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = texts[start:end]
        X = vectorizer.transform(batch)               # sparse float32
        E = svd.transform(X).astype(np.float32)       # dense float32 (temp)
        E = normalize(E, norm="l2").astype(dtype)     # L2-norm, cast to float16
        del X; gc.collect()
        chunks.append(E)
        del E
    result = np.vstack(chunks)
    del chunks; gc.collect()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# FAISS helpers
# ──────────────────────────────────────────────────────────────────────────────
def faiss_search_batched(
    db_emb: np.ndarray,
    q_emb: np.ndarray,
    top_k: int,
    batch_size: int = FAISS_BATCH,
) -> tuple:
    """
    Build FAISS index on db_emb and search q_emb in batches.
    Uses IndexIVFFlat (approximate) for large DBs: 10-50x faster, ~98% recall.
    Falls back to IndexFlatIP for small DBs (<100K).
    Both arrays should be float32 L2-normalised (cosine via inner product).
    Returns (scores, indices) of shape (n_q, top_k).
    """
    import faiss
    d    = db_emb.shape[1]
    n_db = db_emb.shape[0]
    db_f32 = db_emb.astype(np.float32)

    if n_db >= 100_000:
        # IVF: cluster DB into n_lists partitions, search nprobe at query time
        n_lists = max(64, min(4096, int(np.sqrt(n_db))))
        nprobe  = min(n_lists, 64)   # search ~6% of clusters: good recall vs speed
        quantiser = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantiser, d, n_lists, faiss.METRIC_INNER_PRODUCT)
        log.info(f"    Training IVF index (n_lists={n_lists}, nprobe={nprobe}) ...")
        t_ivf = time.time()
        index.train(db_f32)
        index.add(db_f32)
        index.nprobe = nprobe
        log.info(f"    FAISS IVF built: {index.ntotal:,} vectors  ({time.time()-t_ivf:.1f}s)")
    else:
        index = faiss.IndexFlatIP(d)
        index.add(db_f32)
        log.info(f"    FAISS Flat built: {index.ntotal:,} vectors, dim={d}")

    del db_f32; gc.collect()

    n_q = len(q_emb)
    all_scores  = np.empty((n_q, top_k), dtype=np.float32)
    all_indices = np.empty((n_q, top_k), dtype=np.int64)

    t_search = time.time()
    for start in range(0, n_q, batch_size):
        end = min(start + batch_size, n_q)
        s, idx = index.search(q_emb[start:end].astype(np.float32), top_k)
        all_scores[start:end]  = s
        all_indices[start:end] = idx
        if (start // batch_size) % 10 == 0 and start > 0:
            log.info(f"    FAISS search: {end:,}/{n_q:,} queries  ({time.time()-t_search:.0f}s elapsed)")

    log.info(f"    FAISS search done: {n_q:,} queries in {time.time()-t_search:.1f}s")
    del index; gc.collect()
    return all_scores, all_indices


# ──────────────────────────────────────────────────────────────────────────────
# Main blocking function
# ──────────────────────────────────────────────────────────────────────────────
def run_blocking(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k: int = TOP_K,
    threshold: float = 0.0,
    cache_prefix: str = "run",
) -> dict:
    """
    Returns {s1_entity_id: [candidate_entity_ids]} sorted by cosine score.
    Fully cached — re-runs skip all computation.
    """
    cand_cache = CACHE_DIR / f"candidates_{cache_prefix}.pkl"
    if cand_cache.exists():
        log.info(f"Loading candidates from cache: {cand_cache}")
        with open(cand_cache, "rb") as f:
            return pickle.load(f)

    # ── Extract lightweight arrays (free full DataFrames after use) ──────────
    s1_ids      = s1_df["entity_id"].values
    s1_names    = s1_df["business_name"]
    s1_countries= s1_df["country"].fillna("unknown").str.lower().str.strip()

    s2_ids      = s2_df["entity_id"].values
    s2_names    = s2_df["business_name"]
    s2_countries= s2_df["country"].fillna("unknown").str.lower().str.strip()

    s3_ids      = s3_df["entity_id"].values
    s3_names    = s3_df["business_name"]
    s3_countries= s3_df["country"].fillna("unknown").str.lower().str.strip()

    db_ids      = np.concatenate([s2_ids, s3_ids])
    db_names    = pd.concat([s2_names, s3_names], ignore_index=True)
    db_countries= pd.concat([s2_countries, s3_countries], ignore_index=True)

    # ── Fit TF-IDF + SVD on a memory-efficient sample ───────────────────────
    log.info("Building sample for TF-IDF fitting ...")
    rng = np.random.default_rng(42)
    SAMPLE_SIZE = 500_000   # 500K sample is enough for robust TF-IDF fit

    n_s1  = len(s1_ids)
    n_db  = len(db_ids)
    s1_sample_idx = rng.choice(n_s1,  size=min(n_s1,  100_000), replace=False)
    db_sample_idx = rng.choice(n_db,  size=min(n_db,  400_000), replace=False)

    sample_texts = (
        build_text_features_from_arrays(s1_names.iloc[s1_sample_idx],
                                        s1_countries.iloc[s1_sample_idx]) +
        build_text_features_from_arrays(db_names.iloc[db_sample_idx],
                                        db_countries.iloc[db_sample_idx])
    )
    log.info(f"  Sample size: {len(sample_texts):,}")

    vectorizer, svd = fit_tfidf_svd(sample_texts, cache_prefix)
    del sample_texts; gc.collect()

    # ── Discover country partitions ──────────────────────────────────────────
    countries_in_s1 = sorted(s1_countries.unique())
    log.info(f"Country partitions (from S1): {countries_in_s1}")

    # ── Per-country FAISS search ─────────────────────────────────────────────
    # candidates_raw[i] = list of (global_db_idx, score) for s1_ids[i]
    candidates_raw = [[] for _ in range(n_s1)]

    for country in countries_in_s1:
        if country == "unknown":
            continue
        # --- DB side ---
        db_mask    = (db_countries == country).values
        db_ct_ids  = db_ids[db_mask]
        db_ct_idx  = np.where(db_mask)[0]   # global positions in db arrays
        n_db_ct    = int(db_mask.sum())
        if n_db_ct == 0:
            log.info(f"  [{country}] No DB records — skipping")
            continue

        # --- S1 side ---
        s1_mask    = (s1_countries == country).values
        s1_ct_idx  = np.where(s1_mask)[0]   # global positions in s1 arrays
        n_s1_ct    = int(s1_mask.sum())
        if n_s1_ct == 0:
            log.info(f"  [{country}] No S1 records — skipping")
            continue

        log.info(f"  [{country}] S1={n_s1_ct:,}  DB={n_db_ct:,}")

        # --- Embed DB for this country ---
        t0 = time.time()
        db_ct_texts = build_text_features_from_arrays(
            db_names.iloc[db_ct_idx], db_countries.iloc[db_ct_idx]
        )
        db_emb = embed_texts(db_ct_texts, vectorizer, svd)
        log.info(f"    DB embedded: {db_emb.shape}  ({time.time()-t0:.1f}s)")
        del db_ct_texts; gc.collect()

        # --- Embed S1 for this country ---
        t0 = time.time()
        s1_ct_texts = build_text_features_from_arrays(
            s1_names.iloc[s1_ct_idx], s1_countries.iloc[s1_ct_idx]
        )
        s1_emb = embed_texts(s1_ct_texts, vectorizer, svd)
        log.info(f"    S1 embedded: {s1_emb.shape}  ({time.time()-t0:.1f}s)")
        del s1_ct_texts; gc.collect()

        # --- FAISS search ---
        t0 = time.time()
        scores, local_indices = faiss_search_batched(db_emb, s1_emb, top_k)
        log.info(f"    FAISS search done: {time.time()-t0:.1f}s")
        del db_emb, s1_emb; gc.collect()

        # --- Map local indices → global DB indices, store hits ---
        for qi, s1_global_i in enumerate(s1_ct_idx):
            for k in range(top_k):
                local_db_i = int(local_indices[qi, k])
                if local_db_i < 0:
                    continue
                score = float(scores[qi, k])
                if score >= threshold:
                    global_db_i = int(db_ct_idx[local_db_i])
                    candidates_raw[s1_global_i].append((global_db_i, score))

        del scores, local_indices; gc.collect()

    # ── Handle S1 "unknown" country → search against ALL DB ─────────────────
    unk_mask   = (s1_countries == "unknown").values
    n_unk      = int(unk_mask.sum())
    if n_unk > 0:
        log.info(f"  [unknown] S1={n_unk:,} — searching against full DB ...")
        s1_unk_idx = np.where(unk_mask)[0]

        # Embed full DB in chunks of 1M
        # (fallback: if DB too large, embed per country and merge)
        unk_s1_texts = build_text_features_from_arrays(
            s1_names.iloc[s1_unk_idx], s1_countries.iloc[s1_unk_idx]
        )
        s1_unk_emb = embed_texts(unk_s1_texts, vectorizer, svd)

        # For unknown S1, search each country partition and merge
        for country in set(db_countries.unique()):
            db_mask   = (db_countries == country).values
            db_ct_idx = np.where(db_mask)[0]
            db_ct_texts = build_text_features_from_arrays(
                db_names.iloc[db_ct_idx], db_countries.iloc[db_ct_idx]
            )
            db_emb = embed_texts(db_ct_texts, vectorizer, svd)
            scores, local_indices = faiss_search_batched(db_emb, s1_unk_emb, top_k)
            del db_emb, db_ct_texts; gc.collect()

            for qi, s1_global_i in enumerate(s1_unk_idx):
                for k in range(top_k):
                    local_db_i = int(local_indices[qi, k])
                    if local_db_i < 0:
                        continue
                    score = float(scores[qi, k])
                    if score >= threshold:
                        global_db_i = int(db_ct_idx[local_db_i])
                        candidates_raw[s1_global_i].append((global_db_i, score))
            del scores, local_indices; gc.collect()
        del s1_unk_emb; gc.collect()

    del vectorizer, svd; gc.collect()

    # ── Build output dict ────────────────────────────────────────────────────
    log.info("Building candidate dict ...")
    candidates = {}
    for i, s1_id in enumerate(s1_ids):
        hits = candidates_raw[i]
        if hits:
            hits.sort(key=lambda x: -x[1])
            seen, dedup = set(), []
            for db_idx, _ in hits:
                cid = db_ids[db_idx]
                if cid not in seen:
                    seen.add(cid)
                    dedup.append(cid)
            candidates[s1_id] = dedup
        else:
            candidates[s1_id] = []

    n_nonempty = sum(1 for v in candidates.values() if v)
    avg_c = np.mean([len(v) for v in candidates.values()])
    log.info(f"  Done — non-empty: {n_nonempty:,}/{len(candidates):,}  avg cands: {avg_c:.2f}")

    with open(cand_cache, "wb") as f:
        pickle.dump(candidates, f, protocol=4)
    log.info(f"  Candidates cached → {cand_cache}")
    return candidates
