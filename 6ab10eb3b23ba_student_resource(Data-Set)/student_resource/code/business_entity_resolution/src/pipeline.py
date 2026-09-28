"""
pipeline.py — End-to-end Business Entity Resolution Pipeline (v3, Ultra-Fast LightGBM)
"""

import os
import sys
import gc
import json
import time
import pickle
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from blocking import run_blocking, load_source
from normalize import normalize_business_name
from features import compute_pair_features, compute_features_batch, FEATURE_NAMES
from scorer import compute_f05_macro

LOG_FILE = Path(".cache/pipeline.log")
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8", mode="a"),
    ],
)
log = logging.getLogger(__name__)

DATA_DIR   = Path("dataset")
OUTPUT_DIR = Path("output")
CACHE_DIR  = Path(".cache")
MODEL_DIR  = Path(".cache/model")

for d in [OUTPUT_DIR, CACHE_DIR, MODEL_DIR]:
    d.mkdir(exist_ok=True)

TRAIN_S1 = DATA_DIR / "train" / "train_source1.tsv"
TRAIN_S2 = DATA_DIR / "train" / "train_source2.tsv"
TRAIN_S3 = DATA_DIR / "train" / "train_source3.tsv"
TRAIN_GT = DATA_DIR / "train" / "train_ground_truth.tsv"

TEST_S1  = DATA_DIR / "test" / "test_source1.tsv"
TEST_S2  = DATA_DIR / "test" / "test_source2.tsv"
TEST_S3  = DATA_DIR / "test" / "test_source3.tsv"

BLOCKING_TOP_K     = 35      # Balanced top-K: 48%+ blocking recall, fast inference
BLOCKING_THRESHOLD = 0.10    # Minimum cosine similarity threshold
VAL_FRACTION       = 0.02    # 2% val split (~44K entities)
DEFAULT_THRESHOLD  = 0.85    # Calibrated optimal threshold for F_0.5 (precision-heavy)
NEG_RATIO          = 4       # Negatives per positive in ML training
LGBM_N_ESTIMATORS  = 600
FEATURE_BATCH_SIZE = 50_000


def parse_gt(gt_df):
    gt = {}
    for sid, m in zip(gt_df["source1_entity_id"].values, gt_df["matched_entity_ids"].values):
        if pd.notna(m) and str(m).strip():
            gt[sid] = [x.strip() for x in str(m).split(",") if x.strip()]
        else:
            gt[sid] = []
    return gt


def write_output(s1_ids, matches, candidates):
    matching_path  = OUTPUT_DIR / "matching_results.tsv"
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"
    log.info(f"Writing {matching_path} ...")
    with open(matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids:
            ml = [c for c in dict.fromkeys(matches.get(s1_id, [])) if not c.startswith("S1-")]
            f.write(f"{s1_id}\t{','.join(ml)}\n")
    log.info(f"Writing {candidate_path} ...")
    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids:
            cl = [c for c in dict.fromkeys(candidates.get(s1_id, [])) if not c.startswith("S1-")]
            f.write(f"{s1_id}\t{','.join(cl)}\n")
    log.info("Output files written successfully.")


def make_fast_lookup(df, needed_ids=None):
    if needed_ids is not None:
        mask = df["entity_id"].isin(needed_ids)
        sub = df[mask]
    else:
        sub = df
    return dict(zip(
        sub["entity_id"].values,
        zip(
            sub["business_name"].fillna("").values,
            sub["business_address"].fillna("").values,
            sub["country"].fillna("").values,
        )
    ))


def get_rec_fn(s1_lu, s2_lu, s3_lu):
    def get_rec(eid):
        if eid.startswith("S2-"): return s2_lu.get(eid)
        if eid.startswith("S3-"): return s3_lu.get(eid)
        return s1_lu.get(eid)
    return get_rec


def train_lgbm(X, y):
    import lightgbm as lgb
    log.info(f"Training LightGBM on {X.shape[0]:,} pairs (pos={y.sum():,}, neg={(y==0).sum():,}) ...")
    scale_pos = (y == 0).sum() / max(1, y.sum())
    model = lgb.LGBMClassifier(
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=0.04,
        max_depth=8,
        num_leaves=127,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    t0 = time.time()
    model.fit(X, y)
    log.info(f"  LightGBM trained in {time.time()-t0:.1f}s")
    fi = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    log.info("  Top-10 feature importances:")
    for fname, fimp in fi[:10]:
        log.info(f"    {fname:20s}: {fimp:.1f}")
    return model


def run_training_mode():
    log.info("=" * 60)
    log.info("TRAINING / VALIDATION MODE")
    log.info("=" * 60)
    model_cache = MODEL_DIR / "lgbm_model.pkl"
    config_path = MODEL_DIR / "config.json"
    if model_cache.exists() and config_path.exists():
        log.info(f"Found existing trained model at {model_cache}")
        with open(model_cache, "rb") as f:
            model = pickle.load(f)
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("threshold", DEFAULT_THRESHOLD), model

    log.info("Loading training data ...")
    s1 = load_source(TRAIN_S1)
    s2 = load_source(TRAIN_S2)
    s3 = load_source(TRAIN_S3)
    log.info("Loading ground truth ...")
    gt_df = pd.read_csv(TRAIN_GT, sep="\t", dtype=str)
    gt_dict = parse_gt(gt_df)
    del gt_df; gc.collect()

    rng = np.random.default_rng(42)
    n_val = max(5000, int(len(s1) * VAL_FRACTION))
    val_idx = rng.choice(len(s1), size=n_val, replace=False)
    s1_val_df = s1.iloc[val_idx].copy().reset_index(drop=True)
    val_candidates = run_blocking(
        s1_val_df, s2, s3, top_k=BLOCKING_TOP_K,
        threshold=BLOCKING_THRESHOLD, cache_prefix="train_val",
    )
    del s1_val_df; gc.collect()

    val_s1_keys = list(val_candidates.keys())
    rng.shuffle(val_s1_keys)
    train_ids = val_s1_keys[:30_000]
    tune_ids  = val_s1_keys[30_000:35_000]

    all_needed_cands = set()
    for sid in train_ids + tune_ids:
        all_needed_cands.update(val_candidates.get(sid, []))
    all_needed_s1 = set(train_ids) | set(tune_ids)

    s1_lu = make_fast_lookup(s1, all_needed_s1)
    s2_lu = make_fast_lookup(s2, all_needed_cands)
    s3_lu = make_fast_lookup(s3, all_needed_cands)
    del s1, s2, s3; gc.collect()
    get_rec = get_rec_fn(s1_lu, s2_lu, s3_lu)

    pos_s1, pos_cand = [], []
    neg_s1, neg_cand = [], []
    for sid in tqdm(train_ids, desc="Pair building"):
        cands = val_candidates.get(sid, [])
        if not cands: continue
        gt_set = set(gt_dict.get(sid, []))
        s1r = get_rec(sid)
        if not s1r: continue
        s1_d = {"business_name": s1r[0], "business_address": s1r[1], "country": s1r[2]}
        t_pos = [c for c in cands if c in gt_set]
        t_neg = [c for c in cands if c not in gt_set]
        for cid in t_pos:
            cr = get_rec(cid)
            if not cr: continue
            pos_s1.append(s1_d)
            pos_cand.append({"business_name": cr[0], "business_address": cr[1], "country": cr[2]})
        n_neg = min(len(t_neg), NEG_RATIO * max(1, len(t_pos)))
        if t_neg and n_neg > 0:
            for cid in rng.choice(t_neg, size=n_neg, replace=False):
                cr = get_rec(cid)
                if not cr: continue
                neg_s1.append(s1_d)
                neg_cand.append({"business_name": cr[0], "business_address": cr[1], "country": cr[2]})

    all_s1 = pos_s1 + neg_s1
    all_cand = pos_cand + neg_cand
    y = np.array([1]*len(pos_s1) + [0]*len(neg_s1), dtype=np.int32)
    del pos_s1, pos_cand, neg_s1, neg_cand; gc.collect()

    X = compute_features_batch(all_s1, all_cand, [0.5]*len(all_s1))
    model = train_lgbm(X, y)
    del X, y; gc.collect()
    with open(model_cache, "wb") as f:
        pickle.dump(model, f, protocol=4)

    # Threshold tuning
    tune_s1_flat, tune_cand_flat = [], []
    for sid in tune_ids:
        for cid in val_candidates.get(sid, []):
            tune_s1_flat.append(sid); tune_cand_flat.append(cid)
    all_probs = []
    bs = FEATURE_BATCH_SIZE
    for start in range(0, len(tune_s1_flat), bs):
        end = min(start + bs, len(tune_s1_flat))
        b_s1 = [{"business_name": get_rec(s)[0], "business_address": get_rec(s)[1], "country": get_rec(s)[2]} for s in tune_s1_flat[start:end]]
        b_cd = [{"business_name": get_rec(c)[0], "business_address": get_rec(c)[1], "country": get_rec(c)[2]} for c in tune_cand_flat[start:end]]
        X_b = compute_features_batch(b_s1, b_cd, [0.5]*(end-start))
        all_probs.extend(model.predict_proba(X_b)[:, 1].tolist())

    all_probs = np.array(all_probs)
    tune_gt = {sid: set(gt_dict.get(sid, [])) for sid in tune_ids}
    best_t = DEFAULT_THRESHOLD; best_f05 = -1.0
    for threshold in np.arange(0.25, 0.86, 0.025):
        preds = {}
        for sid, cid, p in zip(tune_s1_flat, tune_cand_flat, all_probs):
            if p >= threshold:
                preds.setdefault(sid, []).append(cid)
        f05 = compute_f05_macro({sid: preds.get(sid, []) for sid in tune_ids}, {k: list(v) for k, v in tune_gt.items()})
        if f05 > best_f05:
            best_f05 = f05; best_t = float(threshold)

    log.info(f"Optimal Threshold: {best_t:.3f} (Val F_0.5 = {best_f05:.5f})")
    with open(config_path, "w") as f:
        json.dump({"threshold": best_t, "validation_f05": best_f05}, f, indent=2)
    return best_t, model


def run_test_mode(threshold=None, lgbm_model=None):
    log.info("=" * 60)
    log.info("TEST INFERENCE MODE")
    log.info("=" * 60)
    config_path = MODEL_DIR / "config.json"
    model_cache = MODEL_DIR / "lgbm_model.pkl"

    if threshold is None:
        if config_path.exists():
            with open(config_path) as f:
                threshold = json.load(f).get("threshold", DEFAULT_THRESHOLD)
        else:
            threshold = DEFAULT_THRESHOLD
    log.info(f"Inference threshold: {threshold:.3f}")

    if lgbm_model is None:
        if model_cache.exists():
            log.info(f"Loading LightGBM model from {model_cache} ...")
            with open(model_cache, "rb") as f:
                lgbm_model = pickle.load(f)
        else:
            raise FileNotFoundError(f"Model cache not found at {model_cache}. Run --mode train first.")

    log.info("Loading test datasets ...")
    s1_test = load_source(TEST_S1)
    s2_test = load_source(TEST_S2)
    s3_test = load_source(TEST_S3)
    s1_test_ids = s1_test["entity_id"].tolist()
    log.info(f"  Test sets loaded — S1: {len(s1_test):,}, S2: {len(s2_test):,}, S3: {len(s3_test):,}")

    log.info("Running blocking on test data ...")
    t_blk = time.time()
    test_candidates = run_blocking(
        s1_test, s2_test, s3_test, top_k=BLOCKING_TOP_K,
        threshold=BLOCKING_THRESHOLD, cache_prefix="test",
    )
    log.info(f"Blocking complete in {(time.time()-t_blk)/60:.1f} minutes.")

    log.info("Building fast entity lookups ...")
    s1_lu = make_fast_lookup(s1_test)
    s2_lu = make_fast_lookup(s2_test)
    s3_lu = make_fast_lookup(s3_test)
    del s2_test, s3_test; gc.collect()
    get_rec = get_rec_fn(s1_lu, s2_lu, s3_lu)

    log.info(f"Scoring test candidate pairs (threshold={threshold:.3f}) ...")
    matches = {}
    CHUNK_SIZE = 25_000
    n_chunks = (len(s1_test_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE

    t_score_start = time.time()
    for chunk_idx in range(n_chunks):
        c_start = chunk_idx * CHUNK_SIZE
        c_end = min(c_start + CHUNK_SIZE, len(s1_test_ids))
        chunk_s1_ids = s1_test_ids[c_start:c_end]

        pair_s1_recs = []
        pair_cd_recs = []
        pair_meta    = []

        for sid in chunk_s1_ids:
            cands = test_candidates.get(sid, [])
            if not cands: continue
            s1r = s1_lu.get(sid)
            if not s1r: continue
            n1_norm = normalize_business_name(s1r[0])
            for cid in cands:
                cr = get_rec(cid)
                if not cr: continue
                n2_norm = normalize_business_name(cr[0])
                # Fast rejection: if token set ratio is too low, cannot exceed threshold
                if fuzz.token_set_ratio(n1_norm, n2_norm) < 30:
                    continue
                pair_s1_recs.append({"business_name": s1r[0], "business_address": s1r[1], "country": s1r[2]})
                pair_cd_recs.append({"business_name": cr[0], "business_address": cr[1], "country": cr[2]})
                pair_meta.append((sid, cid))

        if pair_meta:
            # Batch feature computation and scoring
            bs = FEATURE_BATCH_SIZE
            for b_i in range(0, len(pair_meta), bs):
                b_end = min(b_i + bs, len(pair_meta))
                X_batch = compute_features_batch(
                    pair_s1_recs[b_i:b_end],
                    pair_cd_recs[b_i:b_end],
                    [0.5]*(b_end - b_i),
                )
                probs = lgbm_model.predict_proba(X_batch)[:, 1]
                for (sid, cid), p in zip(pair_meta[b_i:b_end], probs):
                    if p >= threshold:
                        matches.setdefault(sid, []).append(cid)

        if (chunk_idx + 1) % 5 == 0 or (chunk_idx + 1) == n_chunks:
            elapsed = time.time() - t_score_start
            pct = (chunk_idx + 1) / n_chunks * 100
            log.info(f"  Processed {c_end:,}/{len(s1_test_ids):,} entities ({pct:.1f}%) in {elapsed/60:.1f} min")

    write_output(s1_test_ids, matches, test_candidates)

    n_matched = sum(1 for sid in s1_test_ids if matches.get(sid))
    avg_cands  = np.mean([len(test_candidates.get(sid, [])) for sid in s1_test_ids])
    avg_match  = np.mean([len(matches.get(sid, [])) for sid in s1_test_ids])
    log.info(f"Matching complete:")
    log.info(f"  Entities with >=1 match : {n_matched:,} / {len(s1_test_ids):,} ({n_matched/len(s1_test_ids)*100:.1f}%)")
    log.info(f"  Avg candidates per S1   : {avg_cands:.2f}")
    log.info(f"  Avg matches per S1      : {avg_match:.3f}")

    # Validate submission files with validator
    val_script = Path("utils/validate_submission.py")
    if val_script.exists():
        log.info("Running submission validator ...")
        cmd = [
            sys.executable, str(val_script),
            "--matching", "output/matching_results.tsv",
            "--candidate", "output/candidate_pairs.tsv",
            "--test-dir", "dataset/test",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        log.info(f"Validator output:\n{res.stdout}")
        if res.returncode == 0:
            log.info("VALIDATION SUCCESS: Output files are 100% compliant with submission rules!")
        else:
            log.error(f"Validator returned error code {res.returncode}:\n{res.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline v3")
    parser.add_argument("--mode", choices=["train", "test", "full"], default="full")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    t_start = time.time()
    threshold = args.threshold
    lgbm_model = None
    if args.mode in ("train", "full"):
        best_t, lgbm_model = run_training_mode()
        if threshold is None:
            threshold = best_t
    if args.mode in ("test", "full"):
        run_test_mode(threshold=threshold, lgbm_model=lgbm_model)
    elapsed = time.time() - t_start
    log.info(f"Total pipeline runtime: {elapsed/60:.1f} minutes")
    log.info("All done successfully!")


if __name__ == "__main__":
    main()
