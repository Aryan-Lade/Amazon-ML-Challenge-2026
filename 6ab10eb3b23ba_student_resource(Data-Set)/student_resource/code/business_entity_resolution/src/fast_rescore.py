"""
fast_rescore.py — High-throughput test candidate re-scoring with calibrated threshold 0.980.

Key features:
1. Pre-computes normalized names, addresses, PINs, tokens, and n-grams for S1 entities.
2. Threshold = 0.980 (calibrated for peak Macro F_0.5 on held-out validation set).
3. Persistent chunk checkpointing: progress saved every 50k entities to resume seamlessly.
4. Auto-generates TSV outputs, runs validator, and synchronizes submission package.
"""

import gc
import sys
import time
import json
import shutil
import pickle
import logging
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
import re

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from normalize import normalize_business_name, normalize_address, extract_pin_or_zip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

ROOT_DIR = Path(".").resolve()
DATA_DIR = ROOT_DIR / "dataset/test"
CACHE_DIR = ROOT_DIR / ".cache"
CHECKPOINT_DIR = CACHE_DIR / "rescore_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_CACHE = CACHE_DIR / "model/lgbm_model.pkl"
CAND_CACHE  = CACHE_DIR / "candidates_test.pkl"
TEST_S1     = DATA_DIR / "test_source1.tsv"
TEST_S2     = DATA_DIR / "test_source2.tsv"
TEST_S3     = DATA_DIR / "test_source3.tsv"

TARGET_THRESHOLD = 0.980
FEATURE_BATCH_SIZE = 50_000
CHUNK_SIZE = 50_000


def extract_city(addr: str) -> str:
    tokens = addr.split()
    for tok in reversed(tokens):
        if tok.isalpha() and len(tok) > 2:
            return tok.lower()
    return ""


def precompute_s1_entity(name: str, addr: str, country: str) -> tuple:
    n = normalize_business_name(name)
    a = normalize_address(addr)
    pin = extract_pin_or_zip(addr)
    city = extract_city(a)
    c = str(country).lower().strip()
    ta = set(n.split())
    nga = set(n[j:j+3] for j in range(len(n)-2)) if len(n) >= 3 else set()
    n_digits = set(re.findall(r"\d+", n))
    n1t = n.split()
    n_first = n1t[0] if n1t else ""
    ta_a = set(a.split())
    nga_a = set(a[j:j+3] for j in range(len(a)-2)) if len(a) >= 3 else set()
    a_digits = set(re.findall(r"\d+", a))
    a1t = a.split()
    a_first = a1t[0] if a1t else ""
    return (n, a, pin, city, c, ta, nga, n_digits, n_first, ta_a, nga_a, a_digits, a_first)


def precompute_db_entity(name: str, addr: str, country: str) -> tuple:
    n = normalize_business_name(name)
    a = normalize_address(addr)
    pin = extract_pin_or_zip(addr)
    city = extract_city(a)
    c = str(country).lower().strip()
    return (n, a, pin, city, c)


def fast_compute_features_batch(pair_e1s: list, pair_e2s: list, blocking_scores: list) -> np.ndarray:
    n = len(pair_e1s)
    X = np.zeros((n, 35), dtype=np.float32)
    
    for i in range(n):
        (n1, a1, pin1, city1, c1, ta, nga, n1d, n1_first, ta_a, nga_a, a1d, a1_first) = pair_e1s[i]
        n2, a2, pin2, city2, c2 = pair_e2s[i]
        bs = blocking_scores[i]
        
        # 1. Name features
        name_exact = float(n1 == n2 and n1 != "")
        tb = set(n2.split())
        if not ta and not tb: name_jaccard = 1.0; name_overlap = 1.0
        elif not ta or not tb: name_jaccard = 0.0; name_overlap = 0.0
        else:
            inter = len(ta & tb)
            name_jaccard = inter / len(ta | tb)
            name_overlap = inter / min(len(ta), len(tb))
            
        name_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
        name_set = fuzz.token_set_ratio(n1, n2) / 100.0
        name_partial = fuzz.partial_ratio(n1, n2) / 100.0
        name_ratio = fuzz.ratio(n1, n2) / 100.0
        name_jw = distance.JaroWinkler.normalized_similarity(n1, n2)
        name_prefix = float(n1[:3] == n2[:3] and len(n1) >= 3 and len(n2) >= 3)
        
        if len(n1) < 3 and len(n2) < 3: name_char_ngram = 1.0 if n1 == n2 else 0.0
        elif len(n1) < 3 or len(n2) < 3: name_char_ngram = 0.0
        else:
            ngb = set(n2[j:j+3] for j in range(len(n2)-2))
            name_char_ngram = len(nga & ngb) / len(nga | ngb)
            
        len1, len2 = max(1, len(n1)), max(1, len(n2))
        name_len_ratio = min(len1, len2) / max(len1, len2)
        n2t = n2.split()
        name_first_word = float(bool(n1_first and n2t and n1_first == n2t[0]))
        n2d = set(re.findall(r"\d+", n2))
        name_digit_match = (float(bool(n1d & n2d)) if (n1d and n2d) else 1.0)
        
        if n1 and n2:
            lev = distance.Levenshtein.distance(n1[:100], n2[:100])
            name_levenshtein = 1.0 - lev / max(1, max(len(n1[:100]), len(n2[:100])))
        else:
            name_levenshtein = 0.0
            
        country_match = (float(c1 == c2) if (c1 and c2) else 0.5)
        
        # 2. Addr features
        tb_a = set(a2.split())
        if not ta_a and not tb_a: addr_jaccard = 1.0; addr_overlap = 1.0
        elif not ta_a or not tb_a: addr_jaccard = 0.0; addr_overlap = 0.0
        else:
            inter_a = len(ta_a & tb_a)
            addr_jaccard = inter_a / len(ta_a | tb_a)
            addr_overlap = inter_a / min(len(ta_a), len(tb_a))
            
        addr_sort = fuzz.token_sort_ratio(a1, a2) / 100.0
        addr_set = fuzz.token_set_ratio(a1, a2) / 100.0
        addr_partial = fuzz.partial_ratio(a1, a2) / 100.0
        pin_match = float(pin1 != "" and pin1 == pin2)
        pin_present = float(pin1 != "" and pin2 != "")
        addr1_empty = float(a1 == ""); addr2_empty = float(a2 == "")
        city_match = float(city1 == city2 and city1 != "")
        a2t = a2.split()
        addr_first_token = float(bool(a1_first and a2t and a1_first == a2t[0]))
        
        if a1 and a2:
            if len(a1) < 3 and len(a2) < 3: addr_char_ngram = 1.0 if a1 == a2 else 0.0
            elif len(a1) < 3 or len(a2) < 3: addr_char_ngram = 0.0
            else:
                ngb_a = set(a2[j:j+3] for j in range(len(a2)-2))
                addr_char_ngram = len(nga_a & ngb_a) / len(nga_a | ngb_a)
            alen1, alen2 = max(1, len(a1)), max(1, len(a2))
            addr_len_ratio = min(alen1, alen2) / max(alen1, alen2)
        else:
            addr_char_ngram = 0.0
            addr_len_ratio = 0.0
            
        a2d = set(re.findall(r"\d+", a2))
        addr_digit_match = float(bool(a1d & a2d)) if (a1d and a2d) else 0.0
        
        name_addr_product = name_sort * addr_sort
        blocking_x_name = bs * name_sort
        name_set_x_addr_set = name_set * addr_set
        pin_x_name = pin_match * name_set
        exact_x_country = name_exact * country_match
        
        X[i] = [
            name_exact, name_jaccard, name_overlap, name_sort, name_set,
            name_partial, name_ratio, name_jw, name_prefix, name_char_ngram,
            name_len_ratio, name_first_word, name_digit_match, name_levenshtein,
            country_match,
            addr_jaccard, addr_overlap, addr_sort, addr_set, addr_partial,
            pin_match, pin_present, addr1_empty, addr2_empty, city_match,
            addr_first_token, addr_char_ngram, addr_len_ratio, addr_digit_match,
            bs,
            name_addr_product, blocking_x_name, name_set_x_addr_set, pin_x_name,
            exact_x_country,
        ]
        
    return X


def main():
    t_start = time.time()
    log.info("=" * 60)
    log.info(f"HIGH-THROUGHPUT RE-SCORING MODE (Threshold={TARGET_THRESHOLD:.3f})")
    log.info(f"Checkpoints directory: {CHECKPOINT_DIR}")
    log.info("=" * 60)
    
    # 1. Load model
    log.info(f"Loading LightGBM model from {MODEL_CACHE} ...")
    with open(MODEL_CACHE, "rb") as f:
        model = pickle.load(f)
    log.info("Model loaded successfully.")
    
    # 2. Load candidates
    log.info(f"Loading test candidate cache from {CAND_CACHE} ...")
    with open(CAND_CACHE, "rb") as f:
        test_candidates = pickle.load(f)
    log.info(f"Loaded candidates for {len(test_candidates):,} S1 entities.")
    
    # Determine all candidate IDs needed
    needed_db_ids = set()
    for cands in test_candidates.values():
        needed_db_ids.update(cands)
    log.info(f"Unique candidate DB IDs needed: {len(needed_db_ids):,}")
    
    # 3. Load test data and precompute
    log.info("Loading and precomputing S1 entities ...")
    s1_df = pd.read_csv(TEST_S1, sep="\t", dtype=str).fillna("")
    s1_ids = s1_df["entity_id"].tolist()
    
    s1_pre = {}
    for sid, name, addr, ctry in zip(s1_df["entity_id"], s1_df["business_name"], s1_df["business_address"], s1_df["country"]):
        s1_pre[sid] = precompute_s1_entity(name, addr, ctry)
    del s1_df; gc.collect()
    log.info(f"S1 precomputed: {len(s1_pre):,} records.")
    
    log.info("Loading and precomputing candidate DB entities ...")
    db_pre = {}
    
    def process_source(path, prefix):
        log.info(f"  Reading {path.name} ...")
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        mask = df["entity_id"].isin(needed_db_ids)
        sub = df[mask]
        for eid, name, addr, ctry in zip(sub["entity_id"], sub["business_name"], sub["business_address"], sub["country"]):
            db_pre[eid] = precompute_db_entity(name, addr, ctry)
        log.info(f"    Loaded {len(sub):,} matching {prefix} records.")
        del df, sub; gc.collect()
        
    process_source(TEST_S2, "S2")
    process_source(TEST_S3, "S3")
    log.info(f"Total DB records precomputed: {len(db_pre):,}")
    
    # 4. Scoring loop with batching & checkpointing
    log.info(f"Starting candidate scoring with calibrated threshold {TARGET_THRESHOLD:.3f} ...")
    n_chunks = (len(s1_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    t_score0 = time.time()
    total_pairs_scored = 0
    total_matches_found = 0
    all_chunk_matches = {}
    
    for chunk_idx in range(n_chunks):
        c_start = chunk_idx * CHUNK_SIZE
        c_end = min(c_start + CHUNK_SIZE, len(s1_ids))
        chunk_s1 = s1_ids[c_start:c_end]
        ckpt_file = CHECKPOINT_DIR / f"chunk_{chunk_idx:03d}.pkl"
        
        # Check if already processed in previous run
        if ckpt_file.exists():
            with open(ckpt_file, "rb") as f:
                c_matches, c_pairs, c_mcount = pickle.load(f)
            all_chunk_matches.update(c_matches)
            total_pairs_scored += c_pairs
            total_matches_found += c_mcount
            log.info(f"  [Checkpoint Loaded] Chunk {chunk_idx+1}/{n_chunks} ({c_end:,}/{len(s1_ids):,}) | +{c_mcount:,} matches")
            continue
            
        chunk_matches = {}
        pair_e1s = []
        pair_e2s = []
        pair_meta = []
        
        for sid in chunk_s1:
            cands = test_candidates.get(sid, [])
            if not cands: continue
            e1 = s1_pre.get(sid)
            if not e1: continue
            n1 = e1[0]
            
            for cid in cands:
                e2 = db_pre.get(cid)
                if not e2: continue
                n2 = e2[0]
                
                # Fast rejection: threshold < 18 (preserves acronyms like SBI vs State Bank of India)
                if fuzz.token_set_ratio(n1, n2) < 18:
                    continue
                    
                pair_e1s.append(e1)
                pair_e2s.append(e2)
                pair_meta.append((sid, cid))
                
        chunk_mcount = 0
        if pair_meta:
            total_pairs_scored += len(pair_meta)
            bs = FEATURE_BATCH_SIZE
            for b_i in range(0, len(pair_meta), bs):
                b_end = min(b_i + bs, len(pair_meta))
                X_batch = fast_compute_features_batch(
                    pair_e1s[b_i:b_end],
                    pair_e2s[b_i:b_end],
                    [0.5] * (b_end - b_i),
                )
                probs = model.predict_proba(X_batch)[:, 1]
                for (sid, cid), p in zip(pair_meta[b_i:b_end], probs):
                    if p >= TARGET_THRESHOLD:
                        chunk_matches.setdefault(sid, []).append(cid)
                        chunk_mcount += 1
                        
        total_matches_found += chunk_mcount
        all_chunk_matches.update(chunk_matches)
        
        # Save checkpoint immediately
        with open(ckpt_file, "wb") as f:
            pickle.dump((chunk_matches, len(pair_meta), chunk_mcount), f)
            
        elapsed = time.time() - t_score0
        pct = (chunk_idx + 1) / n_chunks * 100
        rate = total_pairs_scored / max(1e-5, elapsed)
        log.info(f"  Processed chunk {chunk_idx+1}/{n_chunks} ({c_end:,}/{len(s1_ids):,} entities, {pct:.1f}%) | Pairs scored: {total_pairs_scored:,} ({rate:,.0f} pairs/s) | Matches: {total_matches_found:,} | Time: {elapsed/60:.1f}m")
        
    matches = all_chunk_matches
    
    # 5. Write outputs
    log.info("Writing updated submission files ...")
    out_m = OUTPUT_DIR / "matching_results.tsv"
    out_c = OUTPUT_DIR / "candidate_pairs.tsv"
    
    log.info(f"Writing {out_m} ...")
    with open(out_m, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ids:
            m_list = matches.get(sid, [])
            f.write(f"{sid}\t{','.join(m_list)}\n")
            
    log.info(f"Writing {out_c} ...")
    with open(out_c, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_ids:
            c_list = test_candidates.get(sid, [])
            f.write(f"{sid}\t{','.join(c_list)}\n")
            
    log.info("Output files written successfully.")
    
    # Also sync to workspace root
    root_m = Path("..") / "matching_results.tsv"
    root_c = Path("..") / "candidate_pairs.tsv"
    try:
        shutil.copy(out_m, root_m)
        shutil.copy(out_c, root_c)
        log.info(f"Copied outputs to {root_m.resolve()} and {root_c.resolve()}")
    except Exception as e:
        log.warning(f"Could not copy to root: {e}")
        
    n_matched = sum(1 for sid in s1_ids if matches.get(sid))
    log.info("Summary Statistics:")
    log.info(f"  Total S1 entities       : {len(s1_ids):,}")
    log.info(f"  Entities with >=1 match : {n_matched:,} ({n_matched/len(s1_ids)*100:.1f}%)")
    log.info(f"  Singletons (0 matches)  : {len(s1_ids) - n_matched:,} ({(len(s1_ids) - n_matched)/len(s1_ids)*100:.1f}%)")
    log.info(f"  Total matches predicted : {total_matches_found:,}")
    log.info(f"  Avg matches per S1      : {total_matches_found / len(s1_ids):.3f}")
    
    # 6. Validate with official validator
    val_script = ROOT_DIR / "utils/validate_submission.py"
    if val_script.exists():
        log.info("Running official validator check ...")
        cmd = [
            sys.executable, str(val_script),
            "--matching", str(out_m),
            "--candidate", str(out_c),
            "--test-dir", str(DATA_DIR),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        log.info(f"Validator output:\n{res.stdout}")
        if res.returncode == 0:
            log.info("VALIDATION SUCCESS: Output files are 100% compliant!")
        else:
            log.error(f"Validation failed: {res.stderr}")
            
    # 7. Re-package submission zip
    pkg_script = ROOT_DIR / "scratch/package_submission.py"
    if pkg_script.exists():
        log.info("Re-packaging submission archive ...")
        res_pkg = subprocess.run([sys.executable, str(pkg_script)], capture_output=True, text=True)
        log.info(f"Packager output:\n{res_pkg.stdout}")
        zip_path = ROOT_DIR / "Anonymous Coder_submission.zip"
        root_zip = Path("..") / "Anonymous Coder_submission.zip"
        if zip_path.exists():
            shutil.copy(zip_path, root_zip)
            log.info(f"Synchronized submission zip to {root_zip.resolve()}")
            
    log.info(f"Total re-score runtime: {(time.time() - t_start)/60:.1f} minutes.")


if __name__ == "__main__":
    main()
