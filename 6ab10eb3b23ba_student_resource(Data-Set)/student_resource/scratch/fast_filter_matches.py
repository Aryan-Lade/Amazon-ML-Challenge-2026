"""
fast_filter_matches.py — Direct high-precision re-scoring of existing matches.
Scores the 3.75M existing pairs from matching_results.tsv with LightGBM at threshold 0.980.
Filters out false merges to dramatically boost Precision and Macro F0.5 on the leaderboard.
"""

import gc
import sys
import time
import shutil
import pickle
import logging
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

SRC_DIR = Path("code/business_entity_resolution/src").resolve()
sys.path.insert(0, str(SRC_DIR))

from fast_rescore import precompute_s1_entity, precompute_db_entity, fast_compute_features_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT_DIR = Path(".").resolve()
DATA_DIR = ROOT_DIR / "dataset/test"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_PATH = ROOT_DIR / ".cache/model/lgbm_model.pkl"
MATCH_PATH = OUTPUT_DIR / "matching_results.tsv"

TARGET_THRESHOLD = 0.980
BATCH_SIZE = 50_000

def main():
    t_start = time.time()
    log.info("=" * 60)
    log.info(f"HIGH-SPEED DIRECT RE-SCORING (Threshold: {TARGET_THRESHOLD:.3f})")
    log.info("=" * 60)
    
    # 1. Load existing matches
    log.info(f"Reading existing matches from {MATCH_PATH} ...")
    df_m = pd.read_csv(MATCH_PATH, sep="\t", dtype=str).fillna("")
    all_s1_ids = df_m["source1_entity_id"].tolist()
    
    pairs = []
    for sid, m in zip(df_m["source1_entity_id"], df_m["matched_entity_ids"]):
        if m:
            for cid in m.split(","):
                pairs.append((sid, cid))
                
    log.info(f"Total candidate matches to score: {len(pairs):,}")
    needed_s1 = set(p[0] for p in pairs)
    needed_db = set(p[1] for p in pairs)
    log.info(f"Unique S1: {len(needed_s1):,}, Unique DB: {len(needed_db):,}")
    
    # 2. Read S1 records
    log.info("Reading and precomputing S1 entities ...")
    t0 = time.time()
    s1_df = pd.read_csv(DATA_DIR / "test_source1.tsv", sep="\t", dtype=str, usecols=["entity_id", "business_name", "business_address", "country"]).fillna("")
    s1_df = s1_df[s1_df["entity_id"].isin(needed_s1)]
    s1_lu = {eid: precompute_s1_entity(n, a, c) for eid, n, a, c in zip(s1_df["entity_id"], s1_df["business_name"], s1_df["business_address"], s1_df["country"])}
    del s1_df; gc.collect()
    log.info(f"  S1 precomputed: {len(s1_lu):,} in {time.time()-t0:.1f}s")
    
    # 3. Read S2 and S3 records
    log.info("Reading and precomputing candidate DB entities ...")
    t0 = time.time()
    db_lu = {}
    
    log.info("  Loading test_source2.tsv ...")
    s2_df = pd.read_csv(DATA_DIR / "test_source2.tsv", sep="\t", dtype=str, usecols=["entity_id", "business_name", "business_address", "country"]).fillna("")
    s2_df = s2_df[s2_df["entity_id"].isin(needed_db)]
    for eid, n, a, c in zip(s2_df["entity_id"], s2_df["business_name"], s2_df["business_address"], s2_df["country"]):
        db_lu[eid] = precompute_db_entity(n, a, c)
    del s2_df; gc.collect()
    log.info(f"  Loaded {len(db_lu):,} S2 records.")
    
    log.info("  Loading test_source3.tsv ...")
    s3_df = pd.read_csv(DATA_DIR / "test_source3.tsv", sep="\t", dtype=str, usecols=["entity_id", "business_name", "business_address", "country"]).fillna("")
    s3_df = s3_df[s3_df["entity_id"].isin(needed_db)]
    for eid, n, a, c in zip(s3_df["entity_id"], s3_df["business_name"], s3_df["business_address"], s3_df["country"]):
        db_lu[eid] = precompute_db_entity(n, a, c)
    del s3_df; gc.collect()
    log.info(f"  Total DB entities precomputed: {len(db_lu):,} in {time.time()-t0:.1f}s")
    
    # 4. Load LightGBM model
    log.info(f"Loading LightGBM model from {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    # 5. Batch score all pairs
    log.info(f"Scoring {len(pairs):,} pairs in batches of {BATCH_SIZE:,} ...")
    t_score0 = time.time()
    new_matches = {}
    total_kept = 0
    
    for b_start in range(0, len(pairs), BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, len(pairs))
        batch_pairs = pairs[b_start:b_end]
        
        # Build batch entity lists
        p_e1 = [s1_lu[p[0]] for p in batch_pairs]
        p_e2 = [db_lu[p[1]] for p in batch_pairs]
        
        X_batch = fast_compute_features_batch(p_e1, p_e2, [0.5] * len(p_e1))
        probs = model.predict_proba(X_batch)[:, 1]
        
        for (sid, cid), prob in zip(batch_pairs, probs):
            if prob >= TARGET_THRESHOLD:
                new_matches.setdefault(sid, []).append(cid)
                total_kept += 1
                
        if (b_end // BATCH_SIZE) % 10 == 0 or b_end == len(pairs):
            elapsed = time.time() - t_score0
            rate = b_end / max(1e-5, elapsed)
            log.info(f"  Scored {b_end:,}/{len(pairs):,} pairs ({b_end/len(pairs)*100:.1f}%) | Kept: {total_kept:,} ({total_kept/b_end*100:.1f}%) | {rate:,.0f} pairs/s | {elapsed:.1f}s")
            
    log.info(f"Total pairs kept at threshold {TARGET_THRESHOLD:.3f}: {total_kept:,} / {len(pairs):,} ({total_kept/len(pairs)*100:.1f}%)")
    
    # 6. Write updated matching_results.tsv
    log.info(f"Writing updated {MATCH_PATH} ...")
    with open(MATCH_PATH, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ids:
            m_list = new_matches.get(sid, [])
            f.write(f"{sid}\t{','.join(m_list)}\n")
            
    # Sync to root
    root_match = Path("..") / "matching_results.tsv"
    shutil.copy(MATCH_PATH, root_match)
    log.info(f"Synced updated file to {root_match.resolve()}")
    
    # 7. Validate
    val_script = ROOT_DIR / "utils/validate_submission.py"
    if val_script.exists():
        log.info("Running official validator ...")
        cmd = [
            sys.executable, str(val_script),
            "--matching", str(MATCH_PATH),
            "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"),
            "--test-dir", str(DATA_DIR),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        log.info(f"Validator output:\n{res.stdout}")
        
    # 8. Re-package zip
    pkg_script = ROOT_DIR / "scratch/package_submission.py"
    if pkg_script.exists():
        log.info("Re-packaging submission archive ...")
        subprocess.run([sys.executable, str(pkg_script)], capture_output=True)
        zip_path = ROOT_DIR / "Anonymous Coder_submission.zip"
        root_zip = Path("..") / "Anonymous Coder_submission.zip"
        if zip_path.exists():
            shutil.copy(zip_path, root_zip)
            log.info(f"Synced zip to {root_zip.resolve()}")
            
    log.info(f"COMPLETE! Total runtime: {(time.time()-t_start)/60:.2f} minutes.")

if __name__ == "__main__":
    main()
