"""
train_and_tune.py — Fast, robust LightGBM training and threshold tuning.
"""

import sys
import gc
import json
import time
import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

SRC_DIR = Path("code/business_entity_resolution/src")
sys.path.insert(0, str(SRC_DIR.resolve()))

from features import compute_features_batch, FEATURE_NAMES
from scorer import compute_f05_macro

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

CACHE_DIR = Path(".cache")
MODEL_DIR = Path(".cache/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("dataset/train")
TRAIN_S1 = DATA_DIR / "train_source1.tsv"
TRAIN_S2 = DATA_DIR / "train_source2.tsv"
TRAIN_S3 = DATA_DIR / "train_source3.tsv"
TRAIN_GT = DATA_DIR / "train_ground_truth.tsv"

def load_slim(path):
    log.info(f"Loading {path.name} (slim) ...")
    return pd.read_csv(
        path, sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype=str
    )

def parse_gt(gt_df):
    gt = {}
    for sid, m in zip(gt_df["source1_entity_id"].values, gt_df["matched_entity_ids"].values):
        if pd.notna(m) and str(m).strip():
            gt[sid] = [x.strip() for x in str(m).split(",") if x.strip()]
        else:
            gt[sid] = []
    return gt

def build_lookup_table(df, ids_needed=None):
    if ids_needed is not None:
        mask = df["entity_id"].isin(ids_needed)
        df_sub = df[mask]
    else:
        df_sub = df
    return dict(zip(
        df_sub["entity_id"].values,
        zip(
            df_sub["business_name"].fillna("").values,
            df_sub["business_address"].fillna("").values,
            df_sub["country"].fillna("").values
        )
    ))

def main():
    t_start = time.time()
    log.info("Loading cached validation candidates ...")
    cand_path = CACHE_DIR / "candidates_train_val.pkl"
    with open(cand_path, "rb") as f:
        val_candidates = pickle.load(f)
    log.info(f"Loaded {len(val_candidates):,} candidates from {cand_path}")

    log.info("Loading ground truth ...")
    gt_df = pd.read_csv(TRAIN_GT, sep="\t", dtype=str)
    gt_dict = parse_gt(gt_df)
    del gt_df; gc.collect()
    log.info(f"Ground truth loaded: {len(gt_dict):,} S1 entries")

    val_s1_keys = list(val_candidates.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(val_s1_keys)

    # 30,000 entities for training, 5,000 for tuning threshold
    train_ids = val_s1_keys[:30_000]
    tune_ids  = val_s1_keys[30_000:35_000]
    log.info(f"Training split: {len(train_ids):,} entities, Tuning split: {len(tune_ids):,} entities")

    # Determine needed candidate IDs
    train_cands_needed = set()
    for sid in train_ids:
        train_cands_needed.update(val_candidates.get(sid, []))
    log.info(f"Unique candidate DB IDs needed for training: {len(train_cands_needed):,}")

    tune_cands_needed = set()
    for sid in tune_ids:
        tune_cands_needed.update(val_candidates.get(sid, []))
    log.info(f"Unique candidate DB IDs needed for tuning: {len(tune_cands_needed):,}")

    all_db_needed = train_cands_needed | tune_cands_needed
    all_s1_needed = set(train_ids) | set(tune_ids)

    log.info("Loading source DataFrames ...")
    s1 = load_slim(TRAIN_S1)
    s2 = load_slim(TRAIN_S2)
    s3 = load_slim(TRAIN_S3)

    log.info("Building lookup tables for needed IDs ...")
    s1_lu = build_lookup_table(s1, all_s1_needed)
    s2_lu = build_lookup_table(s2, all_db_needed)
    s3_lu = build_lookup_table(s3, all_db_needed)
    del s1, s2, s3; gc.collect()
    log.info("Source DataFrames freed from memory.")

    def get_rec(eid):
        if eid.startswith("S2-"): return s2_lu.get(eid)
        if eid.startswith("S3-"): return s3_lu.get(eid)
        return s1_lu.get(eid)

    log.info("Constructing ML training pairs ...")
    pos_s1, pos_cand = [], []
    neg_s1, neg_cand = [], []
    neg_ratio = 4

    for sid in tqdm(train_ids, desc="Pair building"):
        cands = val_candidates.get(sid, [])
        if not cands: continue
        gt_set = set(gt_dict.get(sid, []))
        s1r = get_rec(sid)
        if not s1r: continue
        s1_dict = {"business_name": s1r[0], "business_address": s1r[1], "country": s1r[2]}

        true_pos = [c for c in cands if c in gt_set]
        true_neg = [c for c in cands if c not in gt_set]

        for cid in true_pos:
            cr = get_rec(cid)
            if not cr: continue
            pos_s1.append(s1_dict)
            pos_cand.append({"business_name": cr[0], "business_address": cr[1], "country": cr[2]})

        n_neg = min(len(true_neg), neg_ratio * max(1, len(true_pos)))
        if true_neg and n_neg > 0:
            for cid in rng.choice(true_neg, size=n_neg, replace=False):
                cr = get_rec(cid)
                if not cr: continue
                neg_s1.append(s1_dict)
                neg_cand.append({"business_name": cr[0], "business_address": cr[1], "country": cr[2]})

    log.info(f"Training pairs: {len(pos_s1):,} positive, {len(neg_s1):,} negative (Total: {len(pos_s1)+len(neg_s1):,})")

    all_s1 = pos_s1 + neg_s1
    all_cand = pos_cand + neg_cand
    y = np.array([1]*len(pos_s1) + [0]*len(neg_s1), dtype=np.int32)
    del pos_s1, pos_cand, neg_s1, neg_cand; gc.collect()

    log.info("Computing 35 features for training pairs ...")
    t_feat0 = time.time()
    X = compute_features_batch(all_s1, all_cand, [0.5]*len(all_s1))
    log.info(f"Feature matrix computed: {X.shape} in {time.time()-t_feat0:.1f}s")
    del all_s1, all_cand; gc.collect()

    import lightgbm as lgb
    log.info("Training LightGBM model ...")
    scale_pos = (y == 0).sum() / max(1, y.sum())
    model = lgb.LGBMClassifier(
        n_estimators=600,
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
    t_lgb0 = time.time()
    model.fit(X, y)
    log.info(f"LightGBM trained in {time.time()-t_lgb0:.1f}s")

    fi = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    log.info("Top-10 Feature Importances:")
    for fname, fimp in fi[:10]:
        log.info(f"  {fname:20s}: {fimp:.1f}")

    del X, y; gc.collect()

    model_path = MODEL_DIR / "lgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f, protocol=4)
    log.info(f"LightGBM model saved -> {model_path}")

    # Threshold tuning on held-out validation set
    log.info(f"Scoring {len(tune_ids):,} held-out tuning entities ...")
    tune_s1_flat, tune_cand_flat = [], []
    for sid in tune_ids:
        for cid in val_candidates.get(sid, []):
            tune_s1_flat.append(sid)
            tune_cand_flat.append(cid)
    log.info(f"Total tuning candidate pairs: {len(tune_s1_flat):,}")

    bs = 50_000
    all_probs = []
    t_tune0 = time.time()
    for start in tqdm(range(0, len(tune_s1_flat), bs), desc="Tuning scoring"):
        end = min(start + bs, len(tune_s1_flat))
        b_s1 = []
        b_cd = []
        for sid, cid in zip(tune_s1_flat[start:end], tune_cand_flat[start:end]):
            s1r = get_rec(sid)
            cr = get_rec(cid)
            b_s1.append({"business_name": s1r[0] if s1r else "", "business_address": s1r[1] if s1r else "", "country": s1r[2] if s1r else ""})
            b_cd.append({"business_name": cr[0] if cr else "", "business_address": cr[1] if cr else "", "country": cr[2] if cr else ""})
        X_b = compute_features_batch(b_s1, b_cd, [0.5]*(end-start))
        all_probs.extend(model.predict_proba(X_b)[:, 1].tolist())

    log.info(f"Tuning scoring completed in {time.time()-t_tune0:.1f}s")
    all_probs = np.array(all_probs)

    tune_gt = {sid: set(gt_dict.get(sid, [])) for sid in tune_ids}
    best_t = 0.50
    best_f05 = -1.0

    log.info("Sweeping threshold [0.25 to 0.85] for optimal Macro F_0.5 ...")
    for threshold in np.arange(0.25, 0.86, 0.025):
        preds = {}
        for sid, cid, prob in zip(tune_s1_flat, tune_cand_flat, all_probs):
            if prob >= threshold:
                preds.setdefault(sid, []).append(cid)
        full_pred = {sid: preds.get(sid, []) for sid in tune_ids}
        f05 = compute_f05_macro(full_pred, {k: list(v) for k, v in tune_gt.items()})
        log.info(f"  Threshold={threshold:.3f} -> Macro F_0.5 = {f05:.5f}")
        if f05 > best_f05:
            best_f05 = f05
            best_t = float(threshold)

    log.info("=" * 60)
    log.info(f"OPTIMAL THRESHOLD: {best_t:.3f} with Validation Macro F_0.5 = {best_f05:.5f}")
    log.info("=" * 60)

    config = {
        "threshold": best_t,
        "validation_f05": best_f05,
        "n_estimators": 600,
        "n_features": 35,
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = MODEL_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"Config saved -> {config_path}")

    log.info(f"Total train and tune runtime: {(time.time()-t_start)/60:.1f} minutes")

if __name__ == "__main__":
    main()
