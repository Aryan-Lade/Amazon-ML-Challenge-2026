import sys
import gc
import json
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

SRC_DIR = Path("code/business_entity_resolution/src")
sys.path.insert(0, str(SRC_DIR.resolve()))

from features import compute_features_batch
from scorer import compute_f05_macro
from normalize import normalize_business_name

with open(".cache/candidates_train_val.pkl", "rb") as f:
    val_candidates = pickle.load(f)

gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str)
gt_dict = {}
for sid, m in zip(gt_df["source1_entity_id"].values, gt_df["matched_entity_ids"].values):
    if pd.notna(m) and str(m).strip():
        gt_dict[sid] = [x.strip() for x in str(m).split(",") if x.strip()]
    else:
        gt_dict[sid] = []
del gt_df; gc.collect()

val_s1_keys = list(val_candidates.keys())
rng = np.random.default_rng(42)
rng.shuffle(val_s1_keys)
tune_ids = val_s1_keys[30_000:35_000]

tune_cands_needed = set()
for sid in tune_ids:
    tune_cands_needed.update(val_candidates.get(sid, []))
all_needed = set(tune_ids) | tune_cands_needed

def load_slim_lookup(path, ids_needed):
    df = pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"], dtype=str)
    mask = df["entity_id"].isin(ids_needed)
    sub = df[mask]
    return dict(zip(
        sub["entity_id"].values,
        zip(sub["business_name"].fillna("").values, sub["business_address"].fillna("").values, sub["country"].fillna("").values)
    ))

s1_lu = load_slim_lookup("dataset/train/train_source1.tsv", all_needed)
s2_lu = load_slim_lookup("dataset/train/train_source2.tsv", all_needed)
s3_lu = load_slim_lookup("dataset/train/train_source3.tsv", all_needed)

def get_rec(eid):
    return s1_lu.get(eid) or s2_lu.get(eid) or s3_lu.get(eid)

with open(".cache/model/lgbm_model.pkl", "rb") as f:
    model = pickle.load(f)

tune_s1_flat, tune_cand_flat = [], []
for sid in tune_ids:
    for cid in val_candidates.get(sid, []):
        tune_s1_flat.append(sid)
        tune_cand_flat.append(cid)

bs = 50_000
all_probs = []
for start in range(0, len(tune_s1_flat), bs):
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

all_probs = np.array(all_probs)
tune_gt = {sid: set(gt_dict.get(sid, [])) for sid in tune_ids}

print("="*60)
print(f"{'Threshold':<12} | {'Macro F0.5':<12} | {'Precision':<12} | {'Recall':<12} | {'Avg Matches':<12}")
print("="*60)

for threshold in [0.70, 0.75, 0.80, 0.825, 0.85, 0.875, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
    preds = {}
    for sid, cid, prob in zip(tune_s1_flat, tune_cand_flat, all_probs):
        if prob >= threshold:
            preds.setdefault(sid, []).append(cid)
    full_pred = {sid: preds.get(sid, []) for sid in tune_ids}
    
    # Calculate detailed P, R, F0.5
    f05_list, p_list, r_list = [], [], []
    for sid in tune_ids:
        p_set = set(full_pred.get(sid, []))
        t_set = tune_gt.get(sid, set())
        if not t_set and not p_set:
            f05_list.append(1.0); p_list.append(1.0); r_list.append(1.0)
        elif not t_set and p_set:
            f05_list.append(0.0); p_list.append(0.0); r_list.append(1.0)
        elif t_set and not p_set:
            f05_list.append(0.0); p_list.append(1.0); r_list.append(0.0)
        else:
            tp = len(p_set & t_set)
            p = tp / len(p_set)
            r = tp / len(t_set)
            beta_sq = 0.25
            denom = beta_sq * p + r
            f05 = (1 + beta_sq) * p * r / denom if denom > 0 else 0.0
            f05_list.append(f05); p_list.append(p); r_list.append(r)
            
    avg_m = np.mean([len(full_pred.get(sid, [])) for sid in tune_ids])
    print(f"{threshold:<12.3f} | {np.mean(f05_list):<12.5f} | {np.mean(p_list):<12.5f} | {np.mean(r_list):<12.5f} | {avg_m:<12.3f}")
