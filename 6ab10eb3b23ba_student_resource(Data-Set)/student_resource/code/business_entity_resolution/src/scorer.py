"""
scorer.py — Pairwise matching scorer using a trained LightGBM/XGBoost classifier
or a threshold-based scoring approach as fallback.

Pipeline:
1. Given candidate pairs from blocking.py, build feature matrix
2. Score pairs with a trained model
3. Apply threshold to produce final binary match decisions
4. Output: matching_results dict {s1_id → [matched_ids]}

The model is trained on training data with a validation split
for threshold optimization (maximizing F_0.5 score).
"""

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from features import compute_pair_features, compute_features_batch, FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 10000  # Feature computation batch size


def f_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """F_beta score formula."""
    if precision + recall == 0:
        return 0.0
    beta2 = beta ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def compute_f05_macro(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
) -> float:
    """
    Compute macro-averaged F_0.5 score.
    predictions: {s1_id: [matched_ids]}
    ground_truth: {s1_id: [matched_ids]}  (empty list = singleton)
    """
    scores = []
    for s1_id, gt_ids in ground_truth.items():
        pred_ids = set(predictions.get(s1_id, []))
        gt_set = set(gt_ids)

        if not gt_set and not pred_ids:
            # Correctly predicted singleton
            scores.append(1.0)
        elif not gt_set:
            # False merges on singleton
            scores.append(0.0)
        elif not pred_ids:
            # Missed all matches
            precision = 0.0
            recall = 0.0
            scores.append(f_beta_score(precision, recall, 0.5))
        else:
            tp = len(pred_ids & gt_set)
            precision = tp / len(pred_ids)
            recall = tp / len(gt_set)
            scores.append(f_beta_score(precision, recall, 0.5))

    return np.mean(scores)


def build_training_pairs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    candidates: dict[str, list[str]],
    neg_ratio: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """
    Build feature matrix X and labels y for training.

    Positive pairs: (S1, matched S2/S3) from ground truth
    Negative pairs: (S1, non-matched S2/S3 candidates) up to neg_ratio × positives

    Returns:
        X: feature matrix
        y: binary labels
        pair_ids: list of (s1_id, cand_id) tuples
    """
    # Build lookup dicts for fast access
    log.info("Building lookup dictionaries ...")
    s2_lookup = s2_df.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    s3_lookup = s3_df.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    s1_lookup = s1_df.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")

    def get_record(eid: str) -> Optional[dict]:
        if eid.startswith("S2-"):
            return s2_lookup.get(eid)
        elif eid.startswith("S3-"):
            return s3_lookup.get(eid)
        elif eid.startswith("S1-"):
            return s1_lookup.get(eid)
        return None

    # Parse ground truth
    log.info("Parsing ground truth ...")
    gt_dict: dict[str, set[str]] = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if pd.isna(matched) or str(matched).strip() == "":
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(str(matched).split(","))

    log.info("Building training pairs ...")
    pos_s1, pos_cand, pos_scores = [], [], []
    neg_s1, neg_cand, neg_scores = [], [], []

    rng = np.random.default_rng(42)

    for s1_id, cand_ids in tqdm(candidates.items(), desc="Building pairs"):
        if s1_id not in gt_dict:
            continue

        gt_matched = gt_dict[s1_id]
        s1_rec = get_record(s1_id)
        if s1_rec is None:
            continue

        cand_set = set(cand_ids)
        # True positives: ground truth matches that appear in candidates
        true_pos_in_cands = gt_matched & cand_set
        # False negatives: ground truth matches NOT in candidates (for coverage analysis)
        false_negs = gt_matched - cand_set
        # True negatives available: candidates that are not ground truth matches
        true_negs = [c for c in cand_ids if c not in gt_matched]

        # Add positives from candidates
        for cand_id in true_pos_in_cands:
            cand_rec = get_record(cand_id)
            if cand_rec is None:
                continue
            pos_s1.append(s1_rec)
            pos_cand.append(cand_rec)
            pos_scores.append(0.8)  # placeholder, will be recomputed if needed

        # Sample negatives
        n_neg = min(len(true_negs), neg_ratio * max(1, len(true_pos_in_cands)))
        neg_sample = rng.choice(true_negs, size=n_neg, replace=False).tolist() if true_negs else []
        for cand_id in neg_sample:
            cand_rec = get_record(cand_id)
            if cand_rec is None:
                continue
            neg_s1.append(s1_rec)
            neg_cand.append(cand_rec)
            neg_scores.append(0.3)  # placeholder

    log.info(f"  Positives: {len(pos_s1):,}, Negatives: {len(neg_s1):,}")

    all_s1 = pos_s1 + neg_s1
    all_cand = pos_cand + neg_cand
    all_scores = pos_scores + neg_scores
    y = np.array([1] * len(pos_s1) + [0] * len(neg_s1), dtype=np.int32)

    log.info("Computing features for training pairs ...")
    X = compute_features_batch(all_s1, all_cand, all_scores)

    return X, y, []


def train_model(X: np.ndarray, y: np.ndarray, model_type: str = "lgbm") -> object:
    """Train a gradient boosted classifier."""
    log.info(f"Training {model_type} model on {X.shape[0]:,} pairs ...")
    log.info(f"  Class distribution: {(y==1).sum():,} positive, {(y==0).sum():,} negative")

    if model_type == "lgbm":
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=63,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            )
            model.fit(X, y)
            log.info("  LightGBM model trained.")
            return model
        except ImportError:
            log.warning("LightGBM not available, falling back to XGBoost ...")
            model_type = "xgb"

    if model_type == "xgb":
        try:
            import xgboost as xgb
            scale_pos = (y == 0).sum() / max(1, (y == 1).sum())
            model = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=42,
                verbosity=0,
            )
            model.fit(X, y)
            log.info("  XGBoost model trained.")
            return model
        except ImportError:
            log.warning("XGBoost not available, falling back to RandomForest ...")
            model_type = "rf"

    # Fallback: RandomForest
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
    )
    model.fit(X, y)
    log.info("  GradientBoosting model trained.")
    return model


def find_optimal_threshold(
    model: object,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_pairs: list[tuple[str, str]],
    val_gt: dict[str, set[str]],
) -> float:
    """
    Find the threshold that maximizes F_0.5 on the validation set.
    We search thresholds from 0.1 to 0.9.
    """
    probs = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_f05 = -1.0

    for threshold in np.arange(0.1, 0.95, 0.05):
        preds = probs >= threshold

        # Build predictions dict
        pred_dict: dict[str, list[str]] = {}
        for i, (s1_id, cand_id) in enumerate(val_pairs):
            if preds[i]:
                pred_dict.setdefault(s1_id, []).append(cand_id)

        # Build full prediction dict (including singletons)
        full_pred = {s1_id: pred_dict.get(s1_id, []) for s1_id in val_gt}

        f05 = compute_f05_macro(full_pred, {k: list(v) for k, v in val_gt.items()})
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = threshold
            log.info(f"  Threshold={threshold:.2f}: F_0.5={f05:.4f} ← new best")

    log.info(f"  Best threshold: {best_threshold:.2f}, F_0.5={best_f05:.4f}")
    return best_threshold


def score_candidates(
    model: object,
    threshold: float,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    candidates: dict[str, list[str]],
    batch_size: int = BATCH_SIZE,
) -> dict[str, list[str]]:
    """
    Score all candidate pairs and apply threshold to get final matches.
    Returns {s1_id: [matched_ids]}.
    """
    # Build lookup dicts
    log.info("Building lookup dicts for scoring ...")
    s2_lookup = s2_df.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    s3_lookup = s3_df.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    s1_lookup = s1_df.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")

    def get_record(eid: str) -> Optional[dict]:
        if eid.startswith("S2-"):
            return s2_lookup.get(eid)
        elif eid.startswith("S3-"):
            return s3_lookup.get(eid)
        elif eid.startswith("S1-"):
            return s1_lookup.get(eid)
        return None

    # Flatten all pairs
    log.info("Flattening candidate pairs for scoring ...")
    all_s1_ids = []
    all_cand_ids = []
    for s1_id, cand_ids in candidates.items():
        for cand_id in cand_ids:
            all_s1_ids.append(s1_id)
            all_cand_ids.append(cand_id)

    log.info(f"  Total pairs to score: {len(all_s1_ids):,}")

    # Compute features in batches
    all_probs = []
    for start in tqdm(range(0, len(all_s1_ids), batch_size), desc="Scoring batches"):
        end = min(start + batch_size, len(all_s1_ids))
        batch_s1 = []
        batch_cand = []
        batch_scores = []
        for i in range(start, end):
            s1_rec = get_record(all_s1_ids[i]) or {"business_name": "", "business_address": "", "country": ""}
            cand_rec = get_record(all_cand_ids[i]) or {"business_name": "", "business_address": "", "country": ""}
            batch_s1.append(s1_rec)
            batch_cand.append(cand_rec)
            batch_scores.append(0.5)  # placeholder blocking score

        X_batch = compute_features_batch(batch_s1, batch_cand, batch_scores)
        probs = model.predict_proba(X_batch)[:, 1]
        all_probs.extend(probs.tolist())

    # Apply threshold
    log.info(f"Applying threshold {threshold:.3f} ...")
    matches: dict[str, list[str]] = {}
    for i, (s1_id, cand_id) in enumerate(zip(all_s1_ids, all_cand_ids)):
        if all_probs[i] >= threshold:
            matches.setdefault(s1_id, []).append(cand_id)

    log.info(f"  Entities with ≥1 match: {len(matches):,}")
    return matches
