# Business Entity Resolution — Reproduction Guide

## Overview
End-to-end two-stage pipeline for matching business records across three noisy sources (S1 -> S2, S3) at massive scale:
1. **Candidate Generation (Blocking)**: Character n-gram TF-IDF + SVD (128 dims) + Per-Country FAISS IVFFlat indexing.
2. **Matching Classifier**: LightGBM pairwise classifier with 35 engineered features trained on ground-truth labeled positive/negative pairs.
3. **Threshold Calibration**: Precision-oriented grid search maximizing Macro $F_{0.5}$ (calibrated optimal threshold: `0.850`).

---

## Directory Structure
```
student_resource/                      <- Run all commands from this root directory
├── dataset/
│   ├── train/                         <- Training source TSVs and ground truth
│   └── test/                          <- Test source TSVs
├── output/                            <- Submission output TSVs (matching_results.tsv, candidate_pairs.tsv)
├── code/business_entity_resolution/
│   ├── src/                           <- Core pipeline code
│   │   ├── normalize.py               <- Text normalization with precompiled regexes & LRU caching
│   │   ├── blocking.py                <- FAISS IVFFlat vector blocking
│   │   ├── features.py                <- 35 pairwise feature extractors
│   │   ├── scorer.py                  <- Macro F_0.5 evaluation metric
│   │   ├── train_and_tune.py          <- Standalone LightGBM training and threshold tuner
│   │   └── pipeline.py                <- End-to-end orchestrator (train, tune, infer, validate)
│   ├── README.md                      <- This reproduction guide
│   └── requirements.txt               <- Pinned Python dependencies
└── utils/validate_submission.py       <- Official submission format validator
```

---

## Environment Setup
```bash
pip install -r code/business_entity_resolution/requirements.txt
```
Key requirements:
- `python>=3.10`
- `lightgbm>=4.0.0`
- `faiss-cpu>=1.7.4`
- `rapidfuzz>=3.0.0`
- `scikit-learn>=1.3.0`
- `pandas>=2.0.0`, `numpy>=1.24.0`, `tqdm>=4.65.0`

---

## Reproduction Commands

### 1. Full Pipeline (Training + Validation + Test Inference + Validation Check)
```bash
python code/business_entity_resolution/src/pipeline.py --mode full
```

### 2. Standalone Training & Threshold Tuning
```bash
python code/business_entity_resolution/src/train_and_tune.py
```
This trains the 35-feature LightGBM model on 298,826 pairs, performs grid search on the held-out validation set, finds the optimal threshold (`0.850`, Macro $F_{0.5} = 0.64964$), and saves artifacts to `.cache/model/`.

### 3. Test Inference Only (Using Pretrained Model)
```bash
python code/business_entity_resolution/src/pipeline.py --mode test
```
Loads `.cache/model/lgbm_model.pkl` and `.cache/model/config.json`, embeds and searches test partitions (France, India, US), scores candidate pairs, writes output files, and runs the official submission validator.

### 4. Official Validator Verification
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## Submission Package Format
```
Anonymous Coder_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```
