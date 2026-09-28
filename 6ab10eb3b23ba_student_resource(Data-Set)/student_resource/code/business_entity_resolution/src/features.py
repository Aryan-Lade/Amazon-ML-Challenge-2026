"""
features.py -- Feature engineering for pairwise business entity matching (v2, 35 features).
"""

import re
import sys
import numpy as np
from pathlib import Path
from rapidfuzz import fuzz, distance

sys.path.insert(0, str(Path(__file__).parent))
from normalize import normalize_business_name, normalize_address, extract_pin_or_zip


def token_jaccard(a, b):
    ta = set(a.split()); tb = set(b.split())
    if not ta and not tb: return 1.0
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)


def token_overlap(a, b):
    ta = set(a.split()); tb = set(b.split())
    if not ta and not tb: return 1.0
    if not ta or not tb: return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def char_ngram_overlap(a, b, n=3):
    if len(a) < n and len(b) < n: return 1.0 if a == b else 0.0
    if len(a) < n or len(b) < n: return 0.0
    nga = set(a[i:i+n] for i in range(len(a)-n+1))
    ngb = set(b[i:i+n] for i in range(len(b)-n+1))
    return len(nga & ngb) / len(nga | ngb)


def extract_city(addr):
    tokens = addr.split()
    for tok in reversed(tokens):
        if tok.isalpha() and len(tok) > 2:
            return tok.lower()
    return ""


def compute_pair_features(s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country, blocking_score=0.0):
    n1 = normalize_business_name(s1_name)
    n2 = normalize_business_name(cand_name)
    a1 = normalize_address(s1_addr)
    a2 = normalize_address(cand_addr)
    c1 = str(s1_country).lower().strip()
    c2 = str(cand_country).lower().strip()

    name_exact = float(n1 == n2 and n1 != "")
    name_jaccard = token_jaccard(n1, n2)
    name_overlap = token_overlap(n1, n2)
    name_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_set = fuzz.token_set_ratio(n1, n2) / 100.0
    name_partial = fuzz.partial_ratio(n1, n2) / 100.0
    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_jw = distance.JaroWinkler.normalized_similarity(n1, n2)
    name_prefix = float(n1[:3] == n2[:3] and len(n1) >= 3 and len(n2) >= 3)
    name_char_ngram = char_ngram_overlap(n1, n2, 3)
    len1, len2 = max(1, len(n1)), max(1, len(n2))
    name_len_ratio = min(len1, len2) / max(len1, len2)
    n1t = n1.split(); n2t = n2.split()
    name_first_word = float(bool(n1t and n2t and n1t[0] == n2t[0]))
    n1d = set(re.findall(r"\d+", n1)); n2d = set(re.findall(r"\d+", n2))
    name_digit_match = (float(bool(n1d & n2d)) if (n1d and n2d) else 1.0)
    if n1 and n2:
        lev = distance.Levenshtein.distance(n1[:100], n2[:100])
        name_levenshtein = 1.0 - lev / max(1, max(len(n1[:100]), len(n2[:100])))
    else:
        name_levenshtein = 0.0

    country_match = (float(c1 == c2) if (c1 and c2) else 0.5)

    addr_jaccard = token_jaccard(a1, a2)
    addr_overlap = token_overlap(a1, a2)
    addr_sort = fuzz.token_sort_ratio(a1, a2) / 100.0
    addr_set = fuzz.token_set_ratio(a1, a2) / 100.0
    addr_partial = fuzz.partial_ratio(a1, a2) / 100.0
    pin1 = extract_pin_or_zip(s1_addr); pin2 = extract_pin_or_zip(cand_addr)
    pin_match = float(pin1 != "" and pin1 == pin2)
    pin_present = float(pin1 != "" and pin2 != "")
    addr1_empty = float(a1 == ""); addr2_empty = float(a2 == "")
    city1 = extract_city(a1); city2 = extract_city(a2)
    city_match = float(city1 == city2 and city1 != "")
    a1t = a1.split(); a2t = a2.split()
    addr_first_token = float(bool(a1t and a2t and a1t[0] == a2t[0]))
    addr_char_ngram = char_ngram_overlap(a1, a2, 3) if (a1 and a2) else 0.0
    alen1, alen2 = max(1, len(a1)), max(1, len(a2))
    addr_len_ratio = min(alen1, alen2) / max(alen1, alen2) if (a1 and a2) else 0.0
    a1d = set(re.findall(r"\d+", a1)); a2d = set(re.findall(r"\d+", a2))
    addr_digit_match = float(bool(a1d & a2d)) if (a1d and a2d) else 0.0

    name_addr_product = name_sort * addr_sort
    blocking_x_name = blocking_score * name_sort
    name_set_x_addr_set = name_set * addr_set
    pin_x_name = pin_match * name_set
    exact_x_country = name_exact * country_match

    return np.array([
        name_exact, name_jaccard, name_overlap, name_sort, name_set,
        name_partial, name_ratio, name_jw, name_prefix, name_char_ngram,
        name_len_ratio, name_first_word, name_digit_match, name_levenshtein,
        country_match,
        addr_jaccard, addr_overlap, addr_sort, addr_set, addr_partial,
        pin_match, pin_present, addr1_empty, addr2_empty, city_match,
        addr_first_token, addr_char_ngram, addr_len_ratio, addr_digit_match,
        blocking_score,
        name_addr_product, blocking_x_name, name_set_x_addr_set, pin_x_name,
        exact_x_country,
    ], dtype=np.float32)


FEATURE_NAMES = [
    "name_exact", "name_jaccard", "name_overlap", "name_sort", "name_set",
    "name_partial", "name_ratio", "name_jw", "name_prefix", "name_char_ngram",
    "name_len_ratio", "name_first_word", "name_digit_match", "name_levenshtein",
    "country_match",
    "addr_jaccard", "addr_overlap", "addr_sort", "addr_set", "addr_partial",
    "pin_match", "pin_present", "addr1_empty", "addr2_empty", "city_match",
    "addr_first_token", "addr_char_ngram", "addr_len_ratio", "addr_digit_match",
    "blocking_score",
    "name_addr_product", "blocking_x_name", "name_set_x_addr_set", "pin_x_name",
    "exact_x_country",
]

assert len(FEATURE_NAMES) == 35, f"Expected 35, got {len(FEATURE_NAMES)}"


def compute_features_batch(s1_records, cand_records, blocking_scores):
    n = len(s1_records)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    for i in range(n):
        X[i] = compute_pair_features(
            s1_records[i].get("business_name", ""),
            s1_records[i].get("business_address", ""),
            s1_records[i].get("country", ""),
            cand_records[i].get("business_name", ""),
            cand_records[i].get("business_address", ""),
            cand_records[i].get("country", ""),
            blocking_scores[i],
        )
    return X


if __name__ == "__main__":
    f = compute_pair_features(
        "Prabhav Business Center Pvt Ltd",
        "797, Lake Town Block A, Kolkata 700089",
        "India",
        "Prabhav Biz Center Private Limited",
        "797 Lake Town, Block-A, Kolkata, WB 700089",
        "India",
        0.85,
    )
    n = len(f)
    print(f"Features ({n} total):")
    for name, val in zip(FEATURE_NAMES, f):
        print(f"  {name}: {val:.4f}")
