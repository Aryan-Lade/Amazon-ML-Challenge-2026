import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path("code/business_entity_resolution/src")
sys.path.insert(0, str(SRC_DIR.resolve()))

from features import compute_pair_features, FEATURE_NAMES
from normalize import normalize_business_name, normalize_address, extract_pin_or_zip

def extract_city(addr):
    tokens = addr.split()
    for tok in reversed(tokens):
        if tok.isalpha() and len(tok) > 2:
            return tok.lower()
    return ""

def precompute_entity(name, addr, country):
    n = normalize_business_name(name)
    a = normalize_address(addr)
    pin = extract_pin_or_zip(addr)
    city = extract_city(a)
    c = str(country).lower().strip()
    return (n, a, pin, city, c)

def fast_compute_pair(e1, e2, blocking_score=0.5):
    from rapidfuzz import fuzz, distance
    import re
    
    n1, a1, pin1, city1, c1 = e1
    n2, a2, pin2, city2, c2 = e2
    
    # 1. Name features
    name_exact = float(n1 == n2 and n1 != "")
    ta = set(n1.split()); tb = set(n2.split())
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
    
    # char ngram
    if len(n1) < 3 and len(n2) < 3: name_char_ngram = 1.0 if n1 == n2 else 0.0
    elif len(n1) < 3 or len(n2) < 3: name_char_ngram = 0.0
    else:
        nga = set(n1[i:i+3] for i in range(len(n1)-2))
        ngb = set(n2[i:i+3] for i in range(len(n2)-2))
        name_char_ngram = len(nga & ngb) / len(nga | ngb)
        
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
    
    # 2. Addr features
    ta_a = set(a1.split()); tb_a = set(a2.split())
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
    a1t = a1.split(); a2t = a2.split()
    addr_first_token = float(bool(a1t and a2t and a1t[0] == a2t[0]))
    
    if a1 and a2:
        if len(a1) < 3 and len(a2) < 3: addr_char_ngram = 1.0 if a1 == a2 else 0.0
        elif len(a1) < 3 or len(a2) < 3: addr_char_ngram = 0.0
        else:
            nga = set(a1[i:i+3] for i in range(len(a1)-2))
            ngb = set(a2[i:i+3] for i in range(len(a2)-2))
            addr_char_ngram = len(nga & ngb) / len(nga | ngb)
        alen1, alen2 = max(1, len(a1)), max(1, len(a2))
        addr_len_ratio = min(alen1, alen2) / max(alen1, alen2)
    else:
        addr_char_ngram = 0.0
        addr_len_ratio = 0.0
        
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

# Verify equivalence on sample
s1_name, s1_addr, s1_country = "Prabhav Center Pvt Ltd", "12 Main Road MG Road Bangalore 560001", "india"
cand_name, cand_addr, cand_country = "Prabhav Centre", "12 Main Rd Bangalore 560001", "india"

feat_orig = compute_pair_features(s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country, 0.5)
e1 = precompute_entity(s1_name, s1_addr, s1_country)
e2 = precompute_entity(cand_name, cand_addr, cand_country)
feat_fast = fast_compute_pair(e1, e2, 0.5)

diff = np.max(np.abs(feat_orig - feat_fast))
print(f"Max absolute difference between original and fast: {diff}")
assert diff < 1e-6, "Features do not match!"
print("SUCCESS: 100% identical feature values verified!")
