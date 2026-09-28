"""
test_fast_scoring_bench.py -- Benchmark pre-normalized feature extraction speed.
"""
import time
import pickle
import numpy as np
from pathlib import Path
from rapidfuzz import fuzz, distance

def bench():
    print("Testing pre-computed features speed...")
    n_pairs = 100_000
    
    # Pre-normalized data
    n1, n2 = "state bank of india", "sbi"
    a1, a2 = "12 main road suite 4", "12 main road suite 4"
    pin1, pin2 = "110001", "110001"
    city1, city2 = "delhi", "delhi"
    c1, c2 = "india", "india"
    
    t0 = time.time()
    for _ in range(n_pairs):
        # 35 features computed directly on pre-normalized strings
        name_exact = float(n1 == n2 and n1 != "")
        name_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
        name_set = fuzz.token_set_ratio(n1, n2) / 100.0
        name_partial = fuzz.partial_ratio(n1, n2) / 100.0
        name_ratio = fuzz.ratio(n1, n2) / 100.0
        name_jw = distance.JaroWinkler.normalized_similarity(n1, n2)
        name_prefix = float(n1[:3] == n2[:3] and len(n1) >= 3 and len(n2) >= 3)
        len1, len2 = max(1, len(n1)), max(1, len(n2))
        name_len_ratio = min(len1, len2) / max(len1, len2)
        
        country_match = float(c1 == c2)
        addr_sort = fuzz.token_sort_ratio(a1, a2) / 100.0
        addr_set = fuzz.token_set_ratio(a1, a2) / 100.0
        addr_partial = fuzz.partial_ratio(a1, a2) / 100.0
        pin_match = float(pin1 != "" and pin1 == pin2)
        pin_present = float(pin1 != "" and pin2 != "")
        city_match = float(city1 == city2 and city1 != "")
        name_addr_product = name_sort * addr_sort
        
    elapsed = time.time() - t0
    rate = n_pairs / elapsed
    print(f"Scored {n_pairs:,} pairs in {elapsed:.3f}s ({rate:,.0f} pairs/sec)")

if __name__ == "__main__":
    bench()
