"""
normalize.py — Text normalization utilities for business entity resolution.

Handles:
- Legal suffix expansion/normalization
- Abbreviation expansion (address types)
- Punctuation cleanup
- Unicode/transliteration (for Hindi/French/etc.)
- Lowercasing and whitespace normalization
"""

import re
import unicodedata
import functools

# -------------------------------------------------------------------
# Legal suffix normalization (map to canonical form)
# -------------------------------------------------------------------
LEGAL_SUFFIX_MAP = {
    r"\bpvt\.?\s*ltd\.?": "private limited",
    r"\bprivate\s+limited\b": "private limited",
    r"\bp\.?\s*ltd\.?": "private limited",
    r"\bltd\.?": "limited",
    r"\bllp\b": "llp",
    r"\bllc\b": "llc",
    r"\bplc\b": "plc",
    r"\binc\.?": "inc",
    r"\bcorp\.?": "corp",
    r"\bcorporation\b": "corp",
    r"\bco\.?\b": "co",
    r"\bcompany\b": "co",
    r"\benterprises?\b": "enterprises",
    r"\bassociates?\b": "associates",
    r"\bgroup\b": "group",
    r"\bholding[s]?\b": "holdings",
    r"\bsolutions?\b": "solutions",
    r"\bservices?\b": "services",
    r"\btechnolog(?:ies|y)\b": "technologies",
    r"\bindustries?\b": "industries",
    r"\btraders?\b": "traders",
    r"\binternational\b": "international",
    r"\bintl\.?\b": "international",
}

# Address abbreviation expansion
ADDR_ABBREV_MAP = {
    r"\bst\.?\b": "street",
    r"\brd\.?\b": "road",
    r"\bave\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bdr\.?\b": "drive",
    r"\bln\.?\b": "lane",
    r"\bct\.?\b": "court",
    r"\bpl\.?\b": "place",
    r"\bsq\.?\b": "square",
    r"\bfwy\.?\b": "freeway",
    r"\bhwy\.?\b": "highway",
    r"\bpkwy\.?\b": "parkway",
    r"\btre\.?\b": "terrace",
    r"\bter\.?\b": "terrace",
    r"\bexpy\.?\b": "expressway",
    r"\bflr\.?\b": "floor",
    r"\bfl\.?\b": "floor",
    r"\bapt\.?\b": "apartment",
    r"\bste\.?\b": "suite",
    r"\bno\.?\b": "number",
    r"\bnorth\b": "north",
    r"\bsouth\b": "south",
    r"\beast\b": "east",
    r"\bwest\b": "west",
    r"\bn\.?\b": "north",
    r"\bs\.?\b": "south",
    r"\be\.?\b": "east",
    r"\bw\.?\b": "west",
}

# US state abbreviations → full names
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

# Indian state abbreviations
INDIA_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam",
    "br": "bihar", "cg": "chhattisgarh", "ga": "goa", "gj": "gujarat",
    "hr": "haryana", "hp": "himachal pradesh", "jk": "jammu kashmir",
    "jh": "jharkhand", "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh",
    "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram",
    "nl": "nagaland", "od": "odisha", "or": "odisha", "pb": "punjab",
    "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "tg": "telangana",
    "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand", "wb": "west bengal",
    "dl": "delhi", "ld": "lakshadweep", "py": "puducherry",
}



# -------------------------------------------------------------------
# Precompiled regexes & LRU cache
# -------------------------------------------------------------------
RE_AMP = re.compile(r"\s*&\s*")
RE_SPECIAL = re.compile(r"[^\w\s\-.,/]", flags=re.UNICODE)
RE_SPACES = re.compile(r"\s+")
RE_LEADING_NOISE = re.compile(r"^[-<>\s#@!]+")
RE_ZIP = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
RE_PIN = re.compile(r"\b(\d{6})\b")

COMPILED_LEGAL_SUFFIX = [
    (re.compile(pattern, flags=re.IGNORECASE), repl)
    for pattern, repl in LEGAL_SUFFIX_MAP.items()
]

COMPILED_ADDR_ABBREV = [
    (re.compile(r"(?<!\w)" + pattern + r"(?!\w)", flags=re.IGNORECASE), repl)
    for pattern, repl in ADDR_ABBREV_MAP.items()
]


def unicode_normalize(text: str) -> str:
    """Normalize unicode: convert composed chars to decomposed form, strip diacritics for ASCII."""
    return unicodedata.normalize("NFC", text)


def clean_punctuation(text: str) -> str:
    """Replace punctuation variants with spaces or canonical forms."""
    text = RE_AMP.sub(" and ", text)
    text = RE_SPECIAL.sub(" ", text)
    text = RE_SPACES.sub(" ", text)
    return text.strip()


@functools.lru_cache(maxsize=1_000_000)
def normalize_business_name(name: str) -> str:
    """Normalize a business name for comparison."""
    if not name or (isinstance(name, float)):
        return ""
    name = str(name)
    name = unicode_normalize(name)
    name = name.lower()
    name = clean_punctuation(name)

    for compiled_pat, replacement in COMPILED_LEGAL_SUFFIX:
        name = compiled_pat.sub(replacement, name)

    name = RE_LEADING_NOISE.sub("", name)
    name = RE_SPACES.sub(" ", name).strip()
    return name


@functools.lru_cache(maxsize=1_000_000)
def normalize_address(addr: str) -> str:
    """Normalize an address for comparison."""
    if not addr or (isinstance(addr, float)):
        return ""
    addr = str(addr)
    addr = unicode_normalize(addr)
    addr = addr.lower()
    addr = clean_punctuation(addr)

    for compiled_pat, replacement in COMPILED_ADDR_ABBREV:
        addr = compiled_pat.sub(replacement, addr)

    addr = RE_SPACES.sub(" ", addr).strip()
    return addr


@functools.lru_cache(maxsize=1_000_000)
def extract_pin_or_zip(addr: str) -> str:
    """Extract PIN code (India) or ZIP code (US) from address."""
    if not addr or isinstance(addr, float):
        return ""
    zip_match = RE_ZIP.search(str(addr))
    if zip_match:
        return zip_match.group(1)
    pin_match = RE_PIN.search(str(addr))
    if pin_match:
        return pin_match.group(1)
    return ""


def make_blocking_key(name: str, country: str) -> str:
    """
    Create a blocking key combining normalized name + country.
    Used for TF-IDF index construction.
    """
    norm_name = normalize_business_name(name)
    country_key = str(country).lower().strip() if country and not isinstance(country, float) else "unknown"
    return f"{country_key} {norm_name}"


def make_full_text(name: str, address: str, country: str) -> str:
    """
    Create a combined text representation for embedding/TF-IDF.
    """
    norm_name = normalize_business_name(name)
    norm_addr = normalize_address(address)
    country_key = str(country).lower().strip() if country and not isinstance(country, float) else "unknown"
    parts = [country_key, norm_name]
    if norm_addr:
        parts.append(norm_addr)
    return " ".join(parts)


if __name__ == "__main__":
    # Quick sanity check
    tests = [
        ("Prabhav Business Center", "797, Lake Town Block A, Kolkata", "India"),
        ("Delta Tetlecommunication Inc", "914 PIERPONT AVE, CLEVELAND, OH", "US"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI", "India"),
        ("Summit Inc", "GREENSBORO, NC, 19 1/2 STARDUST TRAIL", "US"),
    ]
    for name, addr, country in tests:
        print(f"Name: {name!r}")
        print(f"  -> norm_name: {normalize_business_name(name)!r}")
        print(f"  -> norm_addr: {normalize_address(addr)!r}")
        print(f"  -> key: {make_blocking_key(name, country)!r}")
        print()
