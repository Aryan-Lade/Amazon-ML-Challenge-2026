"""
package_submission.py — Packages the final submission zip in the exact requested directory structure.
"""

import os
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(".").resolve()
ZIP_NAME = "Anonymous Coder_submission.zip"
ZIP_PATH = ROOT_DIR / ZIP_NAME

# Required structure:
# Anonymous Coder_submission.zip
# ├── output/
# │   ├── matching_results.tsv
# │   └── candidate_pairs.tsv
# ├── code/
# │   └── business_entity_resolution/
# │       ├── src/
# │       ├── README.md
# │       └── requirements.txt
# └── Documentation_template.md

FILES_TO_ADD = [
    # (source_path_relative_to_root, arcname_in_zip)
    ("output/matching_results.tsv", "output/matching_results.tsv"),
    ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
    ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
    ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
    ("Documentation_template.md", "Documentation_template.md"),
]

SRC_DIR = Path("code/business_entity_resolution/src")

def main():
    print("=" * 60)
    print(f"Creating submission package: {ZIP_NAME}")
    print("=" * 60)

    # Check files exist
    missing = []
    for src_rel, _ in FILES_TO_ADD:
        p = ROOT_DIR / src_rel
        if not p.exists():
            missing.append(src_rel)

    if missing:
        print(f"ERROR: The following required files are missing:\n{missing}")
        sys.exit(1)

    # Create zip
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add primary files
        for src_rel, arcname in FILES_TO_ADD:
            p = ROOT_DIR / src_rel
            zf.write(p, arcname)
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  Added: {arcname:50s} ({size_mb:.2f} MB)")

        # Add all source code in src/
        src_files = list(SRC_DIR.glob("*.py"))
        for sp in src_files:
            arcname = f"code/business_entity_resolution/src/{sp.name}"
            zf.write(sp, arcname)
            print(f"  Added: {arcname:50s}")

    zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print("=" * 60)
    print(f"SUCCESS: Package created -> {ZIP_PATH} ({zip_size_mb:.2f} MB)")
    print("=" * 60)

    # Verify zip contents
    print("\nVerifying archive contents:")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        namelist = zf.namelist()
        for name in sorted(namelist):
            info = zf.getinfo(name)
            print(f"  {name:50s} ({info.file_size:,} bytes)")

    print(f"\nTotal files in archive: {len(namelist)}")

if __name__ == "__main__":
    main()
