"""
package_submission.py — Packages the final submission zip: Anonymous_Coders_submission.zip

Verifies:
  1. output/matching_results.tsv exists
  2. output/candidate_pairs.tsv exists
  3. Runs utils/validate_submission.py to ensure 100% compliance
  4. Packs into Anonymous_Coders_submission.zip:
       ├── output/
       │   ├── matching_results.tsv
       │   └── candidate_pairs.tsv
       ├── code/
       │   └── business_entity_resolution/
       │       ├── src/
       │       ├── README.md
       │       └── requirements.txt
       └── Documentation_template.md
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "output"
    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"
    doc_file = root / "Documentation_template.md"
    code_dir = root / "code" / "business_entity_resolution"
    zip_path = root / "Anonymous_Coders_submission.zip"

    print("=" * 60)
    print("SUBMISSION PACKAGER — Anonymous_Coders")
    print("=" * 60)

    # 1. Check required output files
    if not matching_file.exists():
        print(f"[-] ERROR: {matching_file} does not exist yet.")
        sys.exit(1)
    if not candidate_file.exists():
        print(f"[-] ERROR: {candidate_file} does not exist yet.")
        sys.exit(1)
    if not doc_file.exists():
        print(f"[-] ERROR: {doc_file} does not exist.")
        sys.exit(1)
    if not code_dir.exists():
        print(f"[-] ERROR: {code_dir} does not exist.")
        sys.exit(1)

    print("[+] All required components found on disk.")

    # 2. Run validator
    validator_script = root / "utils" / "validate_submission.py"
    if validator_script.exists():
        print("[*] Running official validate_submission.py ...")
        cmd = [
            sys.executable,
            str(validator_script),
            "--matching", str(matching_file),
            "--candidate", str(candidate_file),
            "--test-dir", str(root / "dataset" / "test")
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        if res.returncode != 0:
            print("[-] ERROR: validate_submission.py failed! Fix errors before packaging.")
            sys.exit(res.returncode)
        print("[+] Validation passed successfully (exit 0)!")

    # 3. Create zip
    print(f"[*] Creating archive: {zip_path.name} ...")
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add output files
        zf.write(matching_file, arcname="output/matching_results.tsv")
        zf.write(candidate_file, arcname="output/candidate_pairs.tsv")
        print("  + output/matching_results.tsv")
        print("  + output/candidate_pairs.tsv")

        # Add documentation
        zf.write(doc_file, arcname="Documentation_template.md")
        print("  + Documentation_template.md")

        # Add code directory (excluding __pycache__ and .cache)
        for p in code_dir.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and ".cache" not in p.parts:
                arcname = p.relative_to(root).as_posix()
                zf.write(p, arcname=arcname)
                print(f"  + {arcname}")

    print("=" * 60)
    print(f"[+] Successfully created: {zip_path}")
    print(f"[+] Zip file size: {zip_path.stat().st_size / (1024 * 1024):.2f} MB")
    print("=" * 60)

    # List zip structure
    print("[*] Inspecting zip file structure:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename} ({info.file_size:,} bytes)")

if __name__ == "__main__":
    main()
