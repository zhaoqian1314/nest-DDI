#!/usr/bin/env python3
"""Finalize author-supplied archive metadata without inventing identifiers."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(r"D:\BI\DDI")
TEX = ROOT / "latex" / "nest_ddi_eswa_anonymous.tex"
TEMPLATE = ROOT / "docs" / "NEST_DDI_ARCHIVE_METADATA_TEMPLATE.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--license", required=True, dest="license_id")
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--doi", required=True)
    parser.add_argument("--release-date", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def reject_placeholder(value: str, label: str) -> None:
    if not value.strip() or "[" in value or "]" in value or value.startswith("http://example"):
        raise ValueError(f"{label} is empty or still contains a placeholder")


def main() -> None:
    args = parse_args()
    fields = {
        "repository_url": args.repository_url,
        "commit": args.commit,
        "license_spdx": args.license_id,
        "archive_url": args.archive_url,
        "doi": args.doi,
        "release_date": args.release_date,
    }
    for label, value in fields.items():
        reject_placeholder(value, label)
    if not re.match(r"^https?://", args.repository_url) or not re.match(r"^https?://", args.archive_url):
        raise ValueError("repository and archive URLs must be resolvable HTTP(S) URLs")
    if not re.match(r"^10\.\d{4,9}/", args.doi):
        raise ValueError("doi must use the DOI form 10.xxxx/...")

    text = TEX.read_text(encoding="utf-8")
    old = "An anonymized reproducibility contract accompanies this manuscript and lists reconstruction order, seeds, derived panels, audits and interpretation constraints. FAERS/AEMS raw data remain subject to their original FDA distribution terms. A versioned public archive, license and release DOI will be required before external submission. The anonymized manuscript does not identify authors or institutions."
    new = (f"Code, environment specification, reconstruction instructions, processing audits and derived metric summaries are available at {args.repository_url}, "
           f"archived as {args.doi} ({args.archive_url}) under {args.license_id}; release commit {args.commit} dated {args.release_date}. "
           "Raw FAERS/AEMS quarterly files are obtained from the FDA under their original distribution terms and are not redistributed here. "
           "The anonymized manuscript does not identify authors or institutions.")
    if old not in text:
        raise RuntimeError("expected future-tense data/code paragraph was not found; refusing to edit")
    if args.dry_run:
        print(new)
        return
    TEX.write_text(text.replace(old, new, 1), encoding="utf-8")
    TEMPLATE.write_text(TEMPLATE.read_text(encoding="utf-8") + "\n\n## Finalized metadata\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in fields.items()), encoding="utf-8")
    subprocess.run(["python", "scripts/build_nest_release_manifest.py"], cwd=ROOT, check=True)
    print("Archive metadata finalized; recompile the manuscript and rerun the submission audit.")


if __name__ == "__main__":
    main()
