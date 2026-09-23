#!/usr/bin/env python3
"""Audit local ESWA submission artifacts without making external claims."""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results" / "nest_ddi" / "submission_readiness_audit.json"


def exists(relative: str) -> bool:
    return (ROOT / relative).exists()


def main() -> None:
    tex = (ROOT / "latex" / "nest_ddi_eswa_anonymous.tex").read_text(encoding="utf-8")
    highlights = (ROOT / "submission" / "nest_ddi_highlights.txt").read_text(encoding="utf-8")
    cover = (ROOT / "docs" / "COVER_LETTER_ESWA.md").read_text(encoding="utf-8")
    manifest = json.loads((ROOT / "docs" / "NEST_DDI_RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    svg_ok = True
    try:
        ET.parse(ROOT / "submission" / "nest_ddi_graphical_abstract.svg")
    except (OSError, ET.ParseError):
        svg_ok = False

    missing_manifest = [item["path"] for item in manifest["files"] if not exists(item["path"])]
    placeholder_tokens = sorted(set(re.findall(r"\[(?:PUBLIC|COMMIT|IMMUTABLE|ARCHIVE|AUTHOR_APPROVED|SHA256)[^\]]*\]", tex)))
    bullet_count = sum(1 for line in highlights.splitlines() if line.startswith("•"))
    cover_letter_consistent = all(token in cover for token in ("0.90526", "0.90437", "benchmark/protocol")) and not any(token in cover for token in ("0.7779", "0.8731", "315,166"))
    result = {
        "manuscript_exists": exists("latex/nest_ddi_eswa_anonymous.tex"),
        "pdf_exists": exists("latex/build_nest/nest_ddi_eswa_anonymous.pdf"),
        "manifest_entries": len(manifest["files"]),
        "manifest_missing_files": missing_manifest,
        "highlights_bullet_count": bullet_count,
        "graphical_abstract_svg_valid": svg_ok,
        "cover_letter_consistent": cover_letter_consistent,
        "manuscript_archive_placeholders": placeholder_tokens,
        "external_archive_required": True,
        "local_submission_ready": not missing_manifest and bullet_count >= 5 and svg_ok and cover_letter_consistent and not placeholder_tokens,
        "note": "Local readiness does not imply that a public repository, license or DOI exists.",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
