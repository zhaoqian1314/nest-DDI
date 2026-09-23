#!/usr/bin/env python3
"""Create a checksum manifest for the code-and-metadata release package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(r"D:\BI\DDI")
OUTPUT = ROOT / "docs" / "NEST_DDI_RELEASE_MANIFEST.json"
FILES = [
    "environment_temporal.yml",
    "scripts/build_nest_pilot.py",
    "scripts/build_nest_prediction_panel.py",
    "scripts/run_nest_final_comparison.py",
    "scripts/run_nest_extended_baselines.py",
    "scripts/build_nest_pv_sufficient_statistics.py",
    "scripts/build_nest_first_event_discovery.py",
    "scripts/run_nest_first_event_baselines.py",
    "scripts/run_nest_deep_learning_baselines.py",
    "scripts/run_nest_graph_deep_baselines.py",
    "scripts/audit_graph_deep_bootstrap.py",
    "scripts/summarize_deep_seed_sensitivity.py",
    "scripts/audit_submission_readiness.py",
    "scripts/finalize_archive_metadata.py",
    "scripts/optimize_nest_first_task.py",
    "scripts/audit_nest_first_optimized_bootstrap.py",
    "scripts/evaluate_nest_graph_fusion.py",
    "scripts/audit_all_baseline_leaderboard.py",
    "scripts/select_graph_fusion_weight.py",
    "scripts/finalize_validated_fusion_from_existing.py",
    "scripts/run_later_graphsage_baseline.py",
    "scripts/run_nest_2026_baselines.py",
    "scripts/run_drift_temporal_reproduction.py",
    "scripts/run_drugdagt_temporal_reproduction.py",
    "scripts/run_smrddi_temporal_reproduction.py",
    "scripts/run_nest_later_temporal_test.py",
    "scripts/audit_nest_drug_cluster_bootstrap.py",
    "scripts/evaluate_later_operational_gain.py",
    "scripts/audit_later_role_strata.py",
    "scripts/audit_later_calibration.py",
    "scripts/audit_drugbank_reference_independence.py",
    "scripts/audit_openfda_label_concordance.py",
    "figures/temporal-risk-set-lineage.drawio",
    "latex/nest_ddi_eswa_anonymous.tex",
    "submission/nest_ddi_highlights.txt",
    "submission/nest_ddi_graphical_abstract_brief.md",
    "submission/nest_ddi_graphical_abstract.svg",
    "docs/NEST_DDI_REPRODUCIBILITY.md",
    "docs/COVER_LETTER_ESWA.md",
    "docs/SUBMISSION_PACKAGE.md",
    "docs/NEST_DDI_ESWA_PRE_REVIEW_V2.md",
    "docs/NEST_DDI_ESWA_PRE_REVIEW_V3.md",
    "docs/NEST_DDI_ESWA_PRE_REVIEW_V4.md",
    "docs/NEST_DDI_ESWA_PRE_REVIEW_V5.md",
    "docs/NEST_DDI_ESWA_PRE_REVIEW_V6.md",
    "docs/NEST_DDI_ESWA_PRE_REVIEW_V7.md",
    "docs/NEST_DDI_ESWA_PRE_REVIEW_V8.md",
    "docs/NEST_DDI_ESWA_REVISION_RESPONSE_MATRIX_V2.md",
    "docs/NEST_DDI_ESWA_RESPONSE_TO_REVIEWERS_FINAL.md",
    "docs/NEST_DDI_ESWA_ACCEPTANCE_READINESS_FINAL.md",
    "docs/NEST_DDI_ARCHIVE_RELEASE_INSTRUCTIONS.md",
    "docs/NEST_DDI_FINAL_SUBMISSION_AUDIT.md",
    "docs/NEST_DDI_ARCHIVE_METADATA_TEMPLATE.md",
    "docs/NEST_DDI_DEEP_BASELINE_AUDIT.md",
    "docs/NEST_DDI_GRAPH_BASELINE_AUDIT.md",
    "docs/NEST_DDI_SEED_SENSITIVITY_AUDIT.md",
    "docs/NEST_DDI_OPTIMIZATION_AUDIT.md",
    "docs/NEST_DDI_LEADERBOARD_AUDIT.md",
    "docs/NEST_DDI_LATER_GRAPH_AUDIT.md",
    "docs/NEST_DDI_MAIN_MODEL_RERUN_AUDIT.md",
    "docs/NEST_DDI_RESULT_CLEANUP_AUDIT.md",
    "docs/LATEST_2026_DDI_BASELINES_RESEARCH.md",
    "results/nest_ddi/later_temporal_release_local_metrics.json",
    "results/nest_ddi/final_comparison_release_local_metrics.json",
    "results/nest_ddi/rerun_main_2026_metrics.json",
    "results/nest_ddi/extended_baseline_release_local_metrics.json",
    "data/processed/nest_pv_sufficient_statistics_audit_4q_release.json",
    "data/processed/nest_first_event_discovery_2024q3_release_local_audit.json",
    "data/processed/nest_first_event_discovery_2024q4_release_local_audit.json",
    "data/processed/nest_first_event_discovery_2025q1_release_local_audit.json",
    "results/nest_ddi/first_event_baseline_release_local_metrics.json",
    "results/nest_ddi/deep_baselines_release_local_metrics.json",
    "results/nest_ddi/graph_deep_baselines_release_local_metrics.json",
    "results/nest_ddi/graph_deep_baselines_release_local_bootstrap.json",
    "results/nest_ddi/deep_graph_seed_sensitivity.json",
    "results/nest_ddi/deep_ft_seed7_metrics.json",
    "results/nest_ddi/graph_sage_seed7_metrics.json",
    "results/nest_ddi/submission_readiness_audit.json",
    "results/nest_ddi/nest_first_optimized_release_local_metrics.json",
    "results/nest_ddi/nest_first_optimized_release_local_metrics.npz",
    "results/nest_ddi/nest_first_optimized_release_local_bootstrap.json",
    "results/nest_ddi/nest_graph_fusion_release_local_metrics.json",
    "results/nest_ddi/nest_graph_fusion_release_local_metrics.npz",
    "results/nest_ddi/all_baseline_leaderboard_audit.json",
    "results/nest_ddi/graph_fusion_weight_selection.json",
    "results/nest_ddi/graph_fusion_weight_selection.npz",
    "results/nest_ddi/nest_graph_fusion_validated_replay_release_local_metrics.json",
    "results/nest_ddi/nest_graph_fusion_validated_replay_release_local_metrics.npz",
    "results/nest_ddi/later_graphsage_release_local_metrics.json",
    "results/nest_ddi/later_graphsage_release_local_metrics.npz",
    "results/nest_ddi/modern_2026_baselines_release_local_metrics.json",
    "results/nest_ddi/modern_2026_baselines_release_local_scores.npz",
    "results/nest_ddi/drift_temporal_reproduction_metrics.json",
    "results/nest_ddi/drift_chemberta_embeddings.npz",
    "results/nest_ddi/drugdagt_temporal_reproduction_metrics.json",
    "results/nest_ddi/smrddi_temporal_reproduction_metrics.json",
    "results/nest_ddi/later_temporal_release_local_drug_cluster_bootstrap.json",
    "results/nest_ddi/later_temporal_release_local_operational_gain.json",
    "results/nest_ddi/later_temporal_release_local_role_strata.json",
    "results/nest_ddi/later_temporal_release_local_calibration.json",
    "results/nest_ddi/drugbank_reference_independence_audit.json",
    "results/nest_ddi/openfda_label_concordance_2025q3.json",
]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    entries = []
    for relative in FILES:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": checksum(path)})
    manifest = {
        "release_scope": "code, environment specification, reproducibility contract and derived metric summaries",
        "excluded": "raw FAERS/AEMS files and derived row-level panels; obtain raw releases from FDA under applicable terms",
        "versioning_requirement": "assign a repository commit and archival DOI before external submission",
        "files": entries,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
