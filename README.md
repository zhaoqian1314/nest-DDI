# NEST-DDI

**Auditable temporal modeling for prospective drug–drug interaction report monitoring**

NEST-DDI is a cutoff-safe evaluation benchmark for prospective pharmacovigilance DDI monitoring. Instead of splitting a curated interaction table at random, it fixes the candidate risk set and the available information *before* each future quarter, then asks which drug pairs are actually reported next. The protocol separates three distinct targets that are usually conflated:

| Task | Question | Positive rate (2024Q3 cutoff) |
|---|---|---|
| **Pair-report recurrence** | Is a previously observed pair reported again next quarter? | 17,804 / 93,194 |
| **First-pair report** | Does a pair with no previous report receive its first report? | 4,118 / 93,194 |
| **First pair–event discovery** | Does an unseen pair–MedDRA PT triplet occur? | sampled case–control |

The companion model, NEST-DDI, combines a static structural background with multi-scale report history through deterministic temporal operators.

> **Scope.** This is an evaluation protocol and a matched baseline suite, not a causal DDI predictor. FAERS/AEMS co-medication reports are time-stamped observations of a *reporting process* — they are not exposure-normalized incidence. See [Interpretation boundaries](#interpretation-boundaries).

---

## Headline results

Frozen release-local leaderboard. **AUPR is the primary metric**; precision@K saturates at 1.00 for most methods and is therefore uninformative here. `†` marks a prospective binary adaptation whose original checkpoint or multi-class label contract differs from this benchmark.

| Method | Rec-AUROC | Rec-AUPR | First-AUROC | First-AUPR |
|---|---:|---:|---:|---:|
| Recency–popularity | 0.9762 | 0.89444 | 0.8991 | 0.13253 |
| DRIFT-Temporal `†` | 0.87124 | 0.57142 | 0.72649 | 0.04116 |
| DrugDAGT-Temporal `†` | 0.56954 | 0.15807 | 0.53672 | 0.01751 |
| SMR-DDI-Temporal `†` | 0.86529 | 0.55594 | 0.66384 | 0.03205 |
| ExtraTrees | 0.9776 | 0.90192 | 0.9018 | 0.13776 |
| HGB | 0.97810 | 0.90437 | 0.90384 | 0.14570 |
| FT-Transformer | 0.97831 | 0.90448 | 0.90568 | 0.14387 |
| **NEST-DDI** | **0.97832** | **0.90520** | 0.90407 | **0.14870** |

The recurrence gain over HGB is **small but consistent**: +0.00090 AUPR (95% CI 0.00062–0.00120) and +0.00022 AUROC on the frozen 2025Q1→2025Q2 test, replicated on the later 2025Q2→2025Q3 holdout at +0.00110 AUPR (95% CI 0.00081–0.00139). Paired uncertainty uses 300 stratified bootstrap resamples.

A fixed equal-rank NEST–GraphSAGE fusion improves first-pair ranking to **AUPR 0.16407 / AUROC 0.90964** (+0.01468 AUPR, 95% CI [0.01103, 0.01850] over optimized NEST). These fusion results are exploratory and were not used to tune recurrence.

### Feature ablation (frozen recurrence panel)

| Variant | AUPR |
|---|---:|
| NEST-DDI (full) | 0.90526 |
| − all decayed history | 0.88961 |
| − static structural background | 0.90402 |
| − drug-level analogues | 0.90451 |

Temporal history carries most of the signal; the structural background contributes a small complementary increment.

### Additional evidence

- **Later holdout (2025Q2→2025Q3):** AUPR 0.91285 vs 0.91174 for HGB.
- **Calibration:** Brier 0.03301 / ECE 0.00346, versus 0.03334 / 0.00688 for HGB — calibration is reported separately from discrimination.
- **Seed sensitivity:** FT-Transformer first-pair AUPR 0.14387 / 0.14665 across seeds 42 and 7; GraphSAGE fusion 0.15039 / 0.15353. Recurrence AUPR moved by <0.001.
- **Sampling sensitivity:** on 1:10 case–control samples a fixed scorer gives AUROC 0.74320 / 0.74317, while sampled AUPR is 0.20848 / 0.20847. AUROC is the safer cross-sample comparison.
- **Negative control:** historical pair–event count reaches AUPR 0.85475 while Ω₀₂₅ reaches 0.68464 — counting predicts re-reporting, not unseen-triplet discovery.
- **Reporting-role strata (later holdout):** NEST − HGB AUPR difference is 0.00052 (suspect–suspect), 0.00049 (concomitant–concomitant), 0.00130 (suspect–concomitant). The gain is not invariant to reporting role, and some small strata favor HGB.
- **Operational capacity:** at the top 10,000 / 50,000 the later holdout adds +1 / +59 observed positives over HGB. A cutoff-dated openFDA label-text audit mentions 27 of NEST-DDI's top 300 pairs versus 31 for HGB — label absence is not a negative.

---

## Benchmark construction

Quarterly FAERS/AEMS ASCII releases, 2024Q3 through 2025Q3. Within each release only the latest case version is retained, so later releases cannot rewrite historical features. Drug names are normalized to the DDInter inventory, cases with more than 20 mapped drugs are excluded, and each case contributes unique unordered drug pairs and MedDRA preferred terms.

| Cutoff | Cases | Pairs | Recurrence+ | First+ |
|---|---:|---:|---:|---:|
| 2024Q3 | 162,538 | 93,194 | 17,804 | 4,118 |
| 2024Q4 | 173,234 | 97,943 | 18,226 | 4,506 |
| 2025Q1 | 169,726 | 94,655 | 19,041 | 4,772 |
| 2025Q2 | 162,600 | 92,631 | 19,388 | — |

The candidate universe is fixed from the DDInter inventory; labels are opened only in the following quarter. The ratio of future positives is therefore a property of the reporting process, not a chosen class balance. The release-local pair-day file contains 3,005,177 rows and 177,902 observed pairs.

**Protocol.** Rolling cutoffs 2024Q3, 2024Q4 and 2025Q1 define next-quarter folds; 2025Q1→2025Q2 is the **frozen test**; 2025Q2→2025Q3 is a later holdout evaluated with all choices frozen. Matched tabular and historical-graph comparisons share a common **41-feature contract**. Graph edges use historical co-medication only.

---

## Repository layout

```
nest-DDI/
├── environment_temporal.yml     # conda environment specification
├── docs/                        # reconstruction contract, audits, pre-review history, claim ledger
├── scripts/                     # data builders, model runners, baseline suites, audits
├── src/ddi/                     # DDInter dataset builder + PubChem SMILES fetcher
├── results/nest_ddi/            # machine-readable metrics (JSON), one file per run/audit
└── README.md
```

Key documents:

- [`docs/NEST_DDI_REPRODUCIBILITY.md`](docs/NEST_DDI_REPRODUCIBILITY.md) — the ordered reconstruction contract and required audit outputs.
- [`docs/NEST_DDI_RELEASE_MANIFEST.json`](docs/NEST_DDI_RELEASE_MANIFEST.json) — SHA-256 manifest of the released files.
- [`docs/NEST_DDI_MANUSCRIPT_ARGUMENT.md`](docs/NEST_DDI_MANUSCRIPT_ARGUMENT.md) — claim–evidence–boundary ledger and terminology definitions.
- [`docs/NEST_DDI_FINAL_SUBMISSION_AUDIT.md`](docs/NEST_DDI_FINAL_SUBMISSION_AUDIT.md) — acceptance-readiness audit, including the open external gates.
- [`docs/NEST_DDI_ESWA_PRE_REVIEW_V1.md`](docs/NEST_DDI_ESWA_PRE_REVIEW_V1.md) … [`V8`](docs/NEST_DDI_ESWA_PRE_REVIEW_V8.md) — reviewer simulation history.
- [`results/nest_ddi/all_baseline_leaderboard_audit.json`](results/nest_ddi/all_baseline_leaderboard_audit.json) — baseline leaderboard audit record.

---

## Setup

```bash
conda env create -f environment_temporal.yml
conda activate ddi-temporal-obspu
```

Core dependencies: Python 3.12.4, PyTorch 2.5.1, scikit-learn 1.9.0, pandas 2.2.2, NumPy 1.26.4, RDKit 2025.03.3.

> **Portability.** The scripts currently hardcode `ROOT = Path(r"D:\BI\DDI")` at the top of each file. Before running anywhere else, update that path (and the `DATA`/`RESULTS` derivations that follow it) to your checkout location.

---

## Reproduction

The full ordered sequence is in [`docs/NEST_DDI_REPRODUCIBILITY.md`](docs/NEST_DDI_REPRODUCIBILITY.md). The core path:

```bash
# 1. Build dated pair and pair-event records, then the prediction panels
python scripts/build_nest_pilot.py
python scripts/build_nest_prediction_panel.py
python scripts/build_nest_pv_sufficient_statistics.py

# 2. Pair-report baseline suites
python scripts/run_nest_baselines.py
python scripts/run_nest_extended_baselines.py
python scripts/run_nest_final_comparison.py

# 3. Deep and historical-graph baselines
python scripts/run_nest_deep_learning_baselines.py --epochs 8 --batch-size 4096 --seed 42
python scripts/run_nest_graph_deep_baselines.py --epochs 6 --batch-size 16384 --seed 42
python scripts/audit_graph_deep_bootstrap.py

# 4. First pair–event discovery and the later holdout
python scripts/run_nest_first_event_baselines.py --input-tag release_local --output-tag first_event_baseline_release_local
python scripts/run_nest_later_temporal_test.py --input-tag 5q_release --output-tag later_temporal_release_local

# 5. Release gate
python scripts/build_nest_release_manifest.py
python scripts/audit_submission_readiness.py
```

Random seeds are 42 unless a script argument or the output audit states otherwise. The reconstruction contract lists the required audit outputs and their expected locations.

---

## Data availability

**Raw FAERS/AEMS records are not redistributed in this repository.** They are obtained from the FDA under the original distribution terms:

- FDA Adverse Event Monitoring System (AEMS) / FAERS Quarterly Data Extract Files — <https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html>

Place each quarterly ASCII release under `data/processed/faers_<quarter>/ASCII/` before running the reconstruction. Note that the quarterly extracts are **not cumulative**.

Derived row-level panels and case-level outputs are also not redistributed; they are rebuilt from the raw quarters by the scripts above. Drug-name normalization follows the DDInter inventory; independent label-text checks use the [openFDA drug labeling API](https://open.fda.gov/apis/drug/label/). This repository ships derived metric summaries as JSON only.

---

## Interpretation boundaries

These are load-bearing, not boilerplate:

1. **FAERS is a reporting system, not an exposure registry.** It lacks denominators and is affected by indication, publicity, regulatory warnings and duplicate submissions. Labels represent *reporting*, not incidence.
2. **Unreported pairs are not confirmed negatives.** They mix true negatives, unobserved exposure and delayed ascertainment — a positive–unlabeled setting.
3. **The recurrence gain is small.** +0.00090 AUPR over HGB is statistically detectable but practically modest.
4. **Sampled AUPR is not prevalence.** Event-discovery negatives are case–control sampled at a documented ratio; sampled AUPR and precision@100 must retain that sampling contract.
5. **No causal or mechanistic claim.** NEST-DDI does not identify pathways, dose or exposure, and it is not independent clinical validation.
6. **Evidence counts, mechanism codes, severity evidence and raw-source flags are prohibited as model features** (leakage rule, enforced by the audits in `scripts/`).
7. Drug mapping, the 20-drug cap, five-quarter coverage, case–control sampling and receipt-date-vs-onset differences all limit transportability.

---

## Citation

```bibtex
@inproceedings{zhao2027nestddi,
  title     = {NEST-DDI: Auditable Temporal Modeling for Prospective
               Drug--Drug Interaction Report Monitoring},
  author    = {Zhao, Qian and Wang, Zhensong and Wang, Juan},
  booktitle = {IEEE International Conference on Acoustics, Speech and
               Signal Processing (ICASSP)},
  year      = {2027},
  note      = {Code: \url{https://github.com/zhaoqian1314/nest-DDI}}
}
```

The venue, pages and DOI will be filled in at camera-ready. Versioned release metadata (repository commit, archival DOI, SPDX license) is recorded according to [`docs/NEST_DDI_ARCHIVE_RELEASE_INSTRUCTIONS.md`](docs/NEST_DDI_ARCHIVE_RELEASE_INSTRUCTIONS.md).

---

## License

Not yet assigned. Code and documentation licensing is an open author/institutional decision; an SPDX identifier will be recorded here before the archival release. Until then, no redistribution rights are granted.

---

## Acknowledgements

Drug normalization uses the DDInter inventory; independent label-text checks use the openFDA drug labeling API; raw reporting data are obtained from the FDA FAERS/AEMS quarterly extracts. All third-party resources remain under their original terms.
