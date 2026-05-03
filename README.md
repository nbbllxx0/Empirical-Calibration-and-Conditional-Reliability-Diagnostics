# Calibration-Aware Bearing RUL Reliability Benchmark

Code-only release for a reliability-oriented bearing remaining-useful-life
evaluation on a curated PHME subset under operating-regime shift.

This repository implements the experiment pipeline used for the 10-bearing PHME
subset study in the manuscript:

```text
Empirical Calibration and Conditional Reliability of Bearing RUL Prediction under Operating-Regime Shift
```

The release is intentionally code-only. It does not include raw PHME archives,
processed arrays, trained checkpoints, generated result tables, figures,
manuscript source, or PDFs.

## Authors and Contact

Authors: Shaoliang Yang, Jun Wang, and Yunsheng Wang.

Institution: Department of Mechanical Engineering, Santa Clara University, Santa Clara, CA 95053, USA.

Corresponding author: Jun Wang (`jwang22@scu.edu`).

## What Is Implemented

The package implements:

- PHME Zenodo manifest registration, selected-file download, checksum checking,
  and ZIP extraction.
- A custom MATLAB v5 reader for the PHME archive layout.
- PHME preprocessing into windowed vibration arrays, RUL targets, engineered
  features, bearing metadata, load/speed context, and derived load-speed
  regimes.
- Configurable context modes: no context, load/speed, and load/speed plus
  derived regime or condition codes for sensitivity checks.
- Split protocols:
  - leave-one-operating-regime-out;
  - leave-one-bearing-out;
  - optional separate calibration regimes for stricter four-way splits;
  - random bearing splits for smoke tests.
- Models and baselines:
  - predictive latent-representation neural model;
  - fused direct-head ablation using the same encoders;
  - TCN baseline;
  - LSTM and Transformer sequence baseline configs;
  - attention physics-regularized neural baseline;
  - 400-tree random-forest tabular comparator;
  - sklearn gradient-boosting tabular comparator;
  - dependency-free gradient-boosted decision-stump tabular baseline;
  - dependency-free random-subspace ridge tabular baseline.
- Conformal interval construction and interval metrics.
- Diagnostic evaluation:
  - no-context, no-feature, and no-raw-channel ablations;
  - prefix-observation subsets;
  - additive-noise stress;
  - retained-feature raw-stream ablation;
  - stricter raw-plus-engineered sensor-loss probes;
  - high-load/high-speed condition-response diagnostic.
- Matrix runners, claim summaries, split summaries, paired regime-level
  statistical comparisons, separate-calibration sensitivity, conditional
  coverage reports, archive-level exclusion-bias reports, and bounded
  seed-sensitivity diagnostics.

## Experiment Suite Coverage

The release implements the full experiment suite, but ships no generated
evidence bundle. Running the commands creates local outputs under ignored
directories such as `data/`, `runs/`, and `paper_artifacts/`.

| Suite block | Purpose | Main local outputs |
| --- | --- | --- |
| Smoke pipeline | CI-style installation and training check without PHME data | `data/processed/synthetic_tiny`, `runs/` |
| 10-bearing preprocessing | Build the evaluated PHME subset from selected public archives | `data/processed/phme_tvoc_10b` |
| Strict 10-bearing primary matrix | Four-way train/model-selection/calibration/test leave-operating-regime-out comparison | `paper_artifacts/matrix/phme_10b_separate_calibration` |
| Strict paired statistics | Paired statistics for neural/tabular baselines plus the 400-tree random forest | `paper_artifacts/statistics_10b_separate_calibration_with_rf` |
| Matched-validation sensitivity matrix | Matched leave-operating-regime-out comparison | `paper_artifacts/matrix/phme_10b_matched` |
| Matched random-forest-inclusive paired statistics | Secondary paired statistics for neural/tabular baselines plus the 400-tree random forest | `paper_artifacts/statistics_10b_with_rf` |
| Conditional coverage | Per-regime, per-bearing, and nominal-coverage diagnostics | `paper_artifacts/conditional_coverage_10b_separate_calibration` |
| Exclusion-bias audit | Archive-level included/excluded PHME bearing record | `paper_artifacts/exclusion_bias_10b` |
| Leave-bearing-out | Bearing-identity generalization check | `paper_artifacts/matrix/phme_10b_bearing` |
| Fairness and RF comparators | Ensemble-matched neural comparators, feature/raw variants, RF | `paper_artifacts/matrix/phme_10b_fairness` |
| Seed sensitivity | Three-seed, three-regime bounded diagnostic | `paper_artifacts/matrix/phme_10b_seed_sensitivity` |
| Diagnostics and stress | Ablations, prefix observation, noise/raw-channel-loss stress | `paper_artifacts/matrix/phme_10b_diagnostics` |
| Preliminary context control | Five-bearing design-traceability context diagnostic | `paper_artifacts/matrix/phme_context_sensitivity` |

## Repository Layout

```text
bearing_dt/                 Python package
configs/experiments/        Paper-facing experiment configs
configs/suites/             Convenience suite definitions
tests/                      Unit and integration smoke tests
scripts/                    Reproduction helper scripts
docs/DATA.md                Dataset scope and download notes
docs/COMMANDS.md            Full command sequence
requirements-repro.txt      Runtime versions used for the reported CUDA runs
requirements-dev.txt        Test dependency
RELEASE_MANIFEST.md         Included/excluded file boundary
```

Generated local directories are ignored by Git:

```text
data/raw/
data/processed/
runs/
paper_artifacts/
```

## Environment

The paper runs used:

```text
Python 3.10.18
PyTorch 2.5.1
CUDA 12.1
GPU: NVIDIA GeForce RTX 4090
numpy 2.2.6
PyYAML 6.0.2
scikit-learn 1.7.2
```

Set the interpreter used for all commands. Use a CUDA-enabled interpreter for
full neural reproduction:

```powershell
$PY = "python"
```

Install the package and verify the runtime:

```powershell
& $PY -m pip install -e .
& $PY -m pip install -r requirements-dev.txt
& $PY scripts\check_environment.py --require-cuda
```

For GPU reproduction, install a CUDA-enabled PyTorch 2.5.1 build compatible
with CUDA 12.1 before running the neural matrices. CPU-only execution is
acceptable for tests and small synthetic smoke runs, but full neural matrices
will be slow. Do not run the full evidence suite with an unverified base
Anaconda interpreter.

## Quick Smoke Run

This does not require PHME data:

```powershell
.\scripts\run_smoke_pipeline.ps1 -Python $PY
```

Equivalent commands:

```powershell
& $PY -m bearing_dt.data prepare --dataset synthetic --out data\processed\synthetic_tiny --bearings 5 --steps 10 --window-size 96 --seed 9
& $PY -m bearing_dt.train --config configs\experiments\smoke_digital_twin.yaml
& $PY -m pytest
```

## PHME Data Scope

The paper subset study uses these 10 PHME bearing archives:

```text
B01.zip, B02.zip, B03.zip, B04.zip, B05.zip,
B08.zip, B10.zip, B11.zip, B12.zip, B17.zip
```

The complete Zenodo record is much larger than the evaluated subset. This
release downloads only the selected archives when reproducing the paper
evaluation. See `docs/DATA.md` for details.

## Full Paper Pipeline

Run the full pipeline:

```powershell
.\scripts\run_full_paper_pipeline.ps1 -Python $PY
```

If the selected PHME ZIP files are already downloaded and extracted under
`data\raw\phme_tvoc`, skip downloading:

```powershell
.\scripts\run_full_paper_pipeline.ps1 -Python $PY -SkipDownload
```

The script runs:

1. selected PHME download and extraction;
2. PHME preprocessing into `data\processed\phme_tvoc_10b`;
3. matched leave-operating-regime-out sensitivity evaluation;
4. strict separate-validation/separate-calibration primary evaluation;
5. primary claim audit;
6. leave-bearing-out generalization check;
7. ensemble/random-forest fairness matrix;
8. random-forest-inclusive paired statistics;
9. bounded seed sensitivity;
10. 10-bearing diagnostics and stress summaries.

For manual execution and the separate preliminary five-bearing context-control
diagnostic, use the commands in `docs/COMMANDS.md`.

## Main Evidence Commands

The strict primary separate-validation/separate-calibration regime matrix is:

```powershell
& $PY -m bearing_dt.matrix run-regime-matrix --processed-dir data\processed\phme_tvoc_10b --base-config configs\experiments\phme_10b_latent_load_speed.yaml --base-config configs\experiments\phme_10b_tcn_load_speed.yaml --base-config configs\experiments\phme_10b_attention_physics_load_speed.yaml --base-config configs\experiments\phme_10b_sklearn_rf_load_speed.yaml --base-config configs\experiments\phme_10b_sklearn_gb_load_speed.yaml --base-config configs\experiments\phme_10b_gb_load_speed.yaml --base-config configs\experiments\phme_10b_tabular_load_speed.yaml --out paper_artifacts\matrix\phme_10b_separate_calibration --runs-dir runs --prefix phme_10b_separate_cal --separate-calibration
```

Primary claim summary:

```powershell
& $PY -m bearing_dt.claims audit --runs runs --out paper_artifacts\claim_audit_10b_separate_calibration --prefix phme_10b_separate_cal
```

Run the matched-validation sensitivity:

```powershell
& $PY -m bearing_dt.matrix run-regime-matrix --processed-dir data\processed\phme_tvoc_10b --base-config configs\experiments\phme_10b_latent_load_speed.yaml --base-config configs\experiments\phme_10b_tcn_load_speed.yaml --base-config configs\experiments\phme_10b_attention_physics_load_speed.yaml --base-config configs\experiments\phme_10b_gb_load_speed.yaml --base-config configs\experiments\phme_10b_tabular_load_speed.yaml --out paper_artifacts\matrix\phme_10b_matched --runs-dir runs --prefix phme_10b_matched --no-resume
& $PY -m bearing_dt.claims audit --runs runs --out paper_artifacts\claim_audit_10b_matched --prefix phme_10b_matched
```

Run the strict paired statistics and conditional diagnostics:

```powershell
& $PY -m bearing_dt.stats compare --runs runs --out paper_artifacts\statistics_10b_separate_calibration_with_rf --candidate phme_10b_separate_cal_phme_10b_latent_load_speed --baseline phme_10b_separate_cal_phme_10b_tcn_load_speed --baseline phme_10b_separate_cal_phme_10b_attention_physics_load_speed --baseline phme_10b_separate_cal_phme_10b_sklearn_rf_load_speed --baseline phme_10b_separate_cal_phme_10b_sklearn_gb_load_speed --baseline phme_10b_separate_cal_phme_10b_gb_load_speed --baseline phme_10b_separate_cal_phme_10b_tabular_load_speed
& $PY -m bearing_dt.diagnostics conditional-coverage --runs runs --processed-dir data\processed\phme_tvoc_10b --out paper_artifacts\conditional_coverage_10b_separate_calibration --prefix phme_10b_separate_cal
& $PY -m bearing_dt.diagnostics phme-subset-audit --raw-dir data\raw\phme_tvoc --processed-dir data\processed\phme_tvoc_10b --out paper_artifacts\exclusion_bias_10b
```

Run the baseline-fairness matrix with ensemble-matched neural comparators,
feature/raw ablations, and the 400-tree random forest, then generate the random-forest-inclusive
paired statistics:

```powershell
& $PY -m bearing_dt.matrix run-regime-matrix --processed-dir data\processed\phme_tvoc_10b --base-config configs\experiments\phme_10b_latent_single_load_speed.yaml --base-config configs\experiments\phme_10b_tcn_ensemble_load_speed.yaml --base-config configs\experiments\phme_10b_attention_physics_ensemble_load_speed.yaml --base-config configs\experiments\phme_10b_sklearn_rf_load_speed.yaml --base-config configs\experiments\phme_10b_feature_mlp_load_speed.yaml --base-config configs\experiments\phme_10b_tcn_raw_only_load_speed.yaml --out paper_artifacts\matrix\phme_10b_fairness --runs-dir runs --prefix phme_10b_fairness
& $PY -m bearing_dt.claims audit --runs runs --out paper_artifacts\claim_audit_10b_fairness --prefix phme_10b_fairness
& $PY -m bearing_dt.stats compare --runs runs --out paper_artifacts\statistics_10b_with_rf --candidate phme_10b_matched_phme_10b_latent_load_speed --baseline phme_10b_matched_phme_10b_tcn_load_speed --baseline phme_10b_matched_phme_10b_attention_physics_load_speed --baseline phme_10b_matched_phme_10b_gb_load_speed --baseline phme_10b_matched_phme_10b_tabular_load_speed --baseline phme_10b_fairness_phme_10b_sklearn_rf_load_speed
```

Run the bounded three-seed, three-regime diagnostic:

```powershell
& $PY -m bearing_dt.matrix run-regime-matrix --processed-dir data\processed\phme_tvoc_10b --base-config configs\experiments\phme_10b_latent_load_speed.yaml --base-config configs\experiments\phme_10b_tcn_load_speed.yaml --base-config configs\experiments\phme_10b_attention_physics_load_speed.yaml --out paper_artifacts\matrix\phme_10b_seed_sensitivity --runs-dir runs --prefix phme_10b_seed_sens --regime Lhigh_Shigh --regime Llow_Shigh --regime Lmid_Smid --model-seed 101 --model-seed 202 --model-seed 303
```

## Important Interpretation Boundary

This release supports reproducing the reported 10-bearing PHME subset-study
workflow. It does not establish:

- full 17-bearing PHME generalization;
- external validation on XJTU-SY, NASA IMS, FEMTO, or C-MAPSS;
- physics regularization as the proven accuracy mechanism;
- calibrated deployment under raw-channel loss or unseen bearing identities.

Derived operating-regime labels are used for split construction and reporting.
The canonical model input uses measured load and speed only, not the derived
held-out regime label.

## License

This staged payload is distributed under the MIT License. See `LICENSE`.
