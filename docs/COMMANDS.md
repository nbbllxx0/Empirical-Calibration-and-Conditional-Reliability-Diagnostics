# Reproduction Commands

The commands below assume PowerShell on Windows from the repository root.
Set `$PY` to the interpreter you will use for reproduction before running any
command. Full neural reproduction requires a CUDA-enabled interpreter; do not
use an unverified base Anaconda interpreter for the full evidence suite.

```powershell
$PY = "python"
```

## 1. Install

For the exact CUDA runtime used in the paper environment, install PyTorch 2.5.1
with CUDA 12.1 using the official PyTorch installation instructions, then:

```powershell
& $PY -m pip install -e .
& $PY -m pip install -r requirements-repro.txt
& $PY -m pip install -r requirements-dev.txt
& $PY scripts\check_environment.py --require-cuda
```

For CPU-only smoke tests:

```powershell
& $PY -m pip install -e .
& $PY -m pip install -r requirements-dev.txt
```

## 2. Smoke Test Without PHME Data

```powershell
& $PY -m bearing_dt.data prepare --dataset synthetic --out data\processed\synthetic_tiny --bearings 5 --steps 10 --window-size 96 --seed 9
& $PY -m bearing_dt.train --config configs\experiments\smoke_digital_twin.yaml
& $PY -m pytest
```

## 3. Download the 10-Bearing PHME Subset

```powershell
& $PY -m bearing_dt.data fetch --dataset phme_tvoc --out data\raw\phme_tvoc --extract --max-gb 30 --file B01.zip --file B02.zip --file B03.zip --file B04.zip --file B05.zip --file B08.zip --file B10.zip --file B11.zip --file B12.zip --file B17.zip
```

If the archives are already downloaded, place them in `data\raw\phme_tvoc\` and
rerun the same command with `--extract` to verify checksums and extract.

## 4. Prepare Processed Windows

```powershell
& $PY -m bearing_dt.data prepare --dataset phme_tvoc --raw data\raw\phme_tvoc --out data\processed\phme_tvoc_10b --window-size 512 --sample-rate 25600 --regime-bins 3
```

This creates:

```text
data/processed/phme_tvoc_10b/features.csv
data/processed/phme_tvoc_10b/signals.npy
data/processed/phme_tvoc_10b/targets.npy
data/processed/phme_tvoc_10b/manifest.json
```

## 5. Matched Leave-Operating-Regime-Out Sensitivity

```powershell
& $PY -m bearing_dt.matrix run-regime-matrix --processed-dir data\processed\phme_tvoc_10b --base-config configs\experiments\phme_10b_latent_load_speed.yaml --base-config configs\experiments\phme_10b_tcn_load_speed.yaml --base-config configs\experiments\phme_10b_attention_physics_load_speed.yaml --base-config configs\experiments\phme_10b_gb_load_speed.yaml --base-config configs\experiments\phme_10b_tabular_load_speed.yaml --out paper_artifacts\matrix\phme_10b_matched --runs-dir runs --prefix phme_10b_matched --no-resume
```

Summarize matched-sensitivity claims:

```powershell
& $PY -m bearing_dt.claims audit --runs runs --out paper_artifacts\claim_audit_10b_matched --prefix phme_10b_matched
```

## 6. Strict Primary Separate Validation/Calibration Endpoint

```powershell
& $PY -m bearing_dt.matrix run-regime-matrix --processed-dir data\processed\phme_tvoc_10b --base-config configs\experiments\phme_10b_latent_load_speed.yaml --base-config configs\experiments\phme_10b_tcn_load_speed.yaml --base-config configs\experiments\phme_10b_attention_physics_load_speed.yaml --base-config configs\experiments\phme_10b_sklearn_rf_load_speed.yaml --base-config configs\experiments\phme_10b_sklearn_gb_load_speed.yaml --base-config configs\experiments\phme_10b_gb_load_speed.yaml --base-config configs\experiments\phme_10b_tabular_load_speed.yaml --out paper_artifacts\matrix\phme_10b_separate_calibration --runs-dir runs --prefix phme_10b_separate_cal --separate-calibration
& $PY -m bearing_dt.claims audit --runs runs --out paper_artifacts\claim_audit_10b_separate_calibration --prefix phme_10b_separate_cal
& $PY -m bearing_dt.stats compare --runs runs --out paper_artifacts\statistics_10b_separate_calibration_with_rf --candidate phme_10b_separate_cal_phme_10b_latent_load_speed --baseline phme_10b_separate_cal_phme_10b_tcn_load_speed --baseline phme_10b_separate_cal_phme_10b_attention_physics_load_speed --baseline phme_10b_separate_cal_phme_10b_sklearn_rf_load_speed --baseline phme_10b_separate_cal_phme_10b_sklearn_gb_load_speed --baseline phme_10b_separate_cal_phme_10b_gb_load_speed --baseline phme_10b_separate_cal_phme_10b_tabular_load_speed
& $PY -m bearing_dt.diagnostics conditional-coverage --runs runs --processed-dir data\processed\phme_tvoc_10b --out paper_artifacts\conditional_coverage_10b_separate_calibration --prefix phme_10b_separate_cal
& $PY -m bearing_dt.diagnostics phme-subset-audit --raw-dir data\raw\phme_tvoc --processed-dir data\processed\phme_tvoc_10b --out paper_artifacts\exclusion_bias_10b
```

## 7. Leave-Bearing-Out Generalization Check

```powershell
& $PY -m bearing_dt.matrix run-bearing-matrix --processed-dir data\processed\phme_tvoc_10b --base-config configs\experiments\phme_10b_latent_load_speed.yaml --base-config configs\experiments\phme_10b_tcn_load_speed.yaml --base-config configs\experiments\phme_10b_attention_physics_load_speed.yaml --base-config configs\experiments\phme_10b_gb_load_speed.yaml --base-config configs\experiments\phme_10b_tabular_load_speed.yaml --out paper_artifacts\matrix\phme_10b_bearing --runs-dir runs --prefix phme_10b_bearing --no-resume
& $PY -m bearing_dt.claims audit --runs runs --out paper_artifacts\claim_audit_10b_bearing --prefix phme_10b_bearing
```

## 8. Ensemble Fairness, Stronger Tabular Comparator, and Seed Sensitivity

```powershell
& $PY -m bearing_dt.matrix run-regime-matrix --processed-dir data\processed\phme_tvoc_10b --base-config configs\experiments\phme_10b_latent_single_load_speed.yaml --base-config configs\experiments\phme_10b_tcn_ensemble_load_speed.yaml --base-config configs\experiments\phme_10b_attention_physics_ensemble_load_speed.yaml --base-config configs\experiments\phme_10b_sklearn_rf_load_speed.yaml --base-config configs\experiments\phme_10b_feature_mlp_load_speed.yaml --base-config configs\experiments\phme_10b_tcn_raw_only_load_speed.yaml --out paper_artifacts\matrix\phme_10b_fairness --runs-dir runs --prefix phme_10b_fairness
& $PY -m bearing_dt.claims audit --runs runs --out paper_artifacts\claim_audit_10b_fairness --prefix phme_10b_fairness
& $PY -m bearing_dt.stats compare --runs runs --out paper_artifacts\statistics_10b_with_rf --candidate phme_10b_matched_phme_10b_latent_load_speed --baseline phme_10b_matched_phme_10b_tcn_load_speed --baseline phme_10b_matched_phme_10b_attention_physics_load_speed --baseline phme_10b_matched_phme_10b_gb_load_speed --baseline phme_10b_matched_phme_10b_tabular_load_speed --baseline phme_10b_fairness_phme_10b_sklearn_rf_load_speed
& $PY -m bearing_dt.matrix run-regime-matrix --processed-dir data\processed\phme_tvoc_10b --base-config configs\experiments\phme_10b_latent_load_speed.yaml --base-config configs\experiments\phme_10b_tcn_load_speed.yaml --base-config configs\experiments\phme_10b_attention_physics_load_speed.yaml --out paper_artifacts\matrix\phme_10b_seed_sensitivity --runs-dir runs --prefix phme_10b_seed_sens --regime Lhigh_Shigh --regime Llow_Shigh --regime Lmid_Smid --model-seed 101 --model-seed 202 --model-seed 303
```

## 9. Diagnostics: Ablations, Prefix Observation, and Stress

```powershell
& $PY -m bearing_dt.matrix run-regime-matrix --processed-dir data\processed\phme_tvoc_10b --base-config configs\experiments\phme_10b_diag_no_context.yaml --base-config configs\experiments\phme_10b_diag_no_features.yaml --base-config configs\experiments\phme_10b_diag_no_raw.yaml --base-config configs\experiments\phme_10b_diag_stress_prefix.yaml --out paper_artifacts\matrix\phme_10b_diagnostics --runs-dir runs --prefix phme_10b_diag_cuda --no-resume
& $PY -m bearing_dt.claims audit --runs runs --out paper_artifacts\claim_audit_10b_diagnostics --prefix phme_10b_diag_cuda
& $PY -m bearing_dt.evidence summarize-model --runs runs --out paper_artifacts\evidence_10b_diagnostics --model-token phme_10b_diag_cuda_phme_10b_diag_stress_prefix
```

## 10. Preliminary Context-Control Diagnostic

The paper uses this as a design-control analysis, not as the primary 10-bearing
evidence endpoint. It uses the five-bearing development subset, so prepare a
separate processed root:

```powershell
& $PY -m bearing_dt.data fetch --dataset phme_tvoc --out data\raw\phme_tvoc --extract --max-gb 10 --file B01.zip --file B03.zip --file B05.zip --file B11.zip --file B12.zip
& $PY -m bearing_dt.data prepare --dataset phme_tvoc --raw data\raw\phme_tvoc --out data\processed\phme_tvoc --window-size 512 --sample-rate 25600 --regime-bins 3
```

Then run:

```powershell
& $PY -m bearing_dt.matrix run-regime-matrix --processed-dir data\processed\phme_tvoc --base-config configs\experiments\phme_context_latent_none.yaml --base-config configs\experiments\phme_context_latent_load_speed.yaml --base-config configs\experiments\phme_context_latent_load_speed_regime.yaml --base-config configs\experiments\phme_context_tcn_none.yaml --base-config configs\experiments\phme_context_tcn_load_speed.yaml --base-config configs\experiments\phme_context_tcn_load_speed_regime.yaml --base-config configs\experiments\phme_context_gb_none.yaml --base-config configs\experiments\phme_context_gb_load_speed.yaml --base-config configs\experiments\phme_context_gb_load_speed_regime.yaml --out paper_artifacts\matrix\phme_context_sensitivity --runs-dir runs --prefix phme_context5 --no-resume
& $PY -m bearing_dt.claims audit --runs runs --out paper_artifacts\claim_audit_context_sensitivity --prefix phme_context5
```

## 11. Regenerate Paper-Facing Summaries

This code-only release does not ship the manuscript builder. The numerical
tables used in the paper are generated from the CSV/JSON outputs created by the
commands above.

