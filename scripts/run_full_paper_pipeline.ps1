param(
    [string]$Python = "python",
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"

$RawDir = "data\raw\phme_tvoc"
$ProcessedDir = "data\processed\phme_tvoc_10b"
$MainOut = "paper_artifacts\matrix\phme_10b_matched"
$SeparateCalOut = "paper_artifacts\matrix\phme_10b_separate_calibration"
$BearingOut = "paper_artifacts\matrix\phme_10b_bearing"
$FairnessOut = "paper_artifacts\matrix\phme_10b_fairness"
$SeedOut = "paper_artifacts\matrix\phme_10b_seed_sensitivity"
$DiagOut = "paper_artifacts\matrix\phme_10b_diagnostics"

if (-not $SkipDownload) {
    & $Python -m bearing_dt.data fetch --dataset phme_tvoc --out $RawDir --extract --max-gb 30 `
        --file B01.zip --file B02.zip --file B03.zip --file B04.zip --file B05.zip `
        --file B08.zip --file B10.zip --file B11.zip --file B12.zip --file B17.zip
}

& $Python -m bearing_dt.data prepare --dataset phme_tvoc --raw $RawDir --out $ProcessedDir --window-size 512 --sample-rate 25600 --regime-bins 3

& $Python -m bearing_dt.matrix run-regime-matrix --processed-dir $ProcessedDir `
    --base-config configs\experiments\phme_10b_latent_load_speed.yaml `
    --base-config configs\experiments\phme_10b_tcn_load_speed.yaml `
    --base-config configs\experiments\phme_10b_attention_physics_load_speed.yaml `
    --base-config configs\experiments\phme_10b_gb_load_speed.yaml `
    --base-config configs\experiments\phme_10b_tabular_load_speed.yaml `
    --out $MainOut --runs-dir runs --prefix phme_10b_matched --no-resume

& $Python -m bearing_dt.claims audit --runs runs --out paper_artifacts\claim_audit_10b_matched --prefix phme_10b_matched

& $Python -m bearing_dt.matrix run-regime-matrix --processed-dir $ProcessedDir `
    --base-config configs\experiments\phme_10b_latent_load_speed.yaml `
    --base-config configs\experiments\phme_10b_tcn_load_speed.yaml `
    --base-config configs\experiments\phme_10b_attention_physics_load_speed.yaml `
    --base-config configs\experiments\phme_10b_sklearn_rf_load_speed.yaml `
    --base-config configs\experiments\phme_10b_sklearn_gb_load_speed.yaml `
    --base-config configs\experiments\phme_10b_gb_load_speed.yaml `
    --base-config configs\experiments\phme_10b_tabular_load_speed.yaml `
    --out $SeparateCalOut --runs-dir runs --prefix phme_10b_separate_cal --separate-calibration

& $Python -m bearing_dt.claims audit --runs runs --out paper_artifacts\claim_audit_10b_separate_calibration --prefix phme_10b_separate_cal
& $Python -m bearing_dt.stats compare --runs runs --out paper_artifacts\statistics_10b_separate_calibration_with_rf `
    --candidate phme_10b_separate_cal_phme_10b_latent_load_speed `
    --baseline phme_10b_separate_cal_phme_10b_tcn_load_speed `
    --baseline phme_10b_separate_cal_phme_10b_attention_physics_load_speed `
    --baseline phme_10b_separate_cal_phme_10b_sklearn_rf_load_speed `
    --baseline phme_10b_separate_cal_phme_10b_sklearn_gb_load_speed `
    --baseline phme_10b_separate_cal_phme_10b_gb_load_speed `
    --baseline phme_10b_separate_cal_phme_10b_tabular_load_speed

& $Python -m bearing_dt.diagnostics conditional-coverage --runs runs --processed-dir $ProcessedDir --out paper_artifacts\conditional_coverage_10b_separate_calibration --prefix phme_10b_separate_cal
& $Python -m bearing_dt.diagnostics phme-subset-audit --raw-dir $RawDir --processed-dir $ProcessedDir --out paper_artifacts\exclusion_bias_10b

& $Python -m bearing_dt.matrix run-bearing-matrix --processed-dir $ProcessedDir `
    --base-config configs\experiments\phme_10b_latent_load_speed.yaml `
    --base-config configs\experiments\phme_10b_tcn_load_speed.yaml `
    --base-config configs\experiments\phme_10b_attention_physics_load_speed.yaml `
    --base-config configs\experiments\phme_10b_gb_load_speed.yaml `
    --base-config configs\experiments\phme_10b_tabular_load_speed.yaml `
    --out $BearingOut --runs-dir runs --prefix phme_10b_bearing --no-resume

& $Python -m bearing_dt.claims audit --runs runs --out paper_artifacts\claim_audit_10b_bearing --prefix phme_10b_bearing

& $Python -m bearing_dt.matrix run-regime-matrix --processed-dir $ProcessedDir `
    --base-config configs\experiments\phme_10b_latent_single_load_speed.yaml `
    --base-config configs\experiments\phme_10b_tcn_ensemble_load_speed.yaml `
    --base-config configs\experiments\phme_10b_attention_physics_ensemble_load_speed.yaml `
    --base-config configs\experiments\phme_10b_sklearn_rf_load_speed.yaml `
    --base-config configs\experiments\phme_10b_feature_mlp_load_speed.yaml `
    --base-config configs\experiments\phme_10b_tcn_raw_only_load_speed.yaml `
    --out $FairnessOut --runs-dir runs --prefix phme_10b_fairness

& $Python -m bearing_dt.claims audit --runs runs --out paper_artifacts\claim_audit_10b_fairness --prefix phme_10b_fairness
& $Python -m bearing_dt.stats compare --runs runs --out paper_artifacts\statistics_10b_with_rf `
    --candidate phme_10b_matched_phme_10b_latent_load_speed `
    --baseline phme_10b_matched_phme_10b_tcn_load_speed `
    --baseline phme_10b_matched_phme_10b_attention_physics_load_speed `
    --baseline phme_10b_matched_phme_10b_gb_load_speed `
    --baseline phme_10b_matched_phme_10b_tabular_load_speed `
    --baseline phme_10b_fairness_phme_10b_sklearn_rf_load_speed

& $Python -m bearing_dt.matrix run-regime-matrix --processed-dir $ProcessedDir `
    --base-config configs\experiments\phme_10b_latent_load_speed.yaml `
    --base-config configs\experiments\phme_10b_tcn_load_speed.yaml `
    --base-config configs\experiments\phme_10b_attention_physics_load_speed.yaml `
    --out $SeedOut --runs-dir runs --prefix phme_10b_seed_sens `
    --regime Lhigh_Shigh --regime Llow_Shigh --regime Lmid_Smid `
    --model-seed 101 --model-seed 202 --model-seed 303

& $Python -m bearing_dt.matrix run-regime-matrix --processed-dir $ProcessedDir `
    --base-config configs\experiments\phme_10b_diag_no_context.yaml `
    --base-config configs\experiments\phme_10b_diag_no_features.yaml `
    --base-config configs\experiments\phme_10b_diag_no_raw.yaml `
    --base-config configs\experiments\phme_10b_diag_stress_prefix.yaml `
    --out $DiagOut --runs-dir runs --prefix phme_10b_diag_cuda --no-resume

& $Python -m bearing_dt.claims audit --runs runs --out paper_artifacts\claim_audit_10b_diagnostics --prefix phme_10b_diag_cuda
& $Python -m bearing_dt.evidence summarize-model --runs runs --out paper_artifacts\evidence_10b_diagnostics --model-token phme_10b_diag_cuda_phme_10b_diag_stress_prefix
