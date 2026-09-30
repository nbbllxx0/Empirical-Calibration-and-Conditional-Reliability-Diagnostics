# Revised study: bearing RUL forecasts under temperature and vibration stopping criteria

This folder holds the fixed protocols, aggregate results, verification records and data-figure script of the revised study

> S. Yang, J. Wang and Y. Wang, *Evaluating Bearing Remaining Useful Life Forecasts for Maintenance under Temperature and Vibration Stopping Criteria* (manuscript in preparation).

The analysis code is the `bearing_dt.qrei` package in this repository. The folder name `QREI submission/` is kept because the code reads and writes these relative paths; run every command from the repository root.

## What the study does

- Uses the ten single-archive tests of the public PHME run-to-failure bearing archive (B01, B02, B03, B04, B05, B08, B10, B11, B12, B17; Zenodo DOI [10.5281/zenodo.10868257](https://doi.org/10.5281/zenodo.10868257)). The seven multipart tests are not processed.
- Treats the eight tests with a documented vibration or temperature stop as supervised targets. B01 (no preset criterion) and B05 (deliberately interrupted) are used for diagnosis only.
- Processes both accelerometer channels over the full 1.6 s record in physical units (10 g/V), with the true 128 or 64 kHz sampling rate resampled to 64 kHz; features include band powers, a 6–10 kHz Hilbert envelope and power at bearing defect orders, plus causal temperature and threshold histories.
- Uses the set values of static load, dynamic-load amplitude and shaft speed as the operating context and for the defect-order frequencies. The archive's measured load and speed channels are faulty in five tests (B04 speed is half the set speed, B01 speed stops near 3,000 rpm, B10 and B17 static load stays far above its set range, B04 and B05 dynamic load is about half its set amplitude); `evidence/operating_channel_check.csv` compares them with the set values and the vibration spectrum.
- Forecasts residual time in hours without using the test bearing's life, in eight leave-one-bearing-out folds with separate validation and calibration bearings.
- Compares eight learned predictors with six reference controls, resamples whole bearings (10,000 draws) and evaluates a first-trigger maintenance rule.

## Contents

| Path | Contents |
|---|---|
| `protocol.json`, `protocol_endpoint_v3.json` | Analysis settings and evaluation targets, fixed before the corresponding models were fitted. |
| `protocol_sensitivity_addendum.json` | Secondary analyses (extra seeds for every stochastic predictor, two alternative fold-role allocations, ranking and maintenance diagnostics), recorded before they were run; primary results unchanged. |
| `results/primary/` | Vibration-only comparison: per-bearing and summary metrics, bootstrap intervals, rank fractions, maintenance results; `joined_predictions.csv.gz` holds every forecast. |
| `results/endpoint_v3/` | Endpoint-aware comparison (main results of the paper), same files; `comparison/` holds paired changes, seed results, conditional results, threshold crossings and the spectrum record selection. |
| `results/endpoint_v3/sensitivity/` | Results of the secondary analyses: first-rank fractions for every comparison (`rank_scenarios*.csv`), seed and role-allocation summaries, horizon support, paired differences, population controls, life-phase split, calendar gaps. |
| `results/sensitivity/`, `results/endpoint_seed_*/` | Every forecast of the seed and role-allocation refits and of the two extra latent-state seeds (`joined_predictions.csv.gz`). |
| `evidence/` | Endpoint ledger, sensor calibration factors, operating-channel check, nominal-range flags, number ledger and independent verification records. |
| `manuscript/figure_sources/` | `data_figures.py`, which draws the data figures of the paper (Figures 4–7 and S1–S4) from the result files. |
| `reproduce.ps1` | Stage commands: `prepare`, `fit`, `summarize`, `sensitivity`, `verify`, `paper`. |
| `../data/processed/phme_tvoc_10b_endpoint_v3/` | `features.csv.gz` (endpoint-aware features for all 14,297 acquisitions) and `manifest.json`. |

Not included: the raw archive (download it from Zenodo into `data/raw/phme_tvoc/`), waveform arrays, per-fold forecast folders and trained model weights. `pytest` runs without the raw data.

## Environment

The results were produced with Python 3.10.18, NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3 and scikit-learn 1.7.2 for preparation, tabular models and summaries, and Python 3.12 with PyTorch 2.11.0 (CUDA 12.8) for the neural models. Install the analysis extras with

```bash
pip install -e ".[qrei,test]"
```

## Reproducing

```powershell
# Analysis interpreter (pandas) and neural interpreter (PyTorch) may differ.
.\"QREI submission"\reproduce.ps1 -AnalysisPython python -NeuralPython python -Stage prepare
.\"QREI submission"\reproduce.ps1 -AnalysisPython python -NeuralPython python -Stage fit
.\"QREI submission"\reproduce.ps1 -AnalysisPython python -NeuralPython python -Stage summarize
.\"QREI submission"\reproduce.ps1 -AnalysisPython python -NeuralPython python -Stage sensitivity
.\"QREI submission"\reproduce.ps1 -AnalysisPython python -NeuralPython python -Stage verify
```

`prepare` and `fit` need the raw archive and write new processed data and per-fold results. To inspect or re-plot the published results without refitting, decompress the `*.csv.gz` files next to themselves (for example `python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('F.csv.gz'), open('F.csv','wb'))"`). `data_figures.py` reads the `.gz` files directly; the sensor-history and envelope-spectrum panels also need the raw archive.

## Correction to the earlier release

The earlier pipeline in `bearing_dt/data/` (used for arXiv:2607.08273) kept only the first 512 samples of each 1.6 s record, standardized each window to unit variance and labeled spectral features with a nominal 25.6 kHz rate. The archive's true rates are 128 kHz (B01–B08) and 64 kHz (B10–B17). The revised study replaces that pipeline with `bearing_dt/qrei/`; none of the earlier numerical results is reused.

## Secondary analyses (30 September 2026)

`bearing_dt.qrei.sensitivity_runs` refits every stochastic learned predictor with the two further listed seeds and every method under two alternative fold-role allocations (`--roles swap` and `--roles reverse` in `bearing_dt.qrei.evaluate`; the primary allocation is unchanged). `bearing_dt.qrei.sensitivity` computes the diagnostics; it reads the per-fold forecasts, or the published `joined_predictions.csv.gz` files when those are absent. The main finding is that this eight-bearing study does not establish a robust preferred model across the tested choices: the first-rank classification changes with the set of competing models, the training seed, the role allocation and single bearings, and the one nonzero prognostic horizon lasts 12 s. The late-life advantage of the latent-state network over the median-life clock holds under every seed and allocation, and no setting meets the accuracy target.
