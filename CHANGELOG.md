# Changelog

## 2026-09 — revised study

- Added `bearing_dt/qrei/`, a corrected physical-time pipeline. Both accelerometer channels are read by name over the
  full 1.6 s record, converted to g with the 10 g/V factor in every test log and resampled from the true 128 or
  64 kHz rate to 64 kHz; amplitude is retained.
- Endpoints follow the archive's stopping rules: eight tests with a documented vibration or temperature stop are
  targets; B01 and B05 are diagnostic only.
- Forecasts are in hours with fitting-bearing scales only; evaluation uses eight leave-one-bearing-out folds with
  separate validation and calibration bearings and a whole-bearing bootstrap.
- Published the fixed protocols, aggregate results, verification records and figure scripts in `QREI submission/`.

### Correction to the earlier release

The earlier `bearing_dt/data/` pipeline (arXiv:2607.08273) kept the first 512 samples of each record, standardized
each window to unit variance and labeled spectral features with a nominal 25.6 kHz rate. The archive's true rates are
128 kHz (B01–B08) and 64 kHz (B10–B17). The earlier code is kept unchanged for traceability; the revised study does
not reuse any of its numerical results.
