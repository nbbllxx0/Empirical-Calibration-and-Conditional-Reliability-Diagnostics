# Changelog

## 2026-09-30 — operating context from the set values; all models refitted

- The archive's measured operating channels are faulty in five tests: B04 measured speed is 0.498-0.500 times the set
  speed in every acquisition, while the strongest low-frequency vibration line follows the set speed; B01 measured
  speed stays at 3,002-3,003 rpm in 29 acquisitions whose set speed is 3,083-3,552 rpm; B10 and B17 measured static
  load stays near 4,580 N (2,287 acquisitions) and 4,300 N (609), above every set value of these tests; B04 and B05
  measured dynamic-load peak is about half the set amplitude. `QREI submission/evidence/operating_channel_check.csv`
  compares every test.
- `bearing_dt/qrei/prepare.py` now takes the operating context and the shaft speed of the defect-order bands from the
  set columns and keeps the measured columns as `measured_*` (not model inputs). Two B10 acquisitions taken at
  standstill keep their set values.
- Every model of both comparisons, and both extra seeds, was refitted; results, evidence, derived features and the
  spectrum record selection were regenerated. The measured-channel results remain in the history of this repository
  (commit 2ca9394 and earlier). With the set values, the latent-state network has a mean normalized error of 0.312
  (2 of 8 bearings at or below 0.20); the endpoint-aware comparison has no criterion with a first-rank fraction of 0.70,
  and the vibration-only comparison has stable winners under different criteria.
- The earlier `bearing_dt/data` pipeline (arXiv:2607.08273) also read the measured static load and speed
  (`meanAbs_statLoad`, `meanAbs_speed`; see `bearing_dt/data/loaders.py`).

## 2026-09-30 — caption and figure corrections

- Table S5 caption: the temperature statistic is the higher of the two bearing temperatures (T1, T2), not their mean.
- First-rank table caption defines the coverage gap as the absolute difference between coverage and the nominal 0.90.
- Two unused number macros removed from `build_revision.py` (and from `evidence/claims_ledger.csv`).
- Data figures: envelope-spectrum legend moved clear of the defect-order lines, integer time ticks, markers with equal
  values offset in the maintenance panel, and the schematic axis relabelled "Elapsed time". No number changes.

## 2026-09 — revised study

- Added `bearing_dt/qrei/`, a corrected physical-time pipeline. Both accelerometer channels are read by name over the
  full 1.6 s record, converted to g with the 10 g/V factor in every test log and resampled from the true 128 or
  64 kHz rate to 64 kHz; amplitude is retained.
- Endpoints follow the archive's stopping rules: eight tests with a documented vibration or temperature stop are
  targets; B01 and B05 are diagnostic only.
- Forecasts are in hours with fitting-bearing scales only; evaluation uses eight leave-one-bearing-out folds with
  separate validation and calibration bearings and a whole-bearing bootstrap.
- Published the fixed protocols, aggregate results, verification records and the data-figure script in
  `QREI submission/`.

### Correction to the earlier release

The earlier `bearing_dt/data/` pipeline (arXiv:2607.08273) kept the first 512 samples of each record, standardized
each window to unit variance and labeled spectral features with a nominal 25.6 kHz rate. The archive's true rates are
128 kHz (B01–B08) and 64 kHz (B10–B17). The earlier code is kept unchanged for traceability; the revised study does
not reuse any of its numerical results.
