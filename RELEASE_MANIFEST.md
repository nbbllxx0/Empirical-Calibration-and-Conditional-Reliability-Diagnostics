# Repository Contents

This repository is the public code release for the calibration-aware bearing RUL
subset-study evaluation.

## Included

- `bearing_dt/`: implementation for data registration, PHME preprocessing,
  train/evaluate loops, model definitions, uncertainty metrics, matrix runners,
  claim summaries, paired statistics, conditional coverage diagnostics, and
  archive-level exclusion-bias audits.
- `configs/experiments/`: paper-facing experiment configurations for the
  10-bearing main evaluation, separate-calibration sensitivity, stronger
  sequence/tabular comparators, direct-head ablation, ensemble fairness, seed
  sensitivity, diagnostics, preliminary context-control study, and smoke runs.
- `configs/suites/`: convenience suites for smoke, 10-bearing main,
  10-bearing diagnostics, and context-control runs.
- `tests/`: unit and integration tests that exercise feature construction,
  split construction, model paths, uncertainty, custom MATLAB parsing, and a
  tiny end-to-end training/report smoke test.
- `scripts/`: convenience PowerShell scripts for smoke and paper-pipeline
  execution, including an optional preliminary context-control diagnostic, plus
  a Python environment preflight check.
- `README.md`, `docs/DATA.md`, `docs/COMMANDS.md`: comprehensive
  reproduction instructions for the full implemented experiment suite.
- `LICENSE`: MIT license for the source code.
- `requirements-repro.txt`, `requirements-dev.txt`, `pyproject.toml`: package
  and environment metadata.

## Excluded

- Raw PHME archives.
- Processed PHME windows/features.
- `runs/` directories, checkpoints, predictions, metrics, and run manifests.
- Generated `paper_artifacts/` tables, summaries, and figures.
- Manuscript source and PDFs.
- Local audit records and temporary outputs.
- Precomputed result files. Running the commands creates generated outputs in
  ignored directories; none are included in this repository.

## Expected Generated Directories

The commands in the README create the following directories in a clean clone:

```text
data/raw/phme_tvoc/
data/processed/phme_tvoc_10b/
runs/
paper_artifacts/
```

These paths are deliberately ignored by Git.
