# Data Notes

The paper uses the PHME time-varying operating-condition bearing dataset:

- Zenodo record: `10868257`
- DOI: `10.5281/zenodo.10868257`
- Title: `Run-to-failure data set of ball bearings subjected to time-varying load and speed conditions`

The complete Zenodo record is about 152 GB. The paper benchmark uses a bounded
10-bearing subset:

```text
B01, B02, B03, B04, B05, B08, B10, B11, B12, B17
```

Download only these archives for the paper benchmark:

```text
B01.zip
B02.zip
B03.zip
B04.zip
B05.zip
B08.zip
B10.zip
B11.zip
B12.zip
B17.zip
```

The code can query the Zenodo API and write `file_manifest.csv` and
`source_manifest.json`:

```powershell
python -m bearing_dt.data fetch --dataset phme_tvoc --out data\raw\phme_tvoc --manifest-only
```

To download and extract the 10-bearing subset, use the full command in
`docs/COMMANDS.md` or `scripts/run_full_paper_pipeline.ps1`. The command has a
`--max-gb` guard because the selected archives are still large.

No raw or processed data are included in this repository.

The preliminary context-control diagnostic used a smaller five-bearing
processed set:

```text
B01, B03, B05, B11, B12
```

That auxiliary study is not the primary 10-bearing endpoint. If reproducing it,
prepare a separate processed root such as `data/processed/phme_tvoc`.
