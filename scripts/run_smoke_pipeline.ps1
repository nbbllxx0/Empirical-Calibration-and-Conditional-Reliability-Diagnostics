param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

& $Python -m pip install -e .
& $Python -m bearing_dt.data prepare --dataset synthetic --out data\processed\synthetic_tiny --bearings 5 --steps 10 --window-size 96 --seed 9
& $Python -m bearing_dt.train --config configs\experiments\smoke_digital_twin.yaml
& $Python -m pytest
