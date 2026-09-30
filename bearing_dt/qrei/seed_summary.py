"""Summarize all specified latent-state seeds without selecting a new primary."""
import json
from pathlib import Path

import pandas as pd

from .summarize import metrics


def main():
    root = Path("QREI submission")
    cfg = json.loads((root/"protocol_endpoint_v3.json").read_text())
    primary = pd.read_csv(root/"results/endpoint_v3/per_bearing_metrics.csv")
    rows = primary[primary.model == "representation"].copy()
    rows["seed"] = cfg["seeds"][0]
    parts = [rows]
    for seed in cfg["seeds"][1:]:
        folder = root/"results"/f"endpoint_seed_{seed}"
        raw = pd.read_csv(folder/"joined_predictions.csv")
        assert set(raw.model) == {"representation"}
        assert set(raw.bearing_id) == set(cfg["event_bearings"])
        assert len(raw) == int(primary[primary.model == "representation"].records.sum())
        assert not raw.duplicated(["bearing_id", "record_id"]).any()
        rows = []
        for bearing, g in raw.groupby("bearing_id"):
            split = json.loads((folder/bearing/"split.json").read_text())
            payload = json.loads((folder/bearing/"payload.json").read_text())
            assert split["test"] == bearing
            assert abs(g.training_scale_hours.iloc[0]-payload["target_scale_hours"]) < 1e-8
            rows.append({"seed": seed, "model": "representation", "bearing_id": bearing,
                         **metrics(g.sort_values("elapsed_hours"))})
        parts.append(pd.DataFrame(rows))
    per = pd.concat(parts, ignore_index=True)
    oldclock = float(pd.read_csv(root/"results/endpoint_v3/model_summary.csv").set_index("model").loc["time_only", "late_MAE_hours"])
    summary = []
    for seed, g in per.groupby("seed"):
        mean, late, slope = float(g.nMAE.mean()), float(g.late_MAE_hours.mean()), float(g.slope.median())
        count = int((g.nMAE <= .20).sum())
        summary.append({"seed": int(seed), "primary": seed == cfg["seeds"][0], "nMAE": mean,
                        "MAE_hours": float(g.MAE_hours.mean()), "late_MAE_hours": late,
                        "median_slope": slope, "coverage": float(g.coverage.mean()),
                        "bearings_nMAE_le_0.20": count, "G1": mean <= .15 and count >= 6,
                        "G2": late < oldclock, "G3": slope <= -.7})
    out = root/"results/endpoint_v3/comparison"
    per.to_csv(out/"latent_seed_per_bearing.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(out/"latent_seed_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
