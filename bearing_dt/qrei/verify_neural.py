"""Replay persisted neural weights with a different batch size, without pandas."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bearing_dt.models import TCNRegressor, AttnPINNRegressor
from .neural import PhysicalLatentRegressor


def verify(root, output, models=("representation", "tcn", "attention")):
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checks = []
    for split in sorted(root.rglob("split.json")):
        folder = split.parent
        data = np.load(folder/"payload.npz")
        info = json.loads((folder/"payload.json").read_text())
        x = torch.from_numpy(data["features"]).to(device)
        ctx = torch.from_numpy(data["context"]).to(device)
        wave_source = np.load(info["signals_file"], mmap_mode="r")
        for name in models:
            model_path = folder/name/"weights.pt"
            if not model_path.exists():
                raise ValueError("Missing persisted weights: "+str(model_path))
            constructor = {"representation": PhysicalLatentRegressor, "tcn": TCNRegressor,
                           "attention": AttnPINNRegressor}[name]
            model = constructor(x.shape[1], 96, ctx.shape[1]) if name == "attention" else constructor(2, x.shape[1], 96, ctx.shape[1])
            model = model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            for role in ("calibration", "test"):
                idx = data[role]
                parts = []
                with torch.no_grad():
                    for start in range(0, len(idx), 113):
                        ii = idx[start:start+113]
                        batch = {"features": x[ii], "context": ctx[ii]}
                        if name != "attention":
                            wave = torch.from_numpy(np.array(wave_source[data["signal_indices"][ii]], copy=True)).to(device)
                            batch["signal"] = wave/torch.tensor(data["wave_scale"], device=device).view(1, 2, 1)
                        parts.append(model(batch)["rul"].cpu().numpy())
                rebuilt = np.concatenate(parts)
                saved = np.load(folder/name/(role+"_scaled.npy"))
                error = float(np.max(np.abs(saved-rebuilt)))
                assert np.allclose(saved, rebuilt, rtol=5e-5, atol=5e-5), (folder, name, role, error)
                checks.append({"bearing_id": folder.name, "model": name, "role": role,
                    "records": len(idx), "max_scaled_difference": error,
                    "max_hour_difference": error*info["target_scale_hours"], "status": "PASS",
                    "weights_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()})
            del model
        print("Verified persisted neural weights for "+folder.name, flush=True)
    assert len(checks) == 8*len(models)*2
    result = {"folds": 8, "persisted_neural_models": 8*len(models), "role_checks": len(checks),
              "replay_batch_size": 113, "training_output_batch_size": 256,
              "device": str(device), "torch": torch.__version__,
              "max_hour_difference": max(c["max_hour_difference"] for c in checks),
              "checks": checks, "status": "PASS"}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "checks"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("QREI submission/results/lobo_neural"))
    parser.add_argument("--out", type=Path, default=Path("QREI submission/evidence/neural_verification_v2.json"))
    parser.add_argument("--models", nargs="+", choices=("representation", "tcn", "attention"), default=("representation", "tcn", "attention"))
    args = parser.parse_args()
    verify(args.root, args.out, args.models)
