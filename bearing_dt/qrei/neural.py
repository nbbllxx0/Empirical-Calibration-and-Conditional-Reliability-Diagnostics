"""GPU worker without pandas; numeric payloads bridge the two verified environments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from bearing_dt.models import BearingDigitalTwin, TCNRegressor, AttnPINNRegressor


class PhysicalLatentRegressor(BearingDigitalTwin):
    def forward(self,batch):
        out=super().forward(batch)
        # Unbounded positive residual time, converted with a training-only scale.
        out["rul"]=(1-out["damage"])/(out["damage_rate"]+1e-4)
        return out


def fit(payload: Path, model_name: str, seed: int, out: Path, max_epochs: int = 120):
    start=time.time()
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark=False
    rng=np.random.default_rng(seed)
    arrays=np.load(payload/"payload.npz")
    info=json.loads((payload/"payload.json").read_text())
    x=torch.from_numpy(arrays["features"]).to(device)
    context=torch.from_numpy(arrays["context"]).to(device)
    y=torch.from_numpy(arrays["target_scaled"]).to(device)
    if model_name=="attention":
        model=AttnPINNRegressor(x.shape[1],96,context.shape[1]).to(device)
        wave=None
    else:
        wave_np=np.load(info["signals_file"],mmap_mode="r")
        wave=torch.from_numpy(np.array(wave_np[arrays["signal_indices"]],copy=True)).to(device)
        # One source-derived channel scale; preserve variation between acquisitions.
        wave=wave/torch.tensor(arrays["wave_scale"],device=device).view(1,2,1)
        constructor=PhysicalLatentRegressor if model_name=="representation" else TCNRegressor
        model=constructor(2,x.shape[1],96,context.shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    train=arrays["train"]
    val=arrays["validation"]
    prob=arrays["train_sampling_probability"]
    bearing=arrays["bearing_codes"]
    elapsed=arrays["elapsed_hours"]
    out.mkdir(parents=True,exist_ok=True)

    def batch(idx):
        result={"features":x[idx],"context":context[idx]}
        if wave is not None:
            result["signal"]=wave[idx]
        return result

    def predict(idx):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(batch(idx[i:i+256]))["rul"].detach().cpu().numpy() for i in range(0,len(idx),256)])

    best=float("inf")
    stale=0
    trace=[]
    for epoch in range(max_epochs):
        model.train()
        sample=rng.choice(train,size=len(train),replace=True,p=prob)
        losses=[]
        for i in range(0,len(sample),128):
            idx=sample[i:i+128]
            prediction=model(batch(idx))
            loss=F.smooth_l1_loss(prediction["rul"],y[idx],beta=.1)
            if model_name=="attention":
                # Weak surrogate consistency and within-source monotonicity only.
                consistency=F.smooth_l1_loss(prediction["rul"],(1-prediction["damage"])/(prediction["damage_rate"]+1e-4),beta=.1)
                penalties=[]
                for b in np.unique(bearing[idx]):
                    local=np.flatnonzero(bearing[idx]==b)
                    local=local[np.argsort(elapsed[idx[local]])]
                    if len(local)>1:
                        damage=prediction["damage"][local]
                        penalties.append(F.relu(damage[:-1]-damage[1:]).mean())
                monotonic=torch.stack(penalties).mean() if penalties else loss*0
                loss=loss+.05*(consistency+monotonic)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            losses.append(float(loss.detach()))
        p=predict(val)
        metric=float(np.mean(np.abs(p-arrays["target_scaled"][val])))
        trace.append({"epoch":epoch+1,"training_loss":float(np.mean(losses)),"validation_scaled_MAE":metric})
        if metric<best-1e-5:
            best=metric
            stale=0
            torch.save(model.state_dict(),out/"weights.pt")
            best_epoch=epoch+1
        else:
            stale+=1
        if epoch==0 or (epoch+1)%10==0:
            print(f"{model_name} epoch {epoch+1}: validation {metric:.4f}; {time.time()-start:.1f}s",flush=True)
        if stale>=18:
            break
    model.load_state_dict(torch.load(out/"weights.pt",map_location=device,weights_only=True))
    for key in ("calibration","test"):
        np.save(out/(key+"_scaled.npy"),predict(arrays[key]))
    metadata={"seed":seed,"model":model_name,"device":str(device),"torch_version":torch.__version__,
              "best_epoch":best_epoch,"epochs_run":len(trace),"validation_scaled_MAE":best,
              "runtime_seconds":time.time()-start,"target_scale_hours":info["target_scale_hours"],"trace":trace}
    (out/"training.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in metadata.items() if k!="trace"}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--payload",type=Path,required=True)
    parser.add_argument("--model",choices=["representation","tcn","attention"],required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=20260929)
    parser.add_argument("--epochs",type=int,default=120)
    a=parser.parse_args()
    fit(a.payload,a.model,a.seed,a.out,a.epochs)
