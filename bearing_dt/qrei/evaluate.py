from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from .learners import WeightedStumpBoosting, SubspaceRidge
from .endpoint import CompetingStopModel, threshold_predictions
from .roles import ROLE_OFFSETS, fold_roles


LEARNED=("representation","random_forest","standard_boosting","custom_boosting","tcn","attention","subspace_ridge")
CONTROLS=("time_only","elapsed_clock","degradation","context_only")


def input_matrices(frame,manifest,train):
    cols=manifest["feature_columns"]
    raw=frame[cols].to_numpy(dtype=float)
    # Fixed transform keeps signed skew and growth rate. Source-only mean/SD follow.
    raw=np.sign(raw)*np.log1p(np.abs(raw))
    for c in ("elapsed_hours",):
        raw[:,cols.index(c)]=frame[c]
    mean=raw[train].mean(axis=0)
    std=np.maximum(raw[train].std(axis=0),1e-6)
    feat=np.clip((raw-mean)/std,-20,20).astype(np.float32)
    ctx=frame[manifest["context_columns"]].to_numpy(dtype=float)
    ctxmean=ctx[train].mean(axis=0)
    ctxstd=np.maximum(ctx[train].std(axis=0),1e-6)
    ctx=np.clip((ctx-ctxmean)/ctxstd,-20,20).astype(np.float32)
    return feat,ctx,{"feature_mean":mean.tolist(),"feature_std":std.tolist(),"context_mean":ctxmean.tolist(),"context_std":ctxstd.tolist(),"feature_columns":cols}


def degradation_predictions(frame,train,idx):
    """Empirical exponential threshold extrapolation with causal target updates."""
    source=frame.iloc[train]
    early=source[source.elapsed_hours<=.25]
    context_cols=["static_load_N","dynamic_load_N","speed_rpm"]
    ctx=np.log1p(frame[context_cols].to_numpy())
    reg=Ridge(alpha=1.0).fit(ctx[early.index],np.log(np.maximum(early.hi_rms_g,1e-6)))
    loghi=np.log(np.maximum(frame.hi_rms_g.to_numpy(),1e-6))-reg.predict(ctx)+float(reg.intercept_)
    rates=[]
    for _,group in source.groupby("bearing_id"):
        ii=group.index.to_numpy()
        rates.append(np.polyfit(group.elapsed_hours,loghi[ii],1)[0])
    positive=[r for r in rates if r>0]
    prior=float(np.median(positive)) if positive else .1
    cap=2*float(source.observed_duration_hours.max())
    result=[]
    for i in idx:
        row=frame.iloc[i]
        prefix=np.flatnonzero((frame.bearing_id.to_numpy()==row.bearing_id)&(frame.elapsed_hours.to_numpy()<=row.elapsed_hours)&(frame.elapsed_hours.to_numpy()>=row.elapsed_hours-.5))
        rate=prior
        if len(prefix)>=5 and row.elapsed_hours>=.1:
            recent=np.polyfit(frame.elapsed_hours.to_numpy()[prefix],loghi[prefix],1)[0]
            rate=max(.02,.5*prior+.5*max(recent,0))
        result.append(np.clip((np.log(8.0)-loghi[i])/rate,0,cap))
    return np.asarray(result),{"threshold_g":8.0,"threshold_basis":"fixed midpoint of documented 6-10 g range; not test-specific fitted threshold","prior_log_growth_per_hour":prior,"context_coefficients":reg.coef_.tolist(),"context_intercept":float(reg.intercept_),"forecast_cap_hours":cap}


def run(processed:Path,out:Path,protocol:Path,models:list[str],gpu_python:str="py",split="lobo",seed=20260929,epochs=120,test_bearings=None,jobs=1,roles="forward"):
    threadpool_limits(limits=1)
    cfg=json.loads(protocol.read_text(encoding="utf-8"))
    manifest=json.loads((processed/"manifest.json").read_text(encoding="utf-8"))
    frame=pd.read_csv(processed/"features.csv")
    out.mkdir(parents=True,exist_ok=True)
    (out/"protocol.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    frame["physical_regime"]=(np.digitize(frame.static_load_N+frame.dynamic_load_N,[3500,4250])*3+np.digitize(frame.speed_rpm,[1800,2500])).astype(int)
    bearings=cfg["event_bearings"]
    all_predictions=[]
    selected=bearings if test_bearings is None else test_bearings
    if not set(selected).issubset(bearings):
        raise ValueError("Unknown supervised test bearing")
    for test in selected:
        train_b,val_b,cal_b=fold_roles(bearings,test,roles)
        groups={"train":np.flatnonzero(frame.bearing_id.isin(train_b)),"validation":np.flatnonzero(frame.bearing_id==val_b),
                "calibration":np.flatnonzero(frame.bearing_id==cal_b),"test":np.flatnonzero(frame.bearing_id==test)}
        regime=None
        if split=="joint":
            regime=int(frame.iloc[groups["test"]].physical_regime.mode().iloc[0])
            for key in groups:
                cond=(frame.iloc[groups[key]].physical_regime.to_numpy()==regime)
                groups[key]=groups[key][cond if key=="test" else ~cond]
            if any(len(groups[k])<30 for k in groups) or len(groups["train"])<200:
                print(f"Joint {test} cell {regime}: unsupported roles; skipped",flush=True)
                continue
        fold=out/test
        fold.mkdir(exist_ok=True)
        splitmeta={"test":test,"training_bearings":train_b,"validation_bearing":val_b,"calibration_bearing":cal_b,
                   "split":split,"excluded_regime":regime,"role_counts":{k:len(v) for k,v in groups.items()},
                   "protocol_sha256":hashlib.sha256(protocol.read_bytes()).hexdigest(),"seed":seed}
        if roles!="forward":
            splitmeta["roles"]=roles
        if (fold/"split.json").exists():
            previous=json.loads((fold/"split.json").read_text(encoding="utf-8"))
            if previous != splitmeta:
                raise ValueError("Resume requested with a different protocol or split; use a new output folder")
        (fold/"split.json").write_text(json.dumps(splitmeta,indent=2),encoding="utf-8")
        np.savez(fold/"split_indices.npz",**groups)
        x,ctx,scaler=input_matrices(frame,manifest,groups["train"])
        durations=frame.iloc[groups["train"]].groupby("bearing_id").observed_duration_hours.first()
        scale=float(durations.median())
        cap=2*float(durations.max())
        y=frame.rul_hours.to_numpy()/scale
        train=groups["train"]
        counts=frame.iloc[train].bearing_id.value_counts()
        weights=np.array([1/counts[b] for b in frame.iloc[train].bearing_id])
        weights=weights/np.mean(weights)
        design=np.column_stack([x,ctx])
        (fold/"scaling.json").write_text(json.dumps({**scaler,"target_scale_hours":scale,"prediction_cap_hours":cap},indent=2),encoding="utf-8")
        signals_file=Path(manifest.get("signals_file",str((processed/"signals.npy").resolve())))
        signal_np=np.load(signals_file,mmap_mode="r")
        wave_scale=np.sqrt(np.mean(signal_np[frame.iloc[train].signal_index.to_numpy()[::10]]**2,axis=(0,2))).astype(np.float32)
        np.savez(fold/"payload.npz",features=x,context=ctx,target_scaled=y.astype(np.float32),
                 signal_indices=frame.signal_index.to_numpy(),wave_scale=np.maximum(wave_scale,1e-6),
                 bearing_codes=pd.Categorical(frame.bearing_id).codes,elapsed_hours=frame.elapsed_hours.to_numpy(),
                 train_sampling_probability=weights/np.sum(weights),**groups)
        (fold/"payload.json").write_text(json.dumps({"signals_file":str(signals_file.resolve()),"target_scale_hours":scale}),encoding="utf-8")
        for name in models:
            modeldir=fold/name
            modeldir.mkdir(exist_ok=True)
            result_path=modeldir/"predictions.csv"
            if result_path.exists():
                existing=pd.read_csv(result_path)
                expected=frame.iloc[groups["test"]].record_id.to_numpy()
                if not np.array_equal(existing.record_id.to_numpy(),expected):
                    raise ValueError("Resume prediction rows do not match frozen split")
                all_predictions.append(existing)
                print(f"Resume {test} {name}",flush=True)
                continue
            start=time.time()
            extra={}
            cp,tp=None,None
            if name in ("representation","tcn","attention"):
                command=[gpu_python,"-u","-m","bearing_dt.qrei.neural"]
                if gpu_python=="py":
                    command=["py","-3.12","-u","-m","bearing_dt.qrei.neural"]
                command += ["--payload",str(fold),"--model",name,"--out",str(modeldir),"--seed",str(seed),"--epochs",str(epochs)]
                print(f"Start {test} {name} on GPU",flush=True)
                subprocess.run(command,check=True)
                cp=np.load(modeldir/"calibration_scaled.npy")*scale
                tp=np.load(modeldir/"test_scaled.npy")*scale
            elif name=="time_only":
                cp=np.maximum(scale-frame.iloc[groups["calibration"]].elapsed_hours.to_numpy(),0)
                tp=np.maximum(scale-frame.iloc[groups["test"]].elapsed_hours.to_numpy(),0)
            elif name=="elapsed_clock":
                cp=frame.iloc[groups["calibration"]].elapsed_hours.to_numpy()
                tp=frame.iloc[groups["test"]].elapsed_hours.to_numpy()
            elif name=="degradation":
                cp,extra=degradation_predictions(frame,train,groups["calibration"])
                tp,_=degradation_predictions(frame,train,groups["test"])
            elif name=="competing_threshold":
                cp,extra=threshold_predictions(frame,train,groups["calibration"])
                tp,_=threshold_predictions(frame,train,groups["test"])
            elif name=="competing_stop":
                model=CompetingStopModel().fit(frame,train)
                cp=model.predict(frame.iloc[groups["calibration"]])
                tp=model.predict(frame.iloc[groups["test"]])
                extra=model.metadata
                with (modeldir/"model.pkl").open("wb") as f:
                    pickle.dump(model,f)
            else:
                if name=="random_forest" or name=="random_forest_no_elapsed":
                    model=RandomForestRegressor(n_estimators=400,max_features=.75,min_samples_leaf=5,n_jobs=jobs,random_state=seed)
                elif name=="standard_boosting":
                    model=GradientBoostingRegressor(n_estimators=260,learning_rate=.035,max_depth=3,min_samples_leaf=10,random_state=seed)
                elif name=="custom_boosting":
                    model=WeightedStumpBoosting(seed=seed)
                elif name=="subspace_ridge":
                    model=SubspaceRidge(seed=seed)
                elif name in ("context_only","temperature_only"):
                    model=HistGradientBoostingRegressor(max_iter=260,learning_rate=.035,max_leaf_nodes=15,l2_regularization=1,random_state=seed)
                else:
                    raise ValueError(name)
                xx=design
                if name=="context_only":
                    xx=np.column_stack([frame.elapsed_hours.to_numpy(),ctx])
                elif name=="temperature_only":
                    thermal=[i for i,c in enumerate(manifest["feature_columns"]) if c.startswith("ep_") and "vibration" not in c]
                    xx=np.column_stack([frame.elapsed_hours.to_numpy(),ctx,x[:,thermal]])
                elif name=="random_forest_no_elapsed":
                    xx=np.delete(design,manifest["feature_columns"].index("elapsed_hours"),axis=1)
                model.fit(xx[train],y[train],sample_weight=weights)
                cp=model.predict(xx[groups["calibration"]])*scale
                tp=model.predict(xx[groups["test"]])*scale
                with (modeldir/"model.pkl").open("wb") as f:
                    pickle.dump(model,f)
            cp=np.clip(cp,0,cap)
            tp=np.clip(tp,0,cap)
            residual=np.abs(cp-frame.iloc[groups["calibration"]].rul_hours.to_numpy())
            q=float(np.quantile(residual,.9,method="higher"))
            output=frame.iloc[groups["test"]][["bearing_id","record_id","timestamp","elapsed_hours","observed_duration_hours","rul_hours","physical_regime"]].copy()
            output["model"]=name
            output["pred_hours"]=tp
            output["lower_hours"]=np.maximum(0,tp-q)
            output["upper_hours"]=tp+q
            output["training_scale_hours"]=scale
            output["calibration_radius_hours"]=q
            output["split"]=split
            output.to_csv(result_path,index=False)
            (modeldir/"fit_metadata.json").write_text(json.dumps({"runtime_seconds":time.time()-start,"target_scale_hours":scale,
                "calibration_radius_hours":q,"future_target_lifetime_used_for_prediction":False,"extra":extra},indent=2),encoding="utf-8")
            all_predictions.append(output)
            print(f"Completed {test} {name}: nMAE {np.mean(np.abs(tp-output.rul_hours))/float(output.observed_duration_hours.iloc[0]):.4f}; {time.time()-start:.1f}s",flush=True)
        pd.concat(all_predictions,ignore_index=True).to_csv(out/"joined_predictions.csv",index=False)
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--processed",type=Path,default=Path("data/processed/phme_tvoc_10b_v2"))
    p.add_argument("--out",type=Path,default=Path("QREI submission/results/lobo"))
    p.add_argument("--protocol",type=Path,default=Path("QREI submission/protocol.json"))
    p.add_argument("--models",nargs="+",default=list(LEARNED+CONTROLS))
    p.add_argument("--split",choices=["lobo","joint"],default="lobo")
    p.add_argument("--seed",type=int,default=20260929)
    p.add_argument("--epochs",type=int,default=120)
    p.add_argument("--test-bearings",nargs="+")
    p.add_argument("--jobs",type=int,default=1)
    p.add_argument("--roles",choices=sorted(ROLE_OFFSETS),default="forward",help="Fold-role allocation; 'forward' is the primary design")
    p.add_argument("--gpu-python",default="py",help="Neural-worker interpreter; 'py' selects the Windows Python 3.12 launcher")
    a=p.parse_args()
    run(a.processed,a.out,a.protocol,a.models,gpu_python=a.gpu_python,split=a.split,seed=a.seed,epochs=a.epochs,test_bearings=a.test_bearings,jobs=a.jobs,roles=a.roles)
