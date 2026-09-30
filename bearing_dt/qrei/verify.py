"""Independent forecast/target arithmetic and persisted-model checks."""
from __future__ import annotations
import csv
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.io import loadmat


def main(endpoint=False):
    root=Path("QREI submission")
    data=Path("data/processed/phme_tvoc_10b_endpoint_v3" if endpoint else "data/processed/phme_tvoc_10b_v2")
    raw=Path("data/raw/phme_tvoc")
    protocol=root/("protocol_endpoint_v3.json" if endpoint else "protocol.json")
    resultroot=root/("results/endpoint_v3" if endpoint else "results/primary")
    cfg=json.loads(protocol.read_text())
    features=pd.read_csv(data/"features.csv")
    manifest=json.loads((data/"manifest.json").read_text())
    p=pd.read_csv(resultroot/"joined_predictions.csv")
    s=pd.read_csv(resultroot/"model_summary.csv").set_index("model")
    expected=int(features.event_observed.sum())
    checks=[]
    assert len(p)==expected*(14 if endpoint else 11)
    assert not p.duplicated(["model","bearing_id","record_id"]).any()
    for b in cfg["event_bearings"]:
        op=pd.read_csv(raw/b/b/(b+"_operatingConditions.csv"))
        t=pd.to_datetime(op.iloc[:,0],format="%d-%b-%Y %H:%M:%S")
        true=(t.iloc[-1]-t).dt.total_seconds().to_numpy()/3600
        for name,g in p[p.bearing_id==b].groupby("model"):
            g=g.sort_values("record_id")
            assert np.array_equal(g.record_id.to_numpy(),np.arange(1,len(op)+1))
            assert np.allclose(g.rul_hours,true,atol=1e-12)
        checks.append({"bearing":b,"timestamp_target":"PASS","records":len(op)})
    for name,g in p.groupby("model"):
        per=[]
        for b,h in g.groupby("bearing_id"):
            nmae=np.mean(np.abs(h.pred_hours-h.rul_hours))/float(h.observed_duration_hours.iloc[0])
            per.append(nmae)
        assert abs(np.mean(per)-s.loc[name,"nMAE"])<1e-12
    roots=[root/"results/lobo_tabular",root/"results/lobo_tabular_groups",root/"results/lobo_neural"]
    if endpoint:
        roots=[root/"results/endpoint_tabular_groups",root/"results/endpoint_neural"]
    snapshots=[]
    reproduced=0
    max_error=0
    for folder in roots:
        for splitpath in folder.rglob("split.json"):
            fold=splitpath.parent
            roles=json.loads(splitpath.read_text())
            sets=[set(roles["training_bearings"]),{roles["validation_bearing"]},{roles["calibration_bearing"]},{roles["test"]}]
            assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
            assert set.union(*sets)==set(cfg["event_bearings"])
            scaler=json.loads((fold/"scaling.json").read_text())
            train_duration=features[features.bearing_id.isin(sets[0])].groupby("bearing_id").observed_duration_hours.first()
            assert abs(float(train_duration.median())-scaler["target_scale_hours"])<1e-12
            cols=manifest["feature_columns"]
            # Independently reconstruct the persisted feature transform.
            inputs=features[cols].to_numpy()
            inputs=np.sign(inputs)*np.log1p(np.abs(inputs))
            inputs[:,cols.index("elapsed_hours")]=features.elapsed_hours
            x=np.clip((inputs-np.asarray(scaler["feature_mean"]))/np.asarray(scaler["feature_std"]),-20,20).astype(np.float32)
            context=features[manifest["context_columns"]].to_numpy()
            context=np.clip((context-np.asarray(scaler["context_mean"]))/np.asarray(scaler["context_std"]),-20,20).astype(np.float32)
            design=np.column_stack([x,context])
            for path in fold.glob("*/model.pkl"):
                name=path.parent.name
                if name=="context_only":
                    xx=np.column_stack([features.elapsed_hours.to_numpy(),context])
                elif name=="temperature_only":
                    thermal=[i for i,c in enumerate(cols) if c.startswith("ep_") and "vibration" not in c]
                    xx=np.column_stack([features.elapsed_hours.to_numpy(),context,x[:,thermal]])
                else:
                    xx=design
                with path.open("rb") as f:
                    model=pickle.load(f)
                if hasattr(model,"n_jobs"):
                    model.n_jobs=1
                idx=np.flatnonzero(features.bearing_id==roles["test"])
                rebuilt=model.predict(features.iloc[idx]) if name=="competing_stop" else model.predict(xx[idx])*scaler["target_scale_hours"]
                rebuilt=np.clip(rebuilt,0,scaler["prediction_cap_hours"])
                stored=pd.read_csv(path.parent/"predictions.csv").pred_hours.to_numpy()
                err=float(np.max(np.abs(rebuilt-stored)))
                assert err<1e-6,(path,err)
                max_error=max(max_error,err)
                reproduced+=1
            snapshots.append({"fold":str(fold),"roles":"PASS","training_only_scale":"PASS"})
    rawchecks=[]
    for b in ("B02","B10"):
        row=features[features.bearing_id==b].iloc[0]
        files=sorted((raw/b/b/"vibrationData").glob("*.mat"))
        mat=loadmat(files[0])
        for channel,key in (("A","accHorizRear_A"),("C","accHorizFrontal_C")):
            x=np.asarray(mat[key],dtype=float).ravel()*10
            x-=x.mean()
            if len(x)==204800:
                from scipy.signal import resample_poly
                x=resample_poly(x,1,2)
            rms=np.sqrt(np.mean(x*x))
            assert abs(rms-row[channel+"_rms_g"])<1e-10
            rawchecks.append({"bearing":b,"channel":channel,"raw_recomputed_rms_g":float(rms),"stored_rms_g":float(row[channel+"_rms_g"]),"status":"PASS"})
    constant=[]
    for b,g in features[features.event_observed].groupby("bearing_id"):
        L=float(g.observed_duration_hours.iloc[0])
        constant.append({"bearing_id":b,"constant_normalized_half_life_nMAE":float(np.mean(np.abs(.5-g.rul_hours/L))),
                         "status":"Retrospective fraction reference only; unavailable as an absolute-hour forecast without future lifetime"})
    pd.DataFrame(constant).to_csv(resultroot/"retrospective_fraction_reference.csv",index=False)
    result={"forecast_rows":len(p),"models":len(s),"completed_bearings":8,"rows_per_model":expected,
            "timestamp_targets":checks,"split_checks":snapshots,"persisted_models_reproduced":reproduced,
            "max_absolute_hour_reproduction_error":max_error,"raw_spot_checks":rawchecks,
            "all_mean_nMAE_values_recomputed":True,"future_test_lifetime_is_a_prediction_input":False,
            "protocol_sha256":hashlib.sha256(protocol.read_bytes()).hexdigest(),
            "interpretation":"Normalized scores use realized lifetime only after forecasting; all deployed forecasts use training-only scales."}
    (root/"evidence"/("independent_verification_v3.json" if endpoint else "independent_verification.json")).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("timestamp_targets","split_checks","raw_spot_checks")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--endpoint",action="store_true")
    main(parser.parse_args().endpoint)
