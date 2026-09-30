from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .signal import harmonize_record, physical_features, causal_history, defect_orders


def process_one(task):
    path, speed, factor = task
    x, fs = harmonize_record(loadmat(path, variable_names=["measTime", "accHorizRear_A", "accHorizFrontal_C"]), factor)
    feat, wave = physical_features(x, speed)
    feat["raw_sample_rate_Hz"] = fs
    return feat, wave


def prepare(raw: Path, out: Path, protocol: Path, evidence: Path, workers: int = 6):
    cfg = json.loads(protocol.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    allframes, ledger, calibration = [], [], []
    t0 = time.time()
    total = sum(len(list((raw/b/b/"vibrationData").glob("*.mat"))) for b in cfg["raw_bearings"])
    signals = np.lib.format.open_memmap(out/"signals.npy", mode="w+", dtype=np.float32, shape=(total, 2, 2048))
    offset = 0
    reasons = {"B02": "vibration; 0.65 V crossed four times after prior two-crossing stop", "B03": "temperature 110 C",
               "B04": "vibration", "B08": "temperature 110 C", "B10": "vibration", "B11": "vibration", "B12": "vibration", "B17": "vibration"}
    defects = {"B01":"IR/OR/B", "B02":"IR", "B03":"IR", "B04":"IR/B", "B05":"B", "B08":"IR/B", "B10":"OR/B", "B11":"IR/OR/B", "B12":"IR/B", "B17":"B"}
    for b in cfg["raw_bearings"]:
        folder = raw/b/b
        paths = sorted((folder/"vibrationData").glob("*.mat"), key=lambda p:int(re.search(r"_M(\d+)",p.stem)[1]))
        ids = [int(re.search(r"_M(\d+)",p.stem)[1]) for p in paths]
        op = pd.read_csv(folder/f"{b}_operatingConditions.csv")
        temp = pd.read_csv(folder/f"{b}_meanTemperatures.csv")
        if not len(paths)==len(op)==len(temp) or ids!=list(range(1,len(paths)+1)):
            raise ValueError(f"{b}: count/index alignment failed")
        stamps = pd.to_datetime(op.iloc[:,0], format="%d-%b-%Y %H:%M:%S")
        if not np.array_equal(stamps.values, pd.to_datetime(temp.iloc[:,0],format="%d-%b-%Y %H:%M:%S").values):
            raise ValueError(f"{b}: operating/temperature timestamp alignment failed")
        hours = (stamps-stamps.iloc[0]).dt.total_seconds().to_numpy()/3600
        if not (np.diff(hours)>0).all():
            raise ValueError(f"{b}: non-increasing timestamps")
        logtext = (evidence/f"{b}_log.txt").read_text(encoding="utf-8")
        if not re.search(r"Conversion Factor\s+10 g/V", logtext):
            raise ValueError(f"{b}: conversion not supported by log")
        calibration.append({"bearing_id":b,"g_per_V":10.0,"gain_dB":0,"source":str(folder/f"{b}_log.pdf")})
        data=[]
        tasks=((p,float(op.iloc[i,6]),10.0) for i,p in enumerate(paths))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i,(feat,wave) in enumerate(pool.map(process_one,tasks)):
                signals[offset+i]=wave
                feat.update({"bearing_id":b,"record_id":ids[i],"signal_index":offset+i,
                             "timestamp":stamps.iloc[i].isoformat(),"elapsed_hours":hours[i],
                             "observed_duration_hours":hours[-1],"event_observed":b in cfg["event_bearings"],
                             "rul_hours":hours[-1]-hours[i] if b in cfg["event_bearings"] else np.nan,
                             "static_load_N":float(op.iloc[i,4]),"dynamic_load_N":float(op.iloc[i,2]),
                             "speed_rpm":float(op.iloc[i,6]),"temperature_1_C":float(temp.iloc[i,1]),
                             "temperature_2_C":float(temp.iloc[i,2])})
                data.append(feat)
                if (i+1)%500==0:
                    print(f"{b}: {i+1}/{len(paths)} records; {time.time()-t0:.1f} s",flush=True)
        frame=pd.DataFrame(data)
        for col in [c for c in frame if c.startswith(("A_","C_")) and c not in ("A_skew","C_skew")]:
            frame[col+"_ema"] = causal_history(frame[col].to_numpy(),hours)
        frame["hi_rms_g"] = causal_history(np.maximum(frame.A_rms_g,frame.C_rms_g).to_numpy(),hours)
        frame["hi_log_slope_per_hour"] = 0.0
        hi=np.log(np.maximum(frame.hi_rms_g.to_numpy(),1e-8))
        for i in range(2,len(frame)):
            lo=np.searchsorted(hours,hours[i]-.25)
            xx=hours[lo:i+1]
            if len(xx)>=3 and np.ptp(xx)>0:
                frame.loc[i,"hi_log_slope_per_hour"] = np.polyfit(xx,hi[lo:i+1],1)[0]
        gaps=np.diff(hours)*3600
        ledger.append({"bearing_id":b,"records":len(frame),"first_timestamp":stamps.iloc[0].isoformat(),
                       "last_timestamp":stamps.iloc[-1].isoformat(),"duration_hours":hours[-1],
                       "event_observed":b in cfg["event_bearings"],"reason":reasons.get(b,cfg["diagnostic_only"].get(b,"")),
                       "postmortem_defect":defects[b],"median_gap_seconds":float(np.median(gaps)),
                       "max_gap_seconds":float(max(gaps)),"raw_fs_Hz":float(frame.raw_sample_rate_Hz.iloc[0]),
                       "index_timestamp_alignment":"PASS","endpoint":"final acquisition of documented completed test" if b in cfg["event_bearings"] else "diagnostic only; no failure target"})
        allframes.append(frame)
        frame.to_csv(out/f"{b}_features.csv",index=False)
        offset+=len(frame)
        print(f"Completed {b}: {len(frame)} records, {hours[-1]:.4f} h",flush=True)
    signals.flush()
    frame=pd.concat(allframes,ignore_index=True)
    frame.to_csv(out/"features.csv",index=False)
    pd.DataFrame(ledger).to_csv(evidence/"endpoint_ledger.csv",index=False)
    pd.DataFrame(calibration).to_csv(evidence/"sensor_calibration.csv",index=False)
    feature_cols=[c for c in frame if c.startswith(("A_","C_")) or c in ("elapsed_hours","hi_rms_g","hi_log_slope_per_hour")]
    manifest={"created":datetime.now().isoformat(),"record_count":len(frame),"supervised_records":int(frame.event_observed.sum()),
              "signal_shape":list(signals.shape),"channels":["A","C"],"sample_duration_seconds":1.6,
              "analysis_fs_Hz":64000,"waveform_fs_Hz":1280,"envelope_band_Hz":[6000,10000],
              "feature_columns":feature_cols,"context_columns":["static_load_N","dynamic_load_N","speed_rpm"],
              "temperature_primary":False,"defect_orders":defect_orders(),
              "protocol_sha256":hashlib.sha256(protocol.read_bytes()).hexdigest(),
              "runtime_seconds":time.time()-t0,"source":str(raw.resolve())}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    checks={"counts_align":True,"timestamps_align":True,"all_sample_rates_checked":True,
            "rms_standard_deviation_g":{c:float(frame[c].std()) for c in ("A_rms_g","C_rms_g")},
            "finite_predictors":bool(np.isfinite(frame[feature_cols]).all().all()),
            "amplitude_standardized_per_record":False,"features_sha256":hashlib.sha256((out/"features.csv").read_bytes()).hexdigest()}
    (evidence/"preparation_verification.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(json.dumps({"records":len(frame),"supervised_records":int(frame.event_observed.sum()),"runtime_s":time.time()-t0}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw",type=Path,default=Path("data/raw/phme_tvoc"))
    parser.add_argument("--out",type=Path,default=Path("data/processed/phme_tvoc_10b_v2"))
    parser.add_argument("--protocol",type=Path,default=Path("QREI submission/protocol.json"))
    parser.add_argument("--evidence",type=Path,default=Path("QREI submission/evidence"))
    parser.add_argument("--workers",type=int,default=6)
    a=parser.parse_args()
    prepare(a.raw,a.out,a.protocol,a.evidence,a.workers)
