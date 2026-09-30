"""Independent fold processes with isolated output folders and persistent logs."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from pathlib import Path
import subprocess
import sys


def main(endpoint=False):
    root=Path("QREI submission/results/endpoint_tabular_groups" if endpoint else "QREI submission/results/lobo_tabular_groups")
    root.mkdir(parents=True,exist_ok=True)
    groups=[("remaining_B02",["B02"]),("group1",["B03","B04"]),("group2",["B08","B10"]),("group3",["B11","B12","B17"])]
    models=["random_forest","standard_boosting","custom_boosting","subspace_ridge","time_only","elapsed_clock","degradation","context_only"]
    if endpoint:
        groups=[("group1",["B02","B03"]),("group2",["B04","B08"]),("group3",["B10","B11"]),("group4",["B12","B17"])]
        models += ["temperature_only","competing_threshold","competing_stop"]
    def task(item):
        name,bearings=item
        out=Path("QREI submission/results/lobo_tabular") if name=="remaining_B02" else root/name
        command=[sys.executable,"-u","-m","bearing_dt.qrei.evaluate","--out",str(out),"--test-bearings",*bearings,"--jobs","4","--models",*models]
        if endpoint:
            command += ["--processed","data/processed/phme_tvoc_10b_endpoint_v3","--protocol","QREI submission/protocol_endpoint_v3.json"]
        print("Start "+name,flush=True)
        with (root/(name+".log")).open("w",encoding="utf-8") as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print("Complete "+name,flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(task,groups))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--endpoint",action="store_true")
    main(parser.parse_args().endpoint)
