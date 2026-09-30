"""Refits registered in protocol_sensitivity_addendum.json: extra seeds and alternative fold roles.

Every run writes to a new folder under results/sensitivity and never touches a primary result.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path("QREI submission/results/sensitivity")
PROCESSED = "data/processed/phme_tvoc_10b_endpoint_v3"
PROTOCOL = "QREI submission/protocol_endpoint_v3.json"
GROUPS = [("group1", ["B02", "B03"]), ("group2", ["B04", "B08"]), ("group3", ["B10", "B11"]), ("group4", ["B12", "B17"])]
TABULAR = ["random_forest", "standard_boosting", "custom_boosting", "subspace_ridge"]
CONTROLS = ["time_only", "elapsed_clock", "degradation", "context_only", "temperature_only", "competing_threshold", "competing_stop"]
SEEDS = [20260930, 20260931]
ALLOCATIONS = ["swap", "reverse"]


def command(out, models, seed, roles, bearings=None, gpu_python="py"):
    cmd = [sys.executable, "-u", "-m", "bearing_dt.qrei.evaluate", "--processed", PROCESSED, "--protocol", PROTOCOL,
           "--out", str(out), "--models", *models, "--seed", str(seed), "--roles", roles, "--jobs", "1",
           "--gpu-python", gpu_python]
    if bearings:
        cmd += ["--test-bearings", *bearings]
    return cmd


def run_logged(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("Start " + str(log), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print("Complete " + str(log), flush=True)


def tabular_jobs():
    jobs = []
    for seed in SEEDS:
        for name, bearings in GROUPS:
            out = ROOT/f"seed_{seed}_tabular"/name
            jobs.append((command(out, TABULAR, seed, "forward", bearings), out.parent/(name + ".log")))
    for roles in ALLOCATIONS:
        for name, bearings in GROUPS:
            out = ROOT/f"roles_{roles}_tabular"/name
            jobs.append((command(out, TABULAR + CONTROLS, 20260929, roles, bearings), out.parent/(name + ".log")))
    return jobs


def neural_jobs(gpu_python):
    jobs = []
    for seed in SEEDS:
        out = ROOT/f"seed_{seed}_neural"
        jobs.append((command(out, ["tcn", "attention"], seed, "forward", gpu_python=gpu_python), ROOT/f"seed_{seed}_neural.log"))
    for roles in ALLOCATIONS:
        out = ROOT/f"roles_{roles}_neural"
        jobs.append((command(out, ["representation", "tcn", "attention"], 20260929, roles, gpu_python=gpu_python), ROOT/f"roles_{roles}_neural.log"))
    return jobs


def main(stage, workers, gpu_python):
    if stage in ("tabular", "all"):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda job: run_logged(*job), tabular_jobs()))
    if stage in ("neural", "all"):
        for job in neural_jobs(gpu_python):
            run_logged(*job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tabular", "neural", "all"], default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu-python", default="py")
    a = parser.parse_args()
    main(a.stage, a.workers, a.gpu_python)
