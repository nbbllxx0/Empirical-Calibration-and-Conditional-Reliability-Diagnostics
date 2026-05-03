from __future__ import annotations

import argparse
from pathlib import Path

from bearing_dt.config import read_yaml
from bearing_dt.train import train_experiment


def run_suite(suite_path: str | Path, dry_run: bool = False) -> list[Path]:
    suite_path = Path(suite_path)
    suite = read_yaml(suite_path)
    experiments = suite.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"No experiments listed in {suite_path}")
    run_dirs = []
    for entry in experiments:
        cfg = Path(entry["config"])
        if not cfg.is_absolute():
            cfg = suite_path.parent.parent.parent / cfg
        cfg = cfg.resolve()
        if not cfg.exists():
            raise FileNotFoundError(cfg)
        config = read_yaml(cfg)
        processed_dir = Path(config.get("data", {}).get("processed_dir", ""))
        if not processed_dir.exists():
            message = f"Processed data missing for {entry.get('name', cfg.name)}: {processed_dir}"
            if dry_run:
                print(f"DRY-RUN WARNING: {message}")
            else:
                raise FileNotFoundError(message)
        print(f"{'Would run' if dry_run else 'Running'} {entry.get('name', cfg.stem)} -> {cfg}")
        if not dry_run:
            run_dirs.append(train_experiment(cfg))
    return run_dirs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bearing_dt.suite")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--suite", required=True)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        run_suite(args.suite, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
