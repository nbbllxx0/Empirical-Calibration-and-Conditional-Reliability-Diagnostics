from __future__ import annotations

import argparse

from bearing_dt.data.fetch import fetch_phme_tvoc, fetch_xjtu_sy
from bearing_dt.data.prepare import prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bearing_dt.data")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Fetch or register external dataset files.")
    fetch.add_argument("--dataset", required=True, help="phme_tvoc or xjtu_sy")
    fetch.add_argument("--out", required=True, help="Raw data output directory.")
    fetch.add_argument("--file", action="append", default=None, help="Specific file to fetch. Repeatable.")
    fetch.add_argument("--manifest-only", action="store_true", help="Only write source/file manifests.")
    fetch.add_argument("--extract", action="store_true", help="Extract downloaded zip files.")
    fetch.add_argument("--no-md5", action="store_true", help="Skip MD5 verification.")
    fetch.add_argument("--max-gb", type=float, default=None, help="Refuse downloads larger than this total size.")
    prepare = sub.add_parser("prepare", help="Prepare raw bearing data into the processed experiment format.")
    prepare.add_argument("--dataset", required=True, help="synthetic, phme_tvoc, xjtu_sy, nasa_ims, femto, time_varying_oc, generic_bearing")
    prepare.add_argument("--raw", default=None, help="Raw dataset root. Not required for synthetic.")
    prepare.add_argument("--out", required=True, help="Processed output directory.")
    prepare.add_argument("--window-size", type=int, default=512)
    prepare.add_argument("--sample-rate", type=float, default=25_600.0)
    prepare.add_argument("--seed", type=int, default=7)
    prepare.add_argument("--bearings", type=int, default=8, help="Synthetic bearing count.")
    prepare.add_argument("--steps", type=int, default=48, help="Synthetic steps per bearing.")
    prepare.add_argument("--max-windows", type=int, default=None, help="Limit raw windows for smoke runs.")
    prepare.add_argument("--regime-bins", type=int, default=3, help="Number of quantile bins per load/speed axis.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "fetch":
        dataset = args.dataset.lower()
        if dataset in {"phme_tvoc", "phme", "time_varying_oc", "paderborn_tvoc"}:
            result = fetch_phme_tvoc(
                out=args.out,
                files=args.file,
                manifest_only=args.manifest_only,
                extract=args.extract,
                verify_md5=not args.no_md5,
                max_gb=args.max_gb,
            )
        elif dataset == "xjtu_sy":
            result = fetch_xjtu_sy(args.out)
        else:
            raise ValueError(f"Unsupported fetch dataset: {args.dataset}")
        print(result)
    elif args.command == "prepare":
        manifest = prepare_dataset(
            dataset=args.dataset,
            raw=args.raw,
            out=args.out,
            window_size=args.window_size,
            sample_rate=args.sample_rate,
            seed=args.seed,
            synthetic_bearings=args.bearings,
            synthetic_steps=args.steps,
            max_windows=args.max_windows,
            regime_bins=args.regime_bins,
        )
        print(f"Prepared {manifest['samples']} samples at {args.out}")


if __name__ == "__main__":
    main()
