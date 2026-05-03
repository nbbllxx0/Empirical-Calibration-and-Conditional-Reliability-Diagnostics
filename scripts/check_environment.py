"""Validate the Python environment used for paper reproduction."""

from __future__ import annotations

import argparse
import importlib
import sys


EXPECTED = {
    "numpy": "2.2.6",
    "yaml": "6.0.2",
    "sklearn": "1.7.2",
    "torch": "2.5.1",
}


def module_version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if PyTorch cannot see a CUDA device.",
    )
    args = parser.parse_args()

    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    failed = False
    for module_name, expected in EXPECTED.items():
        try:
            actual = module_version(module_name)
        except Exception as exc:  # pragma: no cover - diagnostic script
            print(f"{module_name}: import failed: {exc}")
            failed = True
            continue
        print(f"{module_name}: {actual}")
        if actual != expected:
            print(f"  expected {expected}")
            failed = True

    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        print(f"torch CUDA version: {cuda_version}")
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        elif args.require_cuda:
            failed = True
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"torch CUDA check failed: {exc}")
        failed = True

    if failed:
        print("Environment check failed. Use the documented CUDA Conda interpreter for full reproduction.")
        return 1

    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
