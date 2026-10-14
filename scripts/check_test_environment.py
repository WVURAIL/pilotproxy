#!/usr/bin/env python3
# coding=utf-8
"""Fail before pytest when required test dependencies are unavailable."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

BASE_MODULES = ("h5py", "matplotlib", "yaml", "scipy", "cryptography")
INTEGRATION_MODULES = ("cadcdata", "cadcutils", "dtcli")


def _module_path(module: object) -> str:
    path = getattr(module, "__file__", None)
    return "built-in" if path is None else str(Path(path).resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--integration",
        action="store_true",
        help="also require archive dependencies for the integration test suite",
    )
    args = parser.parse_args(argv)

    names = BASE_MODULES + (INTEGRATION_MODULES if args.integration else ())
    missing: list[str] = []
    loaded: list[tuple[str, str]] = []
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report the import failure exactly.
            missing.append(f"{name}: {type(exc).__name__}: {exc}")
        else:
            loaded.append((name, _module_path(module)))

    if missing:
        print("required test environment is incomplete:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print(
            'install test dependencies with: python -m pip install -e ".[test]"',
            file=sys.stderr,
        )
        if args.integration:
            print(
                'install archive dependencies with: python -m pip install -e ".[archive,test]"',
                file=sys.stderr,
            )
        return 1

    for name, path in loaded:
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
