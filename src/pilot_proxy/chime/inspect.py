# coding=utf-8
"""Small HDF5 inventory tool for CHIME pilot samples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py


def _walk(path: Path, filename_pattern: str | None = None) -> list[Path]:
    if filename_pattern is not None:
        return sorted(path.rglob(filename_pattern))
    return sorted(path.rglob("*.h5")) + sorted(path.rglob("*.hdf5"))


def _print_attrs(prefix: str, attrs: Any) -> None:
    for key, value in attrs.items():
        print(f"{prefix} attr {key} = {value!r}")


def inspect_file(path: Path, *, dataset_path: str | None = None) -> None:
    print(f"\nFILE {path}")
    with h5py.File(path, "r") as h5:
        _print_attrs("  /", h5.attrs)
        if dataset_path is not None:
            if dataset_path not in h5:
                print(f"  {dataset_path} not found")
                return
            obj = h5[dataset_path]
            if isinstance(obj, h5py.Dataset):
                print(f"  /{dataset_path} shape={obj.shape} dtype={obj.dtype}")
                _print_attrs(f"    /{dataset_path}", obj.attrs)
            else:
                print(f"  /{dataset_path} is not a dataset")
            return

        def visitor(name: str, h5_obj: Any) -> None:
            if isinstance(h5_obj, h5py.Dataset):
                print(f"  /{name} shape={h5_obj.shape} dtype={h5_obj.dtype}")
                _print_attrs(f"    /{name}", h5_obj.attrs)

        h5.visititems(visitor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--filename-pattern", default=None)
    args = parser.parse_args()

    files = _walk(args.input_dir, args.filename_pattern)
    print(f"Found {len(files)} HDF5 files under {args.input_dir}")
    for path in files[: args.max_files]:
        inspect_file(path, dataset_path=args.dataset_path)


if __name__ == "__main__":
    main()
