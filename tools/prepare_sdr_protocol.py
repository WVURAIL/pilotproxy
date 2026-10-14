#!/usr/bin/env python3
"""Validate or freeze an offline OTA protocol; never opens a radio."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from pilot_proxy.testbench.sdr_protocol import freeze_protocol, validate_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "freeze"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        spec = json.loads(args.spec.read_text())
        result = validate_spec(spec)
        if args.stage == "freeze":
            if args.output is None:
                parser.error("freeze requires --output")
            freeze_protocol(args.output, spec)
        print(json.dumps({"protocol_valid": True, "hardware_adapter_available": False,
                          "power_check": result}, indent=2))
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps({"protocol_valid": False, "hardware_adapter_available": False,
                          "reason": str(error)}, indent=2))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
