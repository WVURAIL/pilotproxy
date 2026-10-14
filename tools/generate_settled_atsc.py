#!/usr/bin/env python3
"""Generate a new ATSC waveform with an explicit discarded startup and receipt.

The discard is a declared experiment setting, not a universal stationarity
proof. Existing waveform files and historical identities are never overwritten.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pilot_proxy.testbench.audit_atsc_signal import audit_atsc_iq


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-iq-samples", type=int, default=600000)
    parser.add_argument("--discard-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-ts-packets", type=int, default=4096)
    parser.add_argument("--gnuradio-python", default="/usr/bin/python3")
    args = parser.parse_args()
    if args.num_iq_samples < 262144 or args.discard_samples < 0 or args.num_ts_packets < 1:
        parser.error("need at least262144 retained samples, nonnegative discard and positive packet count")
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("new waveform generation requires an empty directory")
    out.mkdir(parents=True, exist_ok=True)
    generator = ROOT / "src/pilot_proxy/testbench/generate_atsc_signal.py"
    raw, settled, ts = out/"as-generated.cfile", out/"settled.cfile", out/"transport-stream.ts"
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", PYTHONPATH=str(ROOT/"src"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    probe = "import json,sys,numpy;from gnuradio import gr;print(json.dumps({'python':sys.version,'executable':sys.executable,'numpy':numpy.__version__,'gnuradio':gr.version()}))"
    runtime = json.loads(subprocess.check_output([args.gnuradio_python, "-c", probe], env=env, text=True))
    command = [args.gnuradio_python, str(generator), "--output-iq", str(raw), "--output-ts", str(ts), "--num-iq-samples", str(args.num_iq_samples+args.discard_samples), "--num-ts-packets", str(args.num_ts_packets), "--seed", str(args.seed)]
    plan = {"schema": "pilotproxy-stationary-waveform-plan-v1", "frozen_utc": datetime.now(timezone.utc).isoformat(), "command": command, "retained_samples": args.num_iq_samples, "discard_samples": args.discard_samples, "generation_runtime": runtime, "source_sha256": {str(generator): sha(generator), str(Path(__file__).resolve()): sha(__file__)}, "interpretation": "Explicit startup discard for this experiment; empirical diagnostics remain necessary. It does not reconstruct earlier waveform generation."}
    for module in list(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source and Path(source).suffix == ".py" and Path(source).is_relative_to(ROOT):
            plan["source_sha256"][str(Path(source).resolve())] = sha(source)
    plan["audit_runtime"] = {"python": sys.version, "numpy": np.__version__}
    dump(out/"plan.json", plan)
    with (out/"generation.log").open("x") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for source, expected in plan["source_sha256"].items():
        if sha(source) != expected:
            raise ValueError("generating source changed during generation")
    iq = np.fromfile(raw, dtype=np.complex64)
    if iq.size != args.num_iq_samples+args.discard_samples or not np.isfinite(iq).all():
        raise ValueError("generated waveform count or finiteness failed")
    kept = iq[args.discard_samples:]
    with settled.open("xb") as stream:
        stream.write(kept.tobytes())
    if not np.array_equal(np.fromfile(settled, dtype=np.complex64), kept):
        raise ValueError("settled output is not the exact prescribed trim")
    audit = audit_atsc_iq(input_iq=settled)
    dump(out/"waveform-audit.json", audit)
    fs = audit["sample_rate_hz"]
    pilot = audit["measured_pilot_frequency_hz"]
    chunks = []
    for start in range(0, kept.size-65536+1, 65536):
        part = kept[start:start+65536].astype(np.complex128)
        carrier = np.exp(2j*np.pi*pilot*np.arange(part.size)/fs)
        amp = np.vdot(carrier, part)/part.size
        chunks.append({"start": start, "samples": part.size, "total_power": float(np.mean(np.abs(part)**2)), "coherent_pilot_power": float(abs(amp)**2)})
    receipt = {"schema": "pilotproxy-stationary-waveform-receipt-v1", "completed_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha(out/"plan.json"), "raw_iq_sha256": sha(raw), "settled_iq_sha256": sha(settled), "transport_stream_sha256": sha(ts), "generator_metadata_sha256": sha(raw.with_suffix(".cfile.json")), "waveform_audit_sha256": sha(out/"waveform-audit.json"), "raw_samples": len(iq), "retained_samples": len(kept), "discard_samples": args.discard_samples, "exact_trim_verified": True, "waveform_quality_passed": bool(audit["quality_passed"]), "segment_diagnostics": chunks, "scope": "Deterministic sample generation and declared trim, not physical RF calibration or independent multi-waveform validation."}
    dump(out/"receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    if not audit["quality_passed"]:
        raise SystemExit("waveform audit failed; retained artifacts are diagnostic only")


if __name__ == "__main__":
    main()
