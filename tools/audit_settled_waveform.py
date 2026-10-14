#!/usr/bin/env python3
"""Recount archived transfer powers and recompute clean waveform diagnostics.

This audit records current source/input identities. It does not reconstruct the
unrecorded historical source revision or claim independent RF calibration.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import current_geometry_sensitivity as geometry
from pilot_proxy.detector_reference import unpack_packed_complex
from pilot_proxy.testbench.evaluate_snr import estimate_complex_scale


def identity(path):
    return {"path": str(path.absolute()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def integrate(freq, psd, low, high):
    inside = (freq > low) & (freq < high)
    x = np.r_[low, freq[inside], high]
    y = np.r_[np.interp(low, freq, psd), psd[inside], np.interp(high, freq, psd)]
    return float(np.trapezoid(y, x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--products", type=Path, default=ROOT.parent / "products/estimator_transfer_2026-09-07")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fs = geometry.GNU_RADIO_ATSC_SYMBOL_RATE_HZ
    width = 6000000.0
    bin_width = 390625.0 / 128
    pilot = -width / 2 + geometry.ATSC_PILOT_OFFSET_HZ
    required = geometry.required_iq_samples(iq_sample_rate_hz=fs, adc_sample_rate_hz=geometry.REFERENCE_ADC_SAMPLE_RATE_HZ, num_output_samples=geometry.FRAME_SAMPLES)
    outputs = {}
    sources = [identity(Path(__file__))]
    iq_arrays = {}
    for label, suffix in [("as_generated", ""), ("settled", "_settled")]:
        iq_path = ROOT / f"generated/atsc/atsc_8vsb_complex64{suffix}.cfile"
        audit_path = ROOT / f"generated/atsc/atsc_waveform_audit{suffix}.json"
        report_path = args.products / "saturation_control" / label / "dtv_snr_eval.json"
        csv_path = report_path.with_suffix(".csv")
        summary_path = report_path.with_name("dtv_snr_summary.csv")
        sources.extend(identity(p) for p in [iq_path, audit_path, report_path, csv_path, summary_path])
        report = json.loads(report_path.read_text())
        audit = json.loads(audit_path.read_text())
        for key in ("weight_bank", "weight_manifest"):
            current = identity(ROOT / report[key]["path"])
            if current["sha256"] != report[key]["sha256"]:
                raise ValueError(f"{label}: current {key} differs from archived report")
            sources.append(current)
        layout = report["selected_weight_layout"]
        iq = np.fromfile(iq_path, dtype=np.complex64)
        iq_arrays[label] = iq
        frame = iq[:required].astype(np.complex128)
        carrier = np.exp(2j * np.pi * pilot * np.arange(frame.size) / fs)
        amplitude = np.vdot(carrier, frame) / frame.size
        data = frame - amplitude * carrier
        psd_checks = []
        for nperseg in [32768, 65536, 131072]:
            f, power = welch(data, fs=fs, window=("kaiser", 12), nperseg=nperseg, noverlap=nperseg // 2, detrend=False, return_onesided=False, scaling="density")
            order = np.argsort(f)
            f, power = f[order], power[order]
            average = integrate(f, power, -width / 2, width / 2) / width
            refs = [integrate(f, power, pilot + sign * 2 * bin_width - bin_width / 2, pilot + sign * 2 * bin_width + bin_width / 2) / bin_width for sign in [-1, 1]]
            psd_checks.append({"nperseg": nperseg, "reference_to_allocation_mean": [v / average for v in refs], "mean_reference_to_allocation_mean": float(np.mean(refs) / average)})
        print(f"Channelizing {label}", flush=True)
        clean = geometry._channelize_one(iq[:required], rf_center_hz=report["detector_geometry"]["rf_center_hz"], channel_index=report["detector_geometry"]["selected_channel_index"])
        rows = clean.reshape(128, 128).astype(np.complex128)
        ideal = geometry._ideal_float_weights_from_layout(layout, detector_window_samples=128)
        bank = geometry.DetectorWeightBank(explicit_path=ROOT / report["weight_bank"]["path"])
        weights, valid = bank.get_weights_for_physical_channel(14)
        assert valid
        wp = unpack_packed_complex(weights, 4)
        scale = estimate_complex_scale(clean[None, :], bits_per_component=4, clip_sigma=3.0)
        packed = geometry.quantize_complex_numpy(rows, 4, scale)
        quantized = unpack_packed_complex(packed, 4)
        clean_stats = {}
        A = width / bin_width * 10 ** (-audit["measured_pilot_below_data_db"] / 10)
        for mode, samples, w in [("float", rows, ideal), ("packed", quantized, wp)]:
            powers = np.abs(samples @ np.conj(w).T) ** 2
            norms = np.sum(np.abs(w) ** 2, axis=1)
            normalized = powers / norms
            total = np.sum(normalized, axis=0)
            q = 2 * total[0] / (total[1] + total[2])
            refs = normalized[:, 1:].sum(axis=1)
            clean_stats[mode] = {"normalized_Q": float(q), "equivalent_g": float(A / (q - 1)), "plateau_db": float(10 * np.log10((q - 1) / A)), "first_16_rows_reference_fraction": float(refs[:16].sum() / refs.sum()), "reference_row0_over_rows16plus_median": float(refs[0] / np.median(refs[16:])), "raw_term_power_sums": powers.sum(axis=0).tolist()}
        trials = list(csv.DictReader(csv_path.open()))
        summaries = list(csv.DictReader(summary_path.open()))
        recount = []
        for summary in summaries:
            snr = float(summary["requested_data_shelf_snr_db"])
            group = [r for r in trials if float(r["requested_data_shelf_snr_db"]) == snr]
            estimates = {}
            for mode, prefix, norm_t, norm_r in [("gpu", "", layout["target_norm_sq"], layout["reference_norm_sum_sq"]), ("cpu_packed", "cpu_packed_", layout["target_norm_sq"], layout["reference_norm_sum_sq"]), ("cpu_float", "cpu_float_", 128, 256)]:
                suffix2 = "_u64" if mode == "gpu" else ""
                cast = int if mode == "gpu" else float
                t = sum(cast(r[prefix + "p_target" + suffix2]) for r in group)
                ref = sum(cast(r[prefix + "p_ref_sum" + suffix2]) for r in group)
                q = t * norm_r / (ref * norm_t)
                value = 10 * math.log10((q - 1) / A)
                recorded = float(summary[mode + "_pooled_estimated_data_shelf_snr_db"])
                assert abs(value - recorded) < 1e-10, (label, snr, mode, value, recorded)
                estimates[mode] = value
            recount.append({"requested_snr_db": snr, "trials": len(group), "pooled_plateau_db": estimates})
        outputs[label] = {"iq_samples": len(iq), "evaluated_samples": required, "coherent_pilot_power": float(abs(amplitude) ** 2), "welch_method": "Remove finite-frame coherent pilot; Kaiser beta12;50% overlap; integrate exact reference-bin edges; allocation mean from residualPSD over6MHz. A finite-window diagnostic, not telescope calibration.", "welch": psd_checks, "clean_quantization_scale": float(scale), "A_from_saved_audit": A, "clean": clean_stats, "saturation_recount": recount, "historical_source_revision": "not recorded in archived report; this audit does not reconstruct it"}
    overlap_a = iq_arrays["as_generated"][200000:]
    overlap_b = iq_arrays["settled"][:len(overlap_a)]
    comparison = {"overlap_samples": len(overlap_a), "startup_offset_samples": 200000, "byte_identical": bool(np.array_equal(overlap_a, overlap_b)), "relative_rms_difference": float(np.linalg.norm(overlap_a - overlap_b) / np.linalg.norm(overlap_a)), "interpretation": "Numerically consistent with separately generated continuation after200000samples; not an exact byte trim of the surviving raw file. No settled-generation sidecar survives."}
    for name, module in sorted(sys.modules.items()):
        f = getattr(module, "__file__", None)
        if f and Path(f).suffix == ".py" and Path(f).is_relative_to(ROOT):
            sources.append(identity(Path(f)))
    out = {"schema_version": "pilotproxy_settled_waveform_reaudit_v1", "waveforms": outputs, "startup_overlap": comparison, "sources": list({x["path"]: x for x in sources}.values()), "scope": "Deterministic waveform and archived arithmetic verification; not independent transmitter-state, received-shelf or visibility calibration."}
    (args.output / "audit.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v["clean"] for k, v in outputs.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
