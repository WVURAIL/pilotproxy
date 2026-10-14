# coding=utf-8
from __future__ import annotations

import json

import pytest

from pilot_proxy.testbench.summarize_results import (
    FSTAT_COLUMN,
    FSTAT_HISTOGRAM_NAME,
    HISTOGRAM_MODE_AUTO,
    HISTOGRAM_MODE_ALWAYS,
    SNR_SHELF_COLUMN,
    SNR_SHELF_HISTOGRAM_NAME,
    REQUESTED_SNR_COLUMN,
    extract_result_rows,
    summarize_result_json,
    summarize_rows,
)

FSTAT_RAW_0 = 1.25
FSTAT_RAW_1 = 1.50
SNR_SHELF_0 = -30.0
SNR_SHELF_1 = -20.0
REQUESTED_SNR_0 = -30.0
REQUESTED_SNR_1 = -20.0
HISTOGRAM_BINS = 5


def test_extract_result_rows_accepts_validation_payload() -> None:
    payload = {
        "schema_version": "pilotproxy_snr_validation_report_v1",
        "results": [
            {FSTAT_COLUMN: FSTAT_RAW_0, SNR_SHELF_COLUMN: SNR_SHELF_0},
            {FSTAT_COLUMN: FSTAT_RAW_1, SNR_SHELF_COLUMN: SNR_SHELF_1},
        ],
    }

    rows = extract_result_rows(payload)

    assert len(rows) == 2
    assert rows[0][FSTAT_COLUMN] == FSTAT_RAW_0


def test_summarize_rows_reports_core_columns() -> None:
    rows = [
        {FSTAT_COLUMN: FSTAT_RAW_0, SNR_SHELF_COLUMN: SNR_SHELF_0},
        {FSTAT_COLUMN: FSTAT_RAW_1, SNR_SHELF_COLUMN: SNR_SHELF_1},
    ]

    summary = summarize_rows(rows)
    by_column = {row["column"]: row for row in summary}

    assert by_column[FSTAT_COLUMN]["finite_count"] == 2
    assert by_column[SNR_SHELF_COLUMN]["min"] == SNR_SHELF_0
    assert by_column[SNR_SHELF_COLUMN]["max"] == SNR_SHELF_1


def test_summarize_result_json_skips_histograms_for_sweep_by_default(tmp_path) -> None:
    input_json = tmp_path / "sweep.json"
    output_dir = tmp_path / "summary"
    input_json.write_text(
        json.dumps(
            {
                "results": [
                    {
                        REQUESTED_SNR_COLUMN: REQUESTED_SNR_0,
                        FSTAT_COLUMN: FSTAT_RAW_0,
                        SNR_SHELF_COLUMN: SNR_SHELF_0,
                    },
                    {
                        REQUESTED_SNR_COLUMN: REQUESTED_SNR_1,
                        FSTAT_COLUMN: FSTAT_RAW_1,
                        SNR_SHELF_COLUMN: SNR_SHELF_1,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    summary = summarize_result_json(
        input_json=input_json,
        output_dir=output_dir,
        bins=HISTOGRAM_BINS,
        histograms=HISTOGRAM_MODE_AUTO,
    )

    assert summary["fstat_histogram_png"] is None
    assert summary["snr_shelf_histogram_png"] is None
    assert "multiple requested_data_shelf_snr_db" in summary["histograms_skipped_reason"]
    assert not (output_dir / FSTAT_HISTOGRAM_NAME).exists()
    assert not (output_dir / SNR_SHELF_HISTOGRAM_NAME).exists()


def test_summarize_result_json_can_force_histograms_for_sweep(tmp_path) -> None:
    # Forcing histograms needs matplotlib (the `plot`/`chime` extra); skip rather
    # than hard-fail (SystemExit) on a minimal `.[test]` install without it,
    # matching the gnuradio importorskip in test_evaluate_dtv_snr.py.
    pytest.importorskip("matplotlib")

    input_json = tmp_path / "sweep.json"
    output_dir = tmp_path / "summary"
    input_json.write_text(
        json.dumps(
            {
                "results": [
                    {
                        REQUESTED_SNR_COLUMN: REQUESTED_SNR_0,
                        FSTAT_COLUMN: FSTAT_RAW_0,
                        SNR_SHELF_COLUMN: SNR_SHELF_0,
                    },
                    {
                        REQUESTED_SNR_COLUMN: REQUESTED_SNR_1,
                        FSTAT_COLUMN: FSTAT_RAW_1,
                        SNR_SHELF_COLUMN: SNR_SHELF_1,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    summary = summarize_result_json(
        input_json=input_json,
        output_dir=output_dir,
        bins=HISTOGRAM_BINS,
        histograms=HISTOGRAM_MODE_ALWAYS,
    )

    assert summary["fstat_histogram_png"] is not None
    assert summary["snr_shelf_histogram_png"] is not None
    assert (output_dir / FSTAT_HISTOGRAM_NAME).exists()
    assert (output_dir / SNR_SHELF_HISTOGRAM_NAME).exists()
