from __future__ import annotations

import importlib.util
import os
import datetime as dt
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "monitor_local_archive.py"
SPEC = importlib.util.spec_from_file_location("monitor_local_archive", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
monitor_local_archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor_local_archive)


def test_process_check_requires_one_scan(tmp_path: Path) -> None:
    process = tmp_path / "123"
    process.mkdir()
    (process / "cmdline").write_bytes(b"/venv/bin/pilot-proxy\0chime-scan\0")

    healthy = monitor_local_archive.process_check(True, proc_root=tmp_path)
    assert healthy["status"] == "ok"
    assert healthy["processes"][0]["pid"] == 123

    (tmp_path / "456").mkdir()
    (tmp_path / "456" / "cmdline").write_bytes(b"python\0other.py\0")
    assert monitor_local_archive.process_check(True, proc_root=tmp_path)["status"] == "ok"


def test_heartbeat_uses_newest_run_activity(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    staging = tmp_path / "staging"
    products = run_dir / "_per_pilot"
    products.mkdir(parents=True)
    staging.mkdir()
    log = tmp_path / "run.log"
    log.write_text("running\n", encoding="utf-8")
    product = products / "506.npz"
    product.write_bytes(b"checkpoint")
    os.utime(log, (1000, 1000))
    os.utime(product, (1600, 1600))

    healthy = monitor_local_archive.heartbeat_check(
        run_dir, staging, log, 20, now=2200
    )
    assert healthy["status"] == "ok"
    assert healthy["newest_path"] == str(product.resolve())

    stale = monitor_local_archive.heartbeat_check(run_dir, staging, log, 5, now=2200)
    assert stale["status"] == "error"


def test_gpu_check_parses_thresholds() -> None:
    command = (
        sys.executable,
        "-c",
        "print('0, RTX 5000 Ada, 64, 16384, 5000, 11384, 92')",
    )
    healthy = monitor_local_archive.gpu_check(1024, 85, command=command)
    assert healthy["status"] == "ok"
    assert healthy["devices"][0]["memory_free_mib"] == 11384

    hot = monitor_local_archive.gpu_check(12000, 60, command=command)
    assert hot["status"] == "error"


def test_parser_keeps_gpu_safety_margin(tmp_path: Path) -> None:
    arguments = monitor_local_archive.parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--staging-dir", str(tmp_path / "staging"),
            "--log", str(tmp_path / "run.log"),
        ]
    )
    assert arguments.min_gpu_free_mib == 2500.0


def test_log_check_reports_concerning_tail(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("checkpoint saved\nTraceback (most recent call last)\n", encoding="utf-8")
    report = monitor_local_archive.log_check(log, 1024)
    assert report["status"] == "error"
    assert report["matches"] == ["Traceback (most recent call last)"]


def test_log_check_allows_zero_quarantine_summary(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "100 unit(s) total, 10 done, 0 quarantined, 90 to process\n",
        encoding="utf-8",
    )
    assert monitor_local_archive.log_check(log, 1024)["status"] == "ok"

    log.write_text("100 total, 1 quarantined, 99 to process\n", encoding="utf-8")
    assert monitor_local_archive.log_check(log, 1024)["status"] == "error"


def test_certificate_check_requires_mode_and_remaining_time(tmp_path: Path) -> None:
    certificate = tmp_path / "cadcproxy.pem"
    certificate.write_text("fixture\n", encoding="utf-8")
    certificate.chmod(0o600)
    command = (
        sys.executable,
        "-c",
        "print('notAfter=Sep 30 23:44:52 2030 GMT')",
    )
    now = dt.datetime(2030, 9, 25, tzinfo=dt.timezone.utc).timestamp()
    report = monitor_local_archive.certificate_check(
        certificate, 72, command=command, now=now
    )
    assert report["status"] == "ok"
    assert report["remaining_hours"] > 72

    certificate.chmod(0o644)
    assert monitor_local_archive.certificate_check(
        certificate, 72, command=command, now=now
    )["status"] == "error"
