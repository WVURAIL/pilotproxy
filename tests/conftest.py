"""Optional hardware gates for release validation."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--require-cuda",
        action="store_true",
        help="Fail GPU tests that would otherwise skip missing hardware or tools.",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    path = str(item.path).replace("\\", "/")
    hardware_test = (
        "/kernel/" in path
        or path.endswith("/test_reference_pfb_gpu.py")
        or item.get_closest_marker("cuda") is not None
    )
    if item.config.getoption("--require-cuda") and hardware_test and report.skipped:
        report.outcome = "failed"
        report.longrepr = f"Required CUDA test skipped: {report.longrepr}"


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    # Collection-level dependency skips must not make a hardware gate green.
    result = yield
    report = result.get_result()
    path = str(collector.path).replace("\\", "/")
    hardware_test = "/kernel/" in path or path.endswith("/test_reference_pfb_gpu.py")
    if (
        collector.config.getoption("--require-cuda")
        and hardware_test
        and report.skipped
    ):
        report.outcome = "failed"
        report.longrepr = f"Required CUDA module skipped: {report.longrepr}"
