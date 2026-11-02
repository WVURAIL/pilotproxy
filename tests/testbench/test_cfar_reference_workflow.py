"""Bounded CFAR studies retain their samples, comparisons, and output meaning."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from pilot_proxy.testbench import cfar_reference as bench


@pytest.fixture
def small_study(monkeypatch):
    settings = {
        "THRESHOLD_FRAMES": 64, "THRESHOLD_CHUNK": 17,
        "FRAMES_PER_POINT": 16, "CLEAN_BATCHES": 3,
        "SNR_DB": np.arange(-20.0, 30.1, 10.0),
        "CLEAN_SNR_DB": np.arange(-14.0, -3.9, 1.0),
    }
    for name, value in settings.items():
        monkeypatch.setattr(bench, name, value, raising=False)
    record = SimpleNamespace(thresholds=[], noise=[])
    original_thresholds, original_noise = bench.thresholds, bench.Noise

    def thresholds(geo, frames=4_000_000, chunk=200_000, seed=1):
        # Fail before an accidental full study can allocate its default draw.
        assert (frames, chunk) == (64, 17)
        result = original_thresholds(geo, frames=frames, chunk=chunk, seed=seed)
        record.thresholds.append((geo, seed, result))
        return result

    class Noise(original_noise):
        def __init__(self, n, nbins, rng):
            assert n == 16
            super().__init__(n, nbins, rng)
            self.calls = []
            record.noise.append(self)

        def T(self, lam):
            values = super().T(lam)
            self.calls.append((np.asarray(lam).copy(), values.copy()))
            return values

    monkeypatch.setattr(bench, "thresholds", thresholds)
    monkeypatch.setattr(bench, "Noise", Noise)
    return record


def _scores(values):
    """Independent row-wise mean and sorted order-statistic references."""
    out = {name: [] for name in ("CA", "OS63", "OS94", "OS113")}
    for row in values:
        peak = max(row[:5])
        bulk = sorted(row[5:])
        out["CA"].append(peak / (sum(bulk) / len(bulk)))
        for name, rank in (("OS63", 63), ("OS94", 94), ("OS113", 113)):
            out[name].append(peak / bulk[rank - 1])
    return out


def _crossing(probabilities, grid, target):
    for index, probability in enumerate(probabilities):
        if probability >= target:
            if index == 0:
                return float(grid[0])
            lower = probabilities[index - 1]
            return float((grid[index] * (target - lower)
                          + grid[index - 1] * (probability - target))
                         / (probability - lower))
    return None


def _mean_and_se(values):
    average = sum(values) / len(values)
    variance_of_mean = sum((value - average) ** 2 for value in values) / (
        len(values) * (len(values) - 1))
    return {"mean": average, "se": math.sqrt(variance_of_mean)}


def _strict_load(path):
    def reject(token):
        raise AssertionError(f"nonstandard JSON token: {token}")
    return json.loads(path.read_text(), parse_constant=reject)


def test_cfar_cli_runs_the_complete_comparison_with_recorded_sample_counts(tmp_path, small_study):
    out = tmp_path / "study"
    assert bench.main(["--out", str(out)]) == 0
    result = _strict_load(out / "bench.json")
    assert "not an operational eta_Pfa calibration" in (out / "REFERENCE.md").read_text()
    assert result["sampling"] == {
        "threshold_frames": 64, "threshold_chunk": 17,
        "threshold_seeds": {"even": 11, "odd": 12},
        "curve_seeds": {"even": 111, "odd": 112},
        "clean_loss_seed": 99,
    }
    assert [seed for _, seed, _ in small_study.thresholds] == [11, 12]
    assert len(small_study.noise) == 2 + 3
    for index, (parity, anchor, bulk_size) in enumerate((("even", 128, 125), ("odd", 127, 126))):
        row = result["parities"][parity]
        assert (row["anchor"], row["bulk_size"], row["frames_per_point"]) == (anchor, bulk_size, 16)
        assert len(row["cases"]) == 10
        assert row["thresholds"] == small_study.thresholds[index][2]
        if parity == "even":
            assert row["bulk_shoulder_gain_sum"] == 0
        else:
            assert row["bulk_shoulder_gain_sum"] > row["bulk_gain_pm3"] > 0
        calls = small_study.noise[index].calls
        expected_first_draw = np.random.default_rng(111 + index).standard_normal((16, 5 + bulk_size))
        np.testing.assert_array_equal(small_study.noise[index].N, expected_first_draw)
        stride = len(bench.SNR_DB) + 1
        assert len(calls) == len(row["cases"]) * stride
        for case_index, (name, case) in enumerate(row["cases"].items()):
            draws = calls[case_index * stride:(case_index + 1) * stride]
            threshold = {detector: item["eta"] for detector, item in row["thresholds"].items()}
            probabilities = {detector: [] for detector in threshold}
            for point, (_, values) in enumerate(draws[:-1]):
                scores = _scores(values)
                for detector, eta in threshold.items():
                    probabilities[detector].append(sum(value > eta for value in scores[detector]) / 16)
                ratios = sorted(ca / os for ca, os in zip(scores["CA"], scores["OS63"]))
                assert case["median_zca_over_zos63"][point] == pytest.approx((ratios[7] + ratios[8]) / 2)
            null_scores = _scores(draws[-1][1])
            for detector, eta in threshold.items():
                assert case["pd"][detector] == probabilities[detector]
                assert case["pfa_actual"][detector] == sum(value > eta for value in null_scores[detector]) / 16
                for target, field in ((0.5, "snr_pd50"), (0.9, "snr_pd90")):
                    expected = _crossing(probabilities[detector], bench.SNR_DB, target)
                    if expected is None:
                        assert case[field][detector] is None
                    else:
                        assert case[field][detector] == pytest.approx(expected)
            # A single noise object supplies every point and contamination
            # case; added bulk power cannot change the five pilot-bin draws.
            for point in range(stride):
                np.testing.assert_array_equal(draws[point][1][:, :5], calls[point][1][:, :5])
            lam = draws[-1][0]
            assert np.count_nonzero(lam[5:]) == case["k"]
            if case["k"]:
                assert np.unique(lam[5:][lam[5:] != 0]).item() == 4096 * 10 ** (case["level_db"] / 10)
            assert name == ("clean" if not case["k"] else f"k{case['k']}_{int(case['level_db'])}dB")

    clean = result["clean_loss_125"]
    assert (clean["frames"], clean["batches"], clean["frames_per_batch"]) == (48, 3, 16)
    assert clean["grid_db"] == [-14, -4, 1]
    batches = []
    thresholds = result["parities"]["even"]["thresholds"]
    clean_draws = np.random.default_rng(99)
    for noise in small_study.noise[2:]:
        np.testing.assert_array_equal(noise.N, clean_draws.standard_normal((16, 130)))
        clean_draws.chisquare(4095, (16, 130))
        clean_draws.chisquare(8192, (16, 130))
        probabilities = {detector: [] for detector in thresholds}
        assert len(noise.calls) == len(bench.CLEAN_SNR_DB)
        for _, values in noise.calls:
            scores = _scores(values)
            for detector in thresholds:
                eta = thresholds[detector]["eta"]
                probabilities[detector].append(sum(value > eta for value in scores[detector]) / 16)
        batches.append({detector: {field: _crossing(probability, bench.CLEAN_SNR_DB, target)
                                  for field, target in (("pd50", 0.5), ("pd90", 0.9))}
                        for detector, probability in probabilities.items()})
    for field in ("pd50", "pd90"):
        baseline = [batch["CA"][field] for batch in batches]
        assert clean[f"CA_{field}_snr_db"] == pytest.approx(_mean_and_se(baseline))
        for detector in ("OS63", "OS94", "OS113"):
            differences = [batch[detector][field] - batch["CA"][field] for batch in batches]
            assert clean[f"{detector}_minus_CA_{field}_db"] == pytest.approx(_mean_and_se(differences))


def test_unreached_crossings_are_null_in_returned_and_written_results(tmp_path, small_study, monkeypatch):
    monkeypatch.setattr(bench, "SNR_DB", np.asarray([-100.0, -99.0]))
    monkeypatch.setattr(bench, "CLEAN_SNR_DB", np.asarray([-100.0, -99.0]))
    out = tmp_path / "unreached"
    result = bench.run(out)
    assert _strict_load(out / "bench.json") == result
    for row in result["parities"].values():
        for case in row["cases"].values():
            assert all(value is None for value in case["snr_pd90"].values())
    for name, value in result["clean_loss_125"].items():
        if isinstance(value, dict):
            assert value == {"mean": None, "se": None}


@pytest.mark.parametrize("shape", [(3, 129), (3, 131), (130,)])
def test_scores_refuse_an_array_that_disagrees_with_the_declared_geometry(shape):
    with pytest.raises(ValueError, match="shape"):
        bench.scores(np.ones(shape), 5, 125)
