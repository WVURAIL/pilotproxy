# Paper-to-code map

The journal repositories hold frozen data and build scripts. Current code must
reproduce the same defined quantities, but later source is not the binary used
for an earlier timing result. Record source, library, inputs and environment
for every new run. Do not overwrite the papers' frozen records.

| Paper quantity or claim | Implementation | Verification |
|---|---|---|
| Detector: matched-filter projection `x conj(w)` and frozen FFT | `cuda/pilotproxy_detector.cu`, `cuda/fxfft256_ref.c`, `fxfft.py` | Native reference tests, `tests/core/test_fxfft256.py`, `tests/kernel/test_fxfft256_gpu.py` |
| Fine powers and frame powers `P_r = sum_f S_r[f]` | `fine_reduction.py`, `product_contract.py` | `test_fine_reduction.py`, `test_product_contract.py`; exact GPU comparison |
| Normalization and signed pilot excess | `detector_contract.py`, `product_contract.pilot_excess_from_fine_powers` | `test_product_contract.py`; v5 decoy row powers in threshold tests |
| Fine score, rational ranking, strict threshold and equality | `fine_decision.py`, `detectors/narrowband_marker/scores.py`, CUDA fine-mask epilogue | `test_fine_decision.py`, `test_scores_equal_fine_decision.py`, fused-mask GPU tests |
| Current-era knee, either bend, equal-event weights | `characterization/knee.py` | `tests/characterization/test_knee.py` |
| Retained signed mean, clustered uncertainty, shelf conversion | `characterization/threshold_rule.sweep` | `test_threshold_rule.py`: direct event calculation |
| Conditional BAO strength `(gamma x_hi)^2 N G` | `threshold_rule.strength`, `bias_over_sigma` | Gain scaling, policy-budget tests; RFIsher's own forecast checks |
| Ideal calendar time `1 / kappa`, recent-year guard | `decide`, `decide_guarded` | Calendar/recent-support tests; `tools/check_paper_thresholds.py` |
| Fixed-point runtime word, calibrated profile and receiver binding | `runtime_bundle.py`, `runtime_candidate.py`, `operating_policy.py` | Corresponding tests under `tests/core/` |
| Receiver packing, output synchronization and mask application | kotekan `cudaPilotProxyPacker`, `cudaPilotProxyDetector`, `DtvRfiMask` | kotekan native detector and pipeline/integration tests |

Paths in the middle column are under `src/pilot_proxy/` unless otherwise shown.
The core test filenames are under `tests/core/` except where their directory is
specified. These tests distinguish arithmetic exactness from physical validity.

## Recheck the frozen threshold decisions

With the companion MNRAS manuscript checkout available:

```sh
python tools/check_paper_thresholds.py --paper-root ../dtv-mnras
```

This uses the paper's frozen sweeps and budgets; no raw telescope data are
needed. It checks the historical record with its recorded calendar factor and
the current ideal-year assignments separately. It also compares the shipped
response table and measured amplitude ratio. It does not validate gamma's
transfer to another population, temporal stationarity, or cross-day covariance.

## Companion repositories

- [RFIsher](https://github.com/WVURAIL/RFIsher): residual template, combined
  `D_V` response, tolerance and finite-amplitude expected-likelihood model.
- [RadioFisher](https://github.com/WVURAIL/RadioFisher): clean CHIME forecast,
  derivatives and unmodeled-residual projection.
- [kotekan](https://github.com/kotekan/kotekan): live receiver integration. The
  manifest in `external/pilotproxy/VENDOR.json` pins its exact upstream core.
  Do not edit vendored files or equate a new main-branch build with an archived
  measurement. PR 1677 merged on October 22, 2026; live commissioning remains
  independent of local arithmetic validation.

## Compatibility and evidence

`PilotProxy_*` is the current native API. Historical fields such as `cfar_rank`,
`deployed*`, old coherence fields and cache rule v1 have specific compatibility
meanings in [TERMINOLOGY.md](TERMINOLOGY.md). Do not infer a statistical model
from those historical tokens. The frame-power
schema, Q16 units, one-based versus zero-based ranks and receiver coordinates
are contracts, not cosmetic names to change by search-and-replace.

`docs/evidence/`, runtime freezes, shipped data and dated era files support
reproduction and remain in place. A tool's presence does not make its old
hypothesis or estimator part of the current paper.
