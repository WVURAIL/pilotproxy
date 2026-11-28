# Test coverage

A release requires the CPU release checks and the GPU checks:

```sh
make release-check PYTHON=python3
make test-gpu PYTHON=python3 NVCC=/path/to/nvcc DETECTOR_WINDOW_SAMPLES=128
```

The GPU gate fails when a GPU test skips because of a missing device, library,
dependency, or compiler. By default, the library is built for K=128, and the
matrix also builds K=64 in temporary directories. These builds do not replace a
pinned campaign library, and no library is installed in the user cache.

GNU Radio uses its own Python environment on some hosts:

```sh
PILOT_PROXY_GNURADIO_PYTHON=/usr/bin/python3 \
  python3 -m pytest tests/testbench/test_generate_atsc_signal.py -q
```

In the transmitter test, the emitted pilot frequency is measured and checked
against the requested RF offset, independently of the generated metadata. When a
GNU Radio
interpreter is configured explicitly, an import or execution error in it fails
the tests.

## Component map

Paths below are relative to `tests/`. Related tests in each directory also
exercise CLI routing, invalid inputs, and schema compatibility.

| Component | Principal testbenches | Acceptance checks |
| --- | --- | --- |
| CUDA C ABI and CPU reference | `../cuda/tests/`, `core/test_kernel_wrapper.py` | Public C header, exact powers and projections, batches, threshold equality, overflow reporting, error propagation |
| CUDA implementations and Python binding | `kernel/test_kernel_matrix_gpu.py` | K=64/128; scalar and DP4A; shared/global/constant weights; 32/64/128 threads; restricted grids; independent integer row products and powers; fixed FFT; masks; reused buffers and live handles |
| Fixed-point fine FFT | `core/test_fxfft*.py`, `kernel/test_fxfft256_gpu.py`, `kernel/test_fine_powers_gpu.py` | Frozen tables, C/Python/device parity, rounding, bounds, batch and non-block-multiple stream counts |
| Fused fine accumulation and decisions | `kernel/test_fused_fine_gpu.py`, `kernel/test_fused_mask_gpu.py`, `kernel/test_fused_mask_median_reference_gpu.py`, `core/test_fine_decision.py`, `core/test_fine_decision_median_reference.py` | Composed/fused equality, optional taps, delayed writers, repeated launches, rational ties, zero denominators, wrapped designation, full uint64 multipliers |
| Fine decision v3 statistic and keep (`fine_stat_v3`, `fine_keep_v3`, `masks_from_stat`) | `core/test_fine_decision_even_median.py`, `core/test_run_stat.py`, `kernel/test_fine_stat_gpu.py`, `../cuda/tests/test_pilotproxy_reference.cpp` | Even-bin median reference and all-bin peak against an oracle, outliers in fewer than half the even bins, no decision at `d_rho = 0`, keep-word change points, agreement with v2 over the even bins, recorded rows decided exactly at the keep boundary, unevaluated and missing rows, the device statistic equal to the reference with the same decision at every keep boundary, the C keep `pilotproxy_v3_keep` against 128-bit products |
| Wide device arithmetic | `kernel/test_kernel_matrix_gpu.py` | 10000 seeded boundary/random cases per build against Python arbitrary-precision 128/192-bit products |
| Weights, receiver profiles, and runtime bundles | `core/test_detector_weights.py`, `core/test_runtime_bundle.py`, `core/test_profile_contract_strict.py`, `core/test_chord_profiles.py` | Checksums, exact integer fields, offsets, channel bindings, calibration provenance, coordinate systems, CHIME/CHORD/Pathfinder geometry |
| Bundle calibration, test and qualification (`calibrate-bundle`, `test-bundle`, `qualify-bundle`) | `core/test_bundle_calibration.py` | Measured and nominal anchors, knee and floor words (v1, v2), the v3 default word scaled from the control, first bundles with no captures, fast words equal to the exact reference, tests on other captures, reused captures and tampered bundles, the masked-fraction and control checks at bias/std at most 1 over hour bins of frame time (capture bins with no frame time; one calibration hour and old reports fail), qualification records and refusals |
| v3 default words of the rule `control_quantile` against the dissertation | `core/test_default_word_pin.py` | The control's night and all-frame 99th percentiles and the 24 default words of dissertation 03ca6bb, from the control's keep words of the v3 score cache |
| False-discovery-rate default word (`false_discovery.py`) | `core/test_false_discovery.py`, `core/test_bundle_calibration.py` | GPD fit, a clean channel masking nothing, more DTV moving the word down, the GPD tail beyond the empirical range, too few null frames refused, the control not masked, the rule and level plan keys and errors |
| FDR default words against the dissertation | `core/test_default_word_fdr_archive.py` | The 23 words of `default_words.json` and the control's tail fit from the archive score cache, and `lambda_c` from the bundled weights; checks the rule of `false_discovery.py`, not `calibrate-bundle` end to end; skips where the cache or the dissertation is absent |
| Power-limit word (`power_limit_word.py`) | `core/test_bundle_calibration.py` | Ratio excess equal to the archive pilot excess, control corrections equal to the direct sums, the largest passing word, coupling, the pilot passband factor, decided control frames only, the v3 power-limit word beside the default word, channels with no passing word, plan errors |
| Night option of the default word (`sun.py`) | `core/test_bundle_calibration.py` | Solar elevation at DRAO, the control quantile over night frames, frame times of run windows and refused formats, the night control in the bundle test |
| Run files and processing (`run_files.py`, `plan-run`, `process-run`) | `core/test_run_files.py`, `core/test_run_stat.py`, `core/test_bundle_calibration.py` | Combined windows and several-frame files read as their sources, F-engine restarts, idle halves, open and partly written files, the v3 words `DtvRfiMask` applies, the time split of each channel, run summaries and spectrograms with the knee and power-limit thresholds and under v3 |
| Physical RF placement | `core/test_chord_tone_injection.py`, `core/test_atsc_channels.py`, `core/test_predicted_designation.py` | Independently synthesized pilot/reference tones across channels 14–36, Nyquist zone, edge and DC placement |
| Packing, stream layouts, and quantization | `core/test_frame_geometry_adapter.py`, `core/test_packing_contract.py`, `core/test_quantization_roundtrip.py`, `core/test_integration_contract.py` | Feed/channel/window order, overlapping batches, diagnostic layouts, per-stream scales, signed nibble limits, invalid numeric inputs |
| PFB and receiver channelizer | `core/test_reference_pfb.py`, `core/test_reference_pfb_gpu.py` | Response, channel ordering, time and phase conventions, CPU/GPU agreement |
| Coarse statistic, fine reductions, masks, and units | `core/test_mask_zero_point.py`, `core/test_dtv_units.py`, `core/test_fine_reduction.py`, `test_mask_boundary.py` | Norm correction, strict thresholds, zero references, exact retention, CFAR, dimensional conversions |
| ATSC generation | `testbench/test_generate_atsc_signal.py`, `testbench/test_audit_atsc_signal.py` | Waveform quality checks; real GNU Radio requires its interpreter |
| CHIME readers and adapters | `chime/test_chime_frame_adapter.py` | Native offset-binary bytes, partial frames, metadata, bounded reading, malformed files |
| Standalone detector and CHIME analysis | `core/test_detect_inputs.py`, `chime/test_chime_reductions.py` | Input contracts and frame results |
| Threshold pipeline and deployment | `characterization/test_release.py`, `characterization/test_*histogram_frames_export.py`, `characterization/test_score_cache.py`, `characterization/test_threshold_rule.py`, `core/test_operating_policy.py` | Authenticated frame exports, exact weights and masks, conditional power scaling, required finite-bias limits, maximum retention under both power constraints, and runtime word binding |
| Product health, provenance, schemas, and atomic I/O | `core/test_archive_health.py`, `core/test_product_contract.py`, `core/test_provenance.py`, `core/test_atomic_io.py`, `core/test_json_utils.py`, `chime/test_validate_products.py` | Health-gate exclusions, exact types, provenance, interrupted writes and validation failures |
| CLI, environment, packaging, and GPU test gate | `test_cli*.py`, `core/test_secondary_python.py`, `core/test_gpu_test_gate.py`, `../.github/workflows/tests.yml` | Argument routing, interrupt-handler restoration, dependency failures, isolated installs/resources and refusal of skipped hardware gates |

The kernel matrix has 128 padded stream slots, with 16 or 32 active inputs and
the remaining slots filled with zeros. This layout matches the padded Pathfinder
buffer of 64 dishes in two polarizations. The tests do not assert which inputs
are active on the instrument.

## Limits and remaining work

A passing test suite confirms only the contracts it tests. Line coverage alone
cannot confirm physical sensitivity, calibration quality, or concurrency safety.
The hardware gate must also run
on the deployment GPU. Compute Sanitizer and a sustained shadow run of the full
pipeline remain as further hardware checks.

Tests that require external artifacts skip when their inputs are absent, and a
skipped test validates nothing about those artifacts. The archive survey, scan
and combine suites were removed with the archive pipeline (tag
`pre-purge-2026-11-23`).
