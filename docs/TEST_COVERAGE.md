# Test coverage

Run the CPU release checks and the required GPU checks before a release:

```sh
make release-check PYTHON=python3
make test-gpu PYTHON=python3 NVCC=/path/to/nvcc DETECTOR_WINDOW_SAMPLES=128
```

The GPU gate fails if a GPU test skips a missing device, library, dependency,
or compiler. The default library is built for K=128; the matrix also builds
K=64 in temporary directories. These builds do not replace a separately pinned
campaign library or install a library into the user cache.

GNU Radio uses its own Python environment on some hosts:

```sh
PILOT_PROXY_GNURADIO_PYTHON=/usr/bin/python3 \
  python3 -m pytest tests/testbench/test_add_awgn.py \
  tests/testbench/test_generate_atsc_signal.py -q
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3 -m pytest \
  tests/testbench/test_evaluate_dtv_snr.py::test_gnuradio_awgn_helper_hits_requested_band_snr -q
```

The noise tests check sample counts on both sides of the vector-alignment
boundary, seed reproducibility, complex noise power, and bandwidth scaling.
The transmitter test checks the emitted pilot frequency against the requested
RF offset independently of the generated metadata.
An explicitly configured GNU Radio interpreter must work; import or execution
errors fail the tests.

## Component map

Paths below are relative to `tests/`. Related tests in each directory also
exercise CLI routing, invalid inputs, and schema compatibility.

| Component | Principal testbenches | Acceptance checks |
| --- | --- | --- |
| CUDA C ABI and CPU reference | `../cuda/tests/`, `core/test_kernel_wrapper.py` | Public C header, exact powers and projections, batches, threshold equality, overflow reporting, error propagation |
| CUDA implementations and Python binding | `kernel/test_kernel_matrix_gpu.py` | K=64/128; scalar and DP4A; shared/global/constant weights; 32/64/128 threads; restricted grids; independent integer row products and powers; fixed FFT; masks; reused buffers and live handles |
| Fixed-point fine FFT | `core/test_fxfft*.py`, `kernel/test_fxfft256_gpu.py`, `kernel/test_fine_powers_gpu.py` | Frozen tables, C/Python/device parity, rounding, bounds, batch and non-block-multiple stream counts |
| Fused fine accumulation and decisions | `kernel/test_fused_fine_gpu.py`, `kernel/test_fused_mask_gpu.py`, `core/test_fine_decision.py` | Composed/fused equality, optional taps, delayed writers, repeated launches, rational ties, zero denominators, wrapped designation, full uint64 multipliers |
| Wide device arithmetic | `kernel/test_kernel_matrix_gpu.py` | 10,000 seeded boundary/random cases per build against Python arbitrary-precision 128/192-bit products |
| Weights, receiver profiles, and runtime bundles | `core/test_detector_weights.py`, `core/test_runtime_bundle.py`, `core/test_profile_contract_strict.py`, `core/test_chord_profiles.py` | Checksums, exact integer fields, offsets, channel bindings, calibration provenance, coordinate systems, CHIME/CHORD/Pathfinder geometry |
| Physical RF placement | `core/test_chord_tone_injection.py`, `core/test_atsc_channels.py`, `core/test_predicted_designation.py` | Independently synthesized pilot/reference tones across channels 14–36, Nyquist zone, edge and DC placement |
| Packing, stream layouts, and quantization | `core/test_frame_geometry_adapter.py`, `core/test_packing_contract.py`, `core/test_quantization_roundtrip.py`, `core/test_integration_contract.py` | Feed/channel/window order, overlapping batches, diagnostic layouts, per-stream scales, signed nibble limits, invalid numeric inputs |
| PFB and receiver channelizer | `core/test_reference_pfb.py`, `core/test_reference_pfb_gpu.py` | Response, channel ordering, time and phase conventions, CPU/GPU agreement |
| Coarse statistic, fine reductions, masks, and units | `core/test_mask_zero_point.py`, `core/test_dtv_units.py`, `core/test_fine_reduction.py`, `test_mask_boundary.py` | Norm correction, strict thresholds, zero references, exact retention, CFAR, dimensional conversions |
| ATSC generation, noise, and sensitivity | `testbench/test_generate_atsc_signal.py`, `testbench/test_add_awgn.py`, `testbench/test_audit_atsc_signal.py`, `testbench/test_evaluate_dtv_snr.py`, `testbench/test_sensitivity_study.py`, `testbench/test_quantize_paths.py` | Waveform quality checks, noise calibration, injection response, ablations, path handling; real GNU Radio requires its interpreter |
| Frontier, census, and plotting | `testbench/test_mask_frontier.py`, `testbench/test_transmitter_census.py`, `testbench/test_plot_results.py`, `testbench/test_summarize_results.py` | Threshold transitions, identity, selection, aggregation and plot contracts |
| CHIME readers and adapters | `chime/test_chime_frame_adapter.py`, `chime/test_segmented_input.py`, `archive/test_packed_reader.py`, `archive/test_reader_contract_guards.py` | Native offset-binary bytes, partial frames, metadata, segmentation, bounded reading, malformed files |
| Standalone detector and CHIME analysis | `core/test_detect_inputs.py`, `chime/test_chime_runner_small.py`, `chime/test_chime_reductions.py`, `chime/test_injection_recovery.py` | Input contracts, frame results, reference parity, and injection recovery |
| Archive detector and real GPU scan | `archive/test_detector_analyzer_parity.py`, `archive/test_detector_resume.py`, `kernel/test_archive_scan_gpu.py` | HDF5 → packing → CUDA → retained products, independent CPU numerical check, actual SIGINT mid-file, checkpoint/resume equivalence and completed-run no-op |
| Archive pipeline and publication | `archive/test_pipeline_regressions.py`, `archive/test_engine_safety.py`, `archive/test_publish_transaction.py`, `archive/test_quarantine_locking.py` | Worker failures, bounded staging, cleanup, ordered consumption, atomic publication, locks and quarantines |
| Source transport, inventory, survey, and selection | `archive/test_cadc_offline.py`, `archive/test_datatrail_adapter.py`, `archive/test_transport_errors.py`, `archive/test_selection*.py`, `archive/test_survey*.py` | Replica contracts, retry classification, exact selections, persistent state, concurrent workers, real process interruption and crash windows |
| Combining | `archive/test_combine*.py`, `archive/test_event_keyed_combine.py` | Event identity, duplicate refusal and frequency/time alignment |
| Threshold pipeline and deployment | `characterization/test_release.py`, `characterization/test_*histogram_frames_export.py`, `characterization/test_power_era_review.py`, `characterization/test_knee.py`, `characterization/test_score_cache.py`, `characterization/test_threshold_rule.py`, `core/test_operating_policy.py` | Release into the frame export, working cohorts, the knee (and, with the recorded inputs, all 24 recorded knees), exact score-cache weights, the rule, and the frozen decisions compiled to the permanent mask and the deployed words on `chime_freq_id` |
| Product health, provenance, schemas, and atomic I/O | `core/test_archive_health.py`, `core/test_product_contract.py`, `core/test_provenance.py`, `core/test_atomic_io.py`, `core/test_json_utils.py`, `chime/test_validate_products.py` | Health exclusions, support evidence, exact types, provenance, interrupted writes and validation failures |
| CLI, environment, packaging, and GPU test gate | `test_cli*.py`, `test_setup_env_guard.py`, `archive/test_accel.py`, `archive/test_standalone_runtime.py`, `core/test_secondary_python.py`, `core/test_gpu_test_gate.py`, `../.github/workflows/tests.yml` | Argument routing, interrupt-handler restoration, dependency failures, isolated installs/resources and refusal of skipped hardware gates |

The kernel matrix includes 128 padded stream slots with 16 or 32 active inputs
and zero-filled remaining slots. These correspond to the padded 64-dish,
two-polarization Pathfinder buffer. They do not assert which inputs are active
on the instrument.

## Limits and remaining work

Passing these tests establishes the tested contracts, not complete correctness.
Line coverage alone cannot establish physical sensitivity, calibration quality,
or concurrency safety. The hardware gate must also run on the deployment GPU.
Compute Sanitizer and a sustained full-pipeline shadow run remain separate
hardware checks.

The archive survey suite exercises inventory compatibility, provenance,
collection restoration, path validation, interrupted writes, and recovery of
exported views. These are executable assertions, not expected-failure
placeholders. Tests that require external artifacts still skip when their
inputs are absent; a skip is not validation of those artifacts.

Large-product tests require `PILOT_PROXY_ATTACHED_PRODUCTS` or `PP_PER_PILOT`.
The frozen-bundle rederivation additionally accepts
`PILOT_PROXY_DEEP_BUNDLE_CHECK=1`; it reads the preserved source product archive.
These checks validate existing artifacts and do not start a new campaign.

The testbench rejects invalid packing/noise requests. These checks do not
change CUDA arithmetic, FFT tables, shipped weights, or valid-input
quantization. The CHIME packed archive path preserves its native integer grid
and does not use the synthetic floating-point quantizer or GNU Radio noise
generator.
