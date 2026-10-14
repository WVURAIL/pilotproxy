# LimeSDR over-the-air estimator transfer — 2026-08-25

This release freezes the direct LimeSDR transmit/receive sweep with the two radio ports separated by 1.45 cm. The commanded data-shelf SNR grid is -42 dB through 0 dB in 3 dB steps. Per-pass tx-zero, signal-only, and noise-only controls calibrate the received input axis and the expected detector output.

The run contains 30 passes, 1,800 mixture captures, and 1,980 total events. Error bars are deterministic 95% pass-cluster bootstrap intervals from 10,000 resamples. No display smoothing is used. The unit-gain fit uses the control-conditioned expected output as its predictor over commanded -27 dB through 0 dB, corresponding to received -28.273 dB through -1.845 dB.

The 6.7 GiB capture set remains outside the repository. `raw/raw_capture_inventory.csv` pins all 1,980 captures by path, byte size, and SHA-256. The release includes the trial table, summary, event ledger, run plan and state, and all 30 session records. `data/plot_points.csv` is sufficient to reproduce the publication figure.

The exact runner and stream-worker sources are preserved by the annotated tag `estimator-transfer-run-source-20260825`. The later archival plotter is preserved by `estimator-transfer-post-run-source-20260825`, and the publication exporter by `estimator-transfer-publication-source-20260826`. Their hashes and references are recorded in `run/source_provenance.json`.

The publication PDF is a vector figure rendered through LaTeX with embedded T1 Latin Modern fonts. Its exact delivered bytes are pinned here. Rebuilds can differ at the byte level because embedded-font subset identifiers vary, even when the vector content and extracted text match.

This validates threshold-free coarse F-statistic SNR estimation through the minimal direct radio path. It does not test a decision threshold, Pfa, Pd, an ROC curve, field propagation, or multipath performance.

Repeat this sweep only after changes to estimator normalization or math, detector geometry or weights, numeric representation, waveform synthesis, or the hardware and radio path. Threshold-policy changes alone do not invalidate this threshold-free estimation evidence.

Freeze completed digital and radio results with:

    PYTHONPATH=src python3 tools/freeze_estimator_transfer.py --digital-results /path/to/digital_results --ota-results /path/to/radio_results
