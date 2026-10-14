# Digital estimator transfer — 2026-08-25

This release freezes the threshold-free synthetic GNU Radio sweep from -60 dB to +60 dB in 3 dB steps. It compares the CPU float, packed CPU, and GPU fixed-point implementations with the ideal local-reference benchmark and the waveform-conditioned expected transfer.

The 41 points contain 9,000 noise trials. Points from -60 dB through -48 dB have 240 trials each, points from -45 dB through -30 dB have 1,000 trials each, and points from -27 dB through +60 dB have 60 trials each. Error bars are deterministic 95% trial-bootstrap intervals from 10,000 resamples. No display smoothing is used.

The raw 40 result shards remain outside the repository. `raw/raw_input_inventory.csv` pins the 120 summary, trial, and metadata files used here. `data/plot_points.csv` is sufficient to reproduce the publication figure without the raw shards. `run/conditioning.json` preserves the conditioning coefficients and their original source hashes.

The exact plotter source that wrote the conditioning record is preserved by the annotated tag `estimator-transfer-source-20260825`. The later archival plotter is preserved by `estimator-transfer-post-run-source-20260825`, the sweep launcher by `estimator-transfer-run-source-20260825`, and the publication exporter by `estimator-transfer-publication-source-20260826`. Their hashes and references are recorded in `run/source_provenance.json`.

The publication PDF is a vector figure rendered through LaTeX with embedded T1 Latin Modern fonts. Its exact delivered bytes are pinned here. Rebuilds can differ at the byte level because embedded-font subset identifiers vary, even when the vector content and extracted text match.

This validates coarse F-statistic normalization and SNR estimation. It does not test a decision threshold, Pfa, Pd, an ROC curve, or the full 2048-stream deployment geometry.

Repeat this sweep only after changes to estimator normalization or math, detector geometry or weights, numeric representation, or waveform and noise synthesis. Threshold-policy changes alone do not invalidate this threshold-free estimation evidence.

Rebuild the raw sweep with:

    scripts/run_estimator_transfer.sh /path/to/new_results

Freeze completed digital and radio results with:

    PYTHONPATH=src python3 tools/freeze_estimator_transfer.py --digital-results /path/to/digital_results --ota-results /path/to/radio_results
