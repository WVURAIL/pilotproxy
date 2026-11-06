# Tools

- threshold_rule.py: maximum-retention threshold from a schema-v3 response
  table, an explicit model power normalization, and per-channel finite-bias
  limits supplied through --power-limits. It returns conditional in-sample choices.
- Detector validation: prepare_fine_validation_waveforms.py,
  run_fine_validation_campaign.py, reduce_fine_validation.py.
- Estimator and SDR checks: validate_fine_detector.py, measure_fine_gain.py,
  generate_settled_atsc.py, audit_settled_waveform.py.
- Runtime checks: validate_sdr_upgrade_adapter.py, validate_sdr_upgrade_streaming.py.
- Archive health: audit_per_pilot.py, audit_local_archive_run.py.

Run each command with --help. Store new outputs separately from existing data.
- Bundle workflow: extract_baseband_channels.py, analyze_dtv_run.py.
