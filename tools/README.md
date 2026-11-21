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
- Pathfinder runs: pathfinder_dtv_switch.py (mask and recording switches over REST),
  check_pathfinder_dtv_run.py (per-channel checks and spectrograms from the recv files),
  combine_pathfinder_dtv_run.py (an acquisition's per-window recv files as one file per half and product),
  compare_pathfinder_dtv_mask.py (visibilities with the mask on against off, in a run that switches it),
  pathfinder_preflight.py (read-only check of the nodes before a calibration-mode run),
  pathfinder_mask_schedule.sh (the mask switched in blocks through choco),
  calibration_length.py (how long a calibration must be, from one long recording),
  pathfinder_input_counts.py (good inputs the detector summed, from the visibility files' bad feed mask),
  make_pathfinder_mock_run.py (synthetic recv files for rehearsals).
  The checklist is docs/PATHFINDER_RECALIBRATION.md.
