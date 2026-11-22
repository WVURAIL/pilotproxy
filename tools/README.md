# Tools

Run each command with --help. Store new outputs separately from existing data.

- threshold_rule.py: maximum-retention threshold from a schema-v3 response
  table, an explicit model power normalization, and per-channel finite-bias
  limits supplied through --power-limits. It returns conditional in-sample choices.
- Detector validation: prepare_fine_validation_waveforms.py,
  run_fine_validation_campaign.py, reduce_fine_validation.py.
- Estimator and SDR checks: validate_fine_detector.py, measure_fine_gain.py,
  generate_settled_atsc.py, audit_settled_waveform.py.
- Runtime checks: validate_sdr_upgrade_adapter.py, validate_sdr_upgrade_streaming.py.
- Archive health: audit_per_pilot.py, audit_local_archive_run.py.

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
  With a fine_decision_v3 bundle (--bundle-dir), check_pathfinder_dtv_run.py, compare_pathfinder_dtv_mask.py and
  analyze_dtv_run.py decide the v3 channels from their dtv_stat rows at the word N2 applies.
  The checklist is docs/PATHFINDER_RECALIBRATION.md.
- Detector validation, further: prepare_fine_validation_cache.py (phase-preserving fixtures through the reference
  PFB), supplement_fine_campaign_plan.py (a sealed campaign plan for a supplementary channel),
  report_fine_null_validation.py (independent H0 outcomes), report_fine_sensitivity.py (frozen H1 cohorts),
  current_geometry_sensitivity.py (the staged synthetic sensitivity experiment),
  audit_fine_native_coordinates.py (inverted native coordinates against a frozen plan),
  fused_real_receipt.py (the fused fine-mask entry point against the staged path on CHIME frames),
  framing_audit.py (baseband framing of the weight ROM).
- Fixed-point FFT: emit_fxfft_tables.py (twiddles as C literals), generate_fxfft_goldens.py (golden vectors),
  fxfft_report.py (quantization error against float64).
- SDR and GNU Radio: generate_gnuradio_reference.py, run_sdr_distribution_reference.py,
  plot_gnuradio_sdr_references.py, analyze_noise_signal_references.py, lime_reference_capture.py,
  limesdr_smoke_capture.py, prepare_sdr_protocol.py, run_sdr_antenna_controls.py, analyze_sdr_antenna_controls.py,
  run_sdr_live_handoff.py, sdr_transfer.py.
- Repository and examples: generate_doc_specs.py (TeX macros for the formal documents), make_code_only_cut.py (the
  tree without the CHIME-derived files), make_capture_replay_example.py and make_operating_policy_example.py
  (synthetic fixtures for rehearsals), freeze_estimator_transfer.py (freeze estimator-transfer evidence).
