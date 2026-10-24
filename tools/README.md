# Tools catalog

Run these commands from the repository root after installing the package.
The supported package and CLI live in `src/pilot_proxy/`; this directory holds
explicit analysis and maintenance commands, not modules to import in production.
Use each command's `--help` for required inputs. Write new outputs outside
frozen evidence directories.

| Task | Entry points |
|---|---|
| Assign current conditional thresholds | `threshold_rule.py --policy-budget ...` |
| Compare current code with the paper's frozen decisions | `check_paper_thresholds.py --paper-root ...` |
| Inspect per-pilot products or archive completion | `audit_per_pilot.py`, `audit_local_archive_run.py` |
| Run fine-detector validation | `prepare_fine_validation_waveforms.py`, `prepare_fine_validation_cache.py`, `run_fine_validation_campaign.py`, `reduce_fine_validation.py` |
| Exact transform and GPU checks | `emit_fxfft_tables.py`, `generate_fxfft_goldens.py`, `fused_real_receipt.py`, `audit_fine_native_coordinates.py` |
| Receiver/SDR integration | `prepare_sdr_protocol.py`, `validate_sdr_upgrade_adapter.py`, `validate_sdr_upgrade_streaming.py`; [streaming guide](../docs/SDR_UPGRADE_STREAMING.md) |
| Bench signal generation and audit | `generate_settled_atsc.py`, `audit_settled_waveform.py`, `generate_gnuradio_reference.py` |
| Documentation and distributable maintenance | `generate_doc_specs.py`, `make_code_only_cut.py` |
| Stack-to-proxy amplitude ratio of the frozen records | `archive/derive_stack_amplitude_ratio.py` |

The confidence-band and mask-frontier drivers are validation studies, not the
production retained-power estimator. `mask_residual_frontier.py` includes a
per-frame-floor comparison, discussed in the detector supplement, that the
method does not use; retained power is the signed mean in
`characterization/threshold_rule.py`.

## Stack amplitude ratio

`archive/derive_stack_amplitude_ratio.py` (`tools/derive_coherent_fraction.py`
in 1.0.0) derives the stack-to-proxy amplitude ratio from the frozen inputs, and
its `c.json` output keeps their field names. It reproduces the operating-duty
scenario of those records and does not select thresholds. The measured ratio is
not an intrinsic coherent fraction; see [terminology](../docs/TERMINOLOGY.md).
