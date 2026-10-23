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
| Historical capture-ratio derivation | `archive/derive_stack_amplitude_ratio.py` |

The confidence-band and mask-frontier drivers are validation studies, not the
production retained-power estimator. In particular, `mask_residual_frontier.py`
retains a superseded per-frame-floor comparison discussed in the detector
supplement; current retained power is the signed mean in
`characterization/threshold_rule.py`. Their tests and recorded outputs explain
why those alternatives were not adopted, so they are kept.

## Historical driver migration

`derive_coherent_fraction.py` moved to
`archive/derive_stack_amplitude_ratio.py`. Its archived inputs and `c.json`
output retain their original fields. It reproduces an older duty scenario and
must not be used to select today's thresholds. The measured ratio is not an
intrinsic coherent fraction; see [terminology](../docs/TERMINOLOGY.md).
