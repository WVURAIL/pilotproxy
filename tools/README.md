# Tools catalog

This directory holds standalone analysis and maintenance commands. Run them
from the repository root after installing the package. The supported package
and CLI are in `src/pilot_proxy/`, and production code does not import these
tools. Each command lists its required inputs under `--help`. New outputs are
never written into the frozen evidence directories.

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

The confidence-band and mask-frontier drivers are validation studies. The
production retained-power estimator is the signed mean in
`characterization/threshold_rule.py`. `mask_residual_frontier.py` also reports
a per-frame-floor variant, discussed in the detector supplement, that the
method does not use.

## Stack amplitude ratio

`archive/derive_stack_amplitude_ratio.py` derives the stack-to-proxy amplitude
ratio from the frozen inputs. In release 1.0.0 it was
`tools/derive_coherent_fraction.py`. Its `c.json` output keeps the field names
of those inputs. The operating-duty scenario of the frozen records is
reproduced, and no thresholds are selected. On the calibrated scale, the ratio
is the amplitude ratio `gamma` of the [terminology](../docs/TERMINOLOGY.md),
the redundant-stack amplitude over the archive shelf proxy. It covers the
spatial phase cancellation of calibrated redundant stacking and the
archive-to-capture transfer.
