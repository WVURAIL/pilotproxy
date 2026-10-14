# Start here

Repository: [WVURAIL/pilotproxy](https://github.com/WVURAIL/pilotproxy)
([PROVENANCE.md](../PROVENANCE.md), [CHIME_DERIVED.md](../CHIME_DERIVED.md)).

The CHIME pipeline, from archive baseband to the mask kotekan applies. The
decisions of record (2026-10-12) rebuild exactly from the 2026-10-04 inputs
through steps 4 and 5.

**1. Archive.** The CUDA detector (`cuda/`, `archive/detector.py`) runs over
CHIME baseband and writes one product per pilot freq_id under
`<run>/_per_pilot/`. CHIME freq_id f sits at 800 MHz - f x 390625 Hz: channels
14 to 36 are freq_ids 844 down to 506, the channel 37 control band is 491
(`projects/chime_atsc/frequency_plan.yaml`). Runbooks:
[LOCAL_PROCESSING.md](LOCAL_PROCESSING.md), [CANFAR_RUNBOOK.md](CANFAR_RUNBOOK.md).

    pilot-proxy chime-survey --name chime-pilots --freq-ids <freq_ids>
    pilot-proxy chime-inventory --inventory-name chime-pilots
    pilot-proxy chime-scan --inventory-name chime-pilots --output-dir <run>

**2. Frame exports.** The release keys each frame to its eras and health gate;
the fine export adds each frame's exact minimum keep word
(`detectors/narrowband_marker/scores.py`, equal to the detector's
`fine_decision`). The control band uses `histogram-frames --control-product`.

    pilot-proxy characterize archive --products <run>/_per_pilot --out <release>
    pilot-proxy characterize histogram-frames --products <run>/_per_pilot --release <release>/characterization --output <frames>
    pilot-proxy characterize fine-histogram-frames --frames <frames> --output <fine>

**3. Eras.** Each channel's monthly level from the era review (2026-10-03),
cut only where a transmitter switches on or off
(`characterization/era_boundaries.py`): Bayesian Blocks change points whose step
lies above the largest gap in the step sizes over all channels, an excursion
that returns to its prior level merged away. The era list of record is
`projects/chime_atsc/eras/on_off_2026-10-14/eras.json`, with its monthly levels
and blocks beside it. The earlier list, `power_eras_2026-10-04.json`
(`characterization/power_eras.py`, `power_era_review.py`), is kept for the record.

    pilot-proxy characterize era-boundaries --review <era review> --previous <earlier era list> --output <eras>

**4. Score cache and knee.** Per channel, the current-era frames with their
words and equal-event weights (`characterization/score_cache.py`). The knee
(`characterization/knee.py`: the word that maximizes the equal-event cumulative
curve minus the log-word diagonal) is recomputed and must match the record. A
cache for new eras needs a working-threshold delivery in the 2026-10-04 layout
(each channel's months, score geometry and accepted replay).

    pilot-proxy characterize score-cache --working <delivery> --exports <export record> --previous <power-era thresholds> \
        --era-list projects/chime_atsc/eras/power_eras_2026-10-04.json --output <scores>

**5. Threshold rule** (`characterization/threshold_rule.py`). Deploy at the
knee when the upper end of the kept DTV power meets RFIsher's tolerance
(`characterization/data/tolerance_table.json`; coherent fraction
`characterization/data/coherent_fraction.json`, from
`tools/derive_coherent_fraction.py`; coherence gain G at the admissible minimum,
`characterization/data/coherent_gain.json`) and 6.574 / kappa calendar years stays
within 30; else at the loosest word below the knee that does; else not deployed.

    python tools/threshold_rule.py --scores <scores> --out <rule> \
        --exclude "33=pilot off the detector's frequency" --exclude "19=DTV without a matching pilot"

**6. Deployment compiler** (`operating_policy.py`,
[CHANNEL_OPERATING_POLICY.md](CHANNEL_OPERATING_POLICY.md)). The plan names the
receiver profile, `chime_freq_id`, `decisions.json`, the bundle, the consumer and
the era. Not deployed becomes `permanent_mask_freq_ids`; deployed becomes
`require_fine_freq_ids`, complete once its bundle profile is `calibrated` at
the decided Q16 word.

    pilot-proxy export-runtime-weight-bundle --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json \
        --weight-coordinate-system post_spectral_sense_normalization --physical-channel-range 14:36 --output-dir <bundle>
    pilot-proxy compile-operating-policy --plan policy_plan.json --output-dir <review> --require-complete

**7. kotekan.** `DtvRfiMask` reads the bundle (`dtv_runtime_bundle_dir`) and
the review's `kotekan_parameters`; start with `dtv_apply_mask: false`. The DTV
include exists today only in the CHORD F-engine template
(`config/fengine/chord.j2`); the CHIME upgrade's `config/chime_upgrade_n2.j2`
uses the same 16,384-sample frame and needs an include with the same keys.
Acceptance: [OPERATOR_HANDOFF.md](OPERATOR_HANDOFF.md).
