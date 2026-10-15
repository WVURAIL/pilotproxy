# CHIME-derived material

Some files in this repository hold results derived from CHIME data: the CHIME/FRB
baseband archive, the 2026 matched baseband capture, and the products and releases
built from them. They are aggregates only. No file here holds voltages, visibilities,
gain values or a listing of archive events. They are kept because the dissertation and
its papers reproduce from them.

Their public release needs CHIME collaboration clearance under its publication policy.
Until that clearance is given, a public copy of this repository must be the code-only
cut described at the end of this page. Material that is not CHIME data (the synthetic
GNU Radio benches, the LimeSDR lab runs, the designed weight banks, the public
FCC/ISED transmitter census and RFIsher's forecast tolerance table) is not listed.

## Data files

| Path | What it holds |
|---|---|
| `docs/evidence/ca_os_archive_bulk_2026-09-29/` (65 files) | Archive-bulk statistics of all 23 ATSC channels (2026-09-29): per-channel bulk statistics, persistent-bin profiles and geometry (`data/bulk/chNN.json`), the per-channel release ledger with product SHA-256, era months and frame counts, the producer and verifier code, and the report |
| `docs/evidence/ch37_rank_injection_2026-09-30/` (143 files) | Channel 37 control study (2026-09-30): rank curves and real-null exceedance rates of the ch37 control product (`data/controls_summary.json`), per-geometry frame metadata, beside a synthetic rank bench; with the result, check and verifier |
| `docs/evidence/youden_j_2026-09-07/` (2 files) | Coarse-versus-fine ROC and Youden J on the released per-pilot products: channel 35 off-era frames as the null, channels 34 to 36 from 2025 as the signal |
| `docs/evidence/F1_INVENTORY.md` | Index of the two packages above (file counts, bytes, manifest SHA-256) |
| `data/provenance/combine_subset_20260714/` (3 files) | Stack-subset decision record: the count of archive events per freq_id presence signature (190 signatures with counts, no event numbers) and the selected 16-channel block |
| `data/provenance/dissertation_summary_v1.json`, `_v2.json`, `_v3.json` | Curated per-channel and per-epoch values of the 2026-08 archive trawl (mask fractions, retained frames, status) used by dissertation drafts |
| `projects/chime_atsc/eras/author_eras_2026-09-23.json` | Era boundaries per channel, dated from the archive's monthly spectrograms |
| `projects/chime_atsc/eras/transmitter_off.json` | Transmitter on/off months per channel, read from the archive's pilot series, with the external record found for each |
| `projects/chime_atsc/eras/power_eras_2026-10-04.json` | Power-era proposals per channel (current-era months) of the 2026-10-04 threshold release |
| `projects/chime_atsc/eras/on_off_2026-10-14/` (3 files) | Era list of record: monthly pilot levels per channel from the 2026-10-03 era review (`monthly_levels.json`), their Bayesian Blocks (`blocks.json`) and the on/off eras (`eras.json`) |
| `src/pilot_proxy/characterization/data/coherent_fraction.json` | Coherent fraction of the DTV shelf, per band and its median, measured from the 2026 matched capture |
| `src/pilot_proxy/characterization/data/coherent_gain.json` | Coherence-gain rule and the frame count of the capture dumps that measured the coherent fraction |
| `tests/data/threshold_decisions_2026-10-12.json` | Decisions of record per channel (2026-10-12): deployed word or reason not deployed, shelf, exclusions and the coherent-fraction inputs; a test golden for the deployment compiler |

## Inline quotes

These files are code or documentation, but they quote CHIME-derived numbers inline.
They stay in both cuts. Review them before any public release:

- `docs/ARCHIVE_HEALTH_REPAIR.md`, `docs/CANFAR_RUNBOOK.md` and
  `docs/RERUN_PARAMETER_REGISTER.md`: archive inventory accounting (units, events,
  exclusions, frame counts).
- `docs/DESIGN_DECISIONS.md`: archive measurements behind design choices (for example
  the coarse flag's false-alarm rate on quiet frames).
- `docs/POWER_ERAS.md` and `docs/START_HERE.md`: current-era spans of channels 20, 27
  and 34, and the two exclusions of record (channels 33 and 19).
- `CHANGELOG.md`: railed-sample counts measured on real CHIME baseband (freq_id 506).
- Test goldens written inline: the knees of record (`tests/characterization/test_knee.py`),
  the era starts and outages (`test_era_boundaries.py`), the verified off dates
  (`test_transmitter_off.py`), the era literals (`tests/config/test_profile_literals.py`)
  and the coherent-fraction range (`test_threshold_rule.py`). The tests that need the
  withheld files skip in the code-only cut, but their source still holds these values.

## Code-only cut

`python tools/make_code_only_cut.py OUTPUT_DIR` (or `make code-only-cut OUT=OUTPUT_DIR`)
copies the tree without the paths below. The project profile needs an era list, so the
cut writes empty stand-ins for `author_eras_2026-09-23.json` and `transmitter_off.json`
(no dated channel, no off interval). Tests that read a withheld file, or a value only the
withheld era list has, skip with the reason "withheld from the code-only cut".

```
docs/evidence/ca_os_archive_bulk_2026-09-29/
docs/evidence/ch37_rank_injection_2026-09-30/
docs/evidence/youden_j_2026-09-07/
docs/evidence/F1_INVENTORY.md
data/provenance/combine_subset_20260714/
data/provenance/dissertation_summary_v1.json
data/provenance/dissertation_summary_v2.json
data/provenance/dissertation_summary_v3.json
projects/chime_atsc/eras/author_eras_2026-09-23.json
projects/chime_atsc/eras/transmitter_off.json
projects/chime_atsc/eras/power_eras_2026-10-04.json
projects/chime_atsc/eras/on_off_2026-10-14/
src/pilot_proxy/characterization/data/coherent_fraction.json
src/pilot_proxy/characterization/data/coherent_gain.json
tests/data/threshold_decisions_2026-10-12.json
```
