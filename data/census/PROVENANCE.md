# Transmitter-census provenance

## Sources and licenses

- Canadian stations: station and license data of Innovation, Science and
  Economic Development Canada (ISED), used in the workbook and in the ISED
  overlay described below (licensed transmitter sites, license status and
  `erp_kw`). Contains information licensed under the Open Government Licence -
  Canada (https://open.canada.ca/en/open-government-licence-canada).
- US stations: public station records of the FCC Licensing and Management
  System (LMS), compiled into the workbook.
- Field strengths: the 42 `detectability_db` values come from a RabbitEars.info
  Signal Search Map study (see "Optional field-strength screen"). The RabbitEars
  terms of service, read on 2026-10-07, are an accuracy disclaimer under a site
  copyright notice, and they state no terms for reusing study results.

The source workbook is `TV_Stations_UHF_within500mi_DRAO.xlsx`, retrieved on
2026-06-09. The workbook, the reduction script, and the upstream source
material are in the lab's archived census repository (dtv-census), available
on request. This directory vendors its derived `census.csv`, which is the
reduction described below plus the ISED overlay of the census repository
(Canadian rows re-sited to licensed transmitter coordinates, license-status
adjudication, and `erp_kw`). All 499 rows declare
`schema_version=dtv_transmitter_census_v1` and carry `evidence_status`, so the
11 license-only candidates can be distinguished from rows reported on air.
The vendored file has SHA-256
`05caae7a509e8a18eb98cb63e8637a985ace4eb503ecf06cc17a6da298dca12f` and is
byte-identical to `census/census.csv` at every commit of the census repository
(`bdc22f56`, `d2cc1e3d`, `5480e166`).

The archive is a full-history git bundle of the census repository
(`dtv-census-all-refs.bundle`, with a `README.md` that lists the rest). Its
`census/PROVENANCE.md` and `census/VERIFICATION.md` hold the rules and
verdicts. The reduction below describes the workbook stage. The workbook is a
manually compiled listing of FCC LMS and ISED station information, and no
propagation model was used to build it. Every UHF ATSC television station
listed in a 500 statute mile radius of DRAO (`49.3208 N`, `119.6239 W`) is in
the census. We used the station class and frequency-tolerance
fields to interpret possible pilot offsets. We did not use the census alone
to claim that a station was detectable at CHIME.

## Optional field-strength screen

The workbook also contains `detectability_db` for a limited nearby subset.
These values came from a RabbitEars Signal Search Map study run on
2026-06-09 at 12:15 ET, shareable ID `2738863`. The study radius was 120
statute miles, and receive height was set to the tool maximum of 99999999 ft
AGL. This setting largely removes terrain blocking, so these field strengths
are optimistic upper-bound estimates and do not predict the field at a site.

The final merged CSV contains 42 rows with field strength. Before the
channel-sharing merge, 43 values were present. Rows past the 120-mile study
radius have no field strength, and the association code ranks them by
distance when it needs a rank. The archived RabbitEars result-list printout was
committed in
the census repository as `sources/rabbitears/dtv_120m.pdf` and is in the
bundle.

## Deterministic reduction

Regenerate the committed CSV (workbook reduction plus ISED overlay) from a
clone of the bundle, with the packages in its `requirements.txt`:

```bash
git clone dtv-census-all-refs.bundle dtv-census
cd dtv-census
python3 ingest/apply_ised_overlay.py
```

The output is byte-identical to the vendored file.

The overlay command runs `census/census_from_xlsx.py` for the deterministic
workbook reduction before it applies the licensed Canadian transmitter sites,
status adjudication, and ERP fields. Running `census_from_xlsx.py` alone
reproduces only the pre-overlay census. The 499-row CSV vendored here needs the
overlay as well.

The row counts below were reproduced by running the reduction of the census
repository over the retained workbook. The RabbitEars study settings above are
recorded source metadata and were not independently verified here. The
reduction applies the following rules:

1. The workbook contains 521 rows. We retain the 485 rows whose `Type` is
   `on-air`. We remove 23 `off-air` rows because they do not describe a
   current emission, 9 analog rows because they have no ATSC pilot, and 4
   ATSC 3.0 rows because OFDM does not contain the 8-VSB pilot used here.
2. We parse every integer in `Physical Ch(s)` and emit one row for each value
   in 14--36. Multi-channel entries therefore expand. One on-air entry,
   `CH5643-DT`, has no UHF token and is excluded because it is VHF-only. Channels
   2--13 occupy 54--216 MHz and lie below CHIME's 400--800 MHz band.
3. The expansion produces 494 emitter-channel rows. Four pairs share an RF
   channel and site, so we merge each pair into one physical carrier. The
   final CSV contains 490 emitter-channel rows: 93 with service class
   `Full-power` and 397 in the other recorded classes.
4. `detectability_db` is the RabbitEars field strength in dBuV/m when one is
   available. Otherwise it is blank. Association ranking then falls back to
   distance.
5. We convert distance from miles to kilometers and carry the bearing,
   frequency tolerance, CHIME channel index, nominal pilot frequency, city,
   and state or province through when present. Five output rows have no CHIME
   index and retain `nan` for nominal pilot frequency because those fields are
   absent in the source workbook.

## Frequency-tolerance boundary

At the unmerged source-row level, full-power, Class A, and Relay entries use
`±1 kHz`, while Translator and LPTV entries use `None specified`. The merge
promotes the service class to the more primary partner, but it keeps the
frequency-tolerance field of the first row. As a result of this rule, two
merged rows (`KJYY-LD+KOPB-TV` and `KCKW-LD+KEZI`) are labeled `Full-power`
and keep `None specified`. For these channel-sharing rows, the tolerance comes
from the first source row and cannot be inferred from the merged class.
