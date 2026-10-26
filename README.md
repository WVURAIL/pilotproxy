# PilotProxy

PilotProxy detects the narrow pilot of ATSC 1.0 digital television (DTV) and
uses its measured excess power to estimate the broadband DTV shelf. It provides
the fixed-point GPU detector, archive reduction, threshold assignment, and
runtime-bundle compiler used by the DTV papers.

The [first-run walkthrough](docs/FIRST_RUN.md) describes how to install the
code and reproduce a saved paper calculation without telescope access. It is
followed by the [pipeline guide](docs/START_HERE.md). The equations, their code
locations, and their verification tests are listed in
[the paper/code map](docs/PAPER_CODE_MAP.md), and the
[terminology reference](docs/TERMINOLOGY.md) defines the measured power ratios,
the rank-based detection score, and the conditional forecast quantities.

## Workflows

| Goal | Entry point |
|---|---|
| Get a first result in a clean environment | [First-run walkthrough](docs/FIRST_RUN.md) |
| Understand or reproduce the paper calculations | [Pipeline guide](docs/START_HERE.md), [paper/code map](docs/PAPER_CODE_MAP.md) |
| Process a local CHIME archive inventory | [Local runbook](docs/LOCAL_PROCESSING.md) |
| Process archive data on CANFAR | [CANFAR runbook](docs/CANFAR_RUNBOOK.md) |
| Install, build CUDA, or run a synthetic bench | [Installation and usage](docs/USAGE.md) |
| Compile a receiver policy | [Operating policy](docs/CHANNEL_OPERATING_POLICY.md) |
| Commission the kotekan integration | [Operator handoff](docs/OPERATOR_HANDOFF.md) |
| Find an analysis or development utility | [Tools catalog](tools/README.md) |

Access to the CHIME archive requires collaboration credentials. This source
tree is [WVURAIL/pilotproxy](https://github.com/WVURAIL/pilotproxy), and its
access and data boundaries are recorded in [CHIME_DERIVED.md](CHIME_DERIVED.md).
[PROVENANCE.md](PROVENANCE.md) records the snapshot this repository starts from.

## Installation and checks

Use a virtual environment with Python 3.10 or newer, then run from the checkout:

```sh
python -m pip install -e ".[test]"
pilot-proxy --help
make test-python
```

The Python tests do not require archive access. GPU checks run when their
prerequisites are present. A CUDA host can build and run the detector with:

```sh
make build-kernel DETECTOR_WINDOW_SAMPLES=128
make test-gpu DETECTOR_WINDOW_SAMPLES=128
```

The detector window is 128 samples for CHIME and 64 for CHORD. Each stream
contains 128 windows per detector block. Receiver geometry, packing, and sample
order are specified by a receiver profile. The library name carries none of them.

```python
from pilot_proxy.kernel import PilotProxyKernel

kernel = PilotProxyKernel()
print(kernel.specs.as_descriptive_dict())
print(kernel.version.as_string())
```

`PilotProxyKernel` binds the `PilotProxy_*` C API of detector core 4 in
`libpilotproxy.so`. The library and its callers must be rebuilt together. The names
of earlier cores are mapped in
[the terminology reference](docs/TERMINOLOGY.md#native-names-before-core-4).

## Repository layout

| Location | Purpose |
|---|---|
| `cuda/` | Fixed-point detector, C ABI, CPU reference, native tests |
| `src/pilot_proxy/` | Supported package, CLI, archive adapters, calibration and policy compiler |
| `configs/`, `weights/` | Receiver profiles, layouts and shipped packed weights |
| `projects/chime_atsc/` | Frequency plan and dated era definitions |
| `tests/` | Numerical contracts, archive integrity, and runtime interface tests |
| `scripts/` | Installation, CI and archive operating scripts |
| `tools/` | Explicit analysis, validation and maintenance commands |
| `tools/archive/` | Study drivers that reproduce frozen records |
| `docs/` | Contracts and operating guides |
| `docs/evidence/`, `docs/runtime_freezes/` | Frozen receipts and provenance records |

The detector package imports no development tool, and the frozen records are
kept apart from the guides. Generated data, figures, shared libraries, and
documentation PDFs are written to output and build directories.

## Scientific and compatibility boundaries

Each frame power is the all-bin sum of its fixed-point fine spectrum. Products
use schema v6. Readers also open the papers' schema-v5 archives and form the
same powers from their stored fine spectra.

The threshold rule is formed from the measured shelf proxy, RFIsher's response,
a measured stack-to-proxy amplitude ratio, and an assumed effective temporal
gain. Assigned words are conditional candidates. This package does not cover
telescope safety or a cross-day baryon acoustic oscillation (BAO) contamination
bound, and neither follows from a compiled bundle or a passing replay.

By default, the threshold rule counts time in the ideal calendar years of the
CHIME Overview, where retaining a fraction `kappa` of the data costs `1 / kappa`
years.
[The pipeline guide](docs/START_HERE.md) documents the operating-duty option
that reproduces the frozen records made with it.

## Maintainer checks

```sh
make lint
make test-python
make docs-specs-check
```

On a CUDA host, `make test-gpu` is also run. [Test coverage](docs/TEST_COVERAGE.md)
describes the wider release checks. Frozen inputs and records stay unchanged when
the source changes, and new runs record their own source and build identities.
Dated evidence is never regenerated to make a refactor pass.

## Citation

Use [CITATION.cff](CITATION.cff), the version from `pilot-proxy --version`, and
the actual source and build identities of a run. See [CHANGELOG.md](CHANGELOG.md)
for interface changes and [the survey trust contract](docs/survey_trust.md)
for provenance and recovery rules.

### Figure colors

Plots and diagrams use the WVU colors from the
[Beamer template](https://github.com/WVURAIL/wvu_beamer_template), with a standalone
copy in `src/pilot_proxy/wvu_palette.py`. The copy is updated with the template's
`tools/export_plot_palette.py`. Blue marks measurements, and old gold marks model
curves and small labels. Further series use the WVU accent colors with distinct
markers or line styles where needed. Grayscale masks keep their binary meaning.
Journal requirements take precedence over this palette, which changes no
numerical calculation or frozen record.
