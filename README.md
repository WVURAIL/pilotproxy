# PilotProxy

PilotProxy detects the narrow ATSC 1.0 television pilot and uses its measured
excess power to estimate the broadband DTV shelf. It provides the fixed-point
GPU detector, archive reduction, threshold assignment, and runtime-bundle
compiler used by the DTV papers.

**Start with [the pipeline guide](docs/START_HERE.md).** For equations, code
locations, and verification tests, use [the paper/code map](docs/PAPER_CODE_MAP.md).
The [terminology reference](docs/TERMINOLOGY.md) distinguishes the measured power
ratios, rank-based detection score, and conditional forecast quantities.

## Choose a workflow

| Goal | Entry point |
|---|---|
| Understand or reproduce the paper calculations | [Pipeline guide](docs/START_HERE.md), [paper/code map](docs/PAPER_CODE_MAP.md) |
| Process a local CHIME archive inventory | [Local runbook](docs/LOCAL_PROCESSING.md) |
| Process archive data on CANFAR | [CANFAR runbook](docs/CANFAR_RUNBOOK.md) |
| Install, build CUDA, or run a synthetic bench | [Installation and usage](docs/USAGE.md) |
| Compile a receiver policy | [Operating policy](docs/CHANNEL_OPERATING_POLICY.md) |
| Commission the kotekan integration | [Operator handoff](docs/OPERATOR_HANDOFF.md) |
| Find an analysis or development utility | [Tools catalog](tools/README.md) |

CHIME archive access requires collaboration credentials. This source tree is
[WVURAIL/pilotproxy](https://github.com/WVURAIL/pilotproxy); access and data
boundaries are recorded in [CHIME_DERIVED.md](CHIME_DERIVED.md).
[PROVENANCE.md](PROVENANCE.md) identifies the historical source snapshots.

## Install and check

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
order are specified by a receiver profile, not inferred from the library name.

```python
from pilot_proxy.kernel import PilotProxyKernel

kernel = PilotProxyKernel()
print(kernel.specs.as_descriptive_dict())
print(kernel.version.as_string())
```

`PilotProxyKernel` binds the `PilotProxy_*` C API in `libpilotproxy.so`.
Package 2 / core 4 removes the old F-statistic interface names. Rebuild the
library and update callers together; see [migration notes](docs/TERMINOLOGY.md#native-interface-migration).

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
| `tools/archive/` | Superseded study drivers retained for reproducing old records |
| `docs/` | Current contracts and operating guides |
| `docs/evidence/`, `docs/runtime_freezes/` | Frozen receipts and provenance; not onboarding instructions |

Development tools are not imported by the detector package. Useful historical
records stay separate from active instructions. Generated data, figures,
shared libraries and documentation PDFs belong in output/build directories.

## Scientific and compatibility boundaries

Core 3.0.0 defines each frame power as the all-bin sum of its fixed-point fine
spectrum. Current products use schema v6. The paper's schema-v5 archives remain
unchanged; current readers reconstruct these powers from their stored fine
spectra. They do not use the retired row-domain powers.

The threshold rule combines the measured shelf proxy with RFIsher's response,
a measured stack-to-proxy amplitude ratio, and an assumed effective temporal
gain. Assigned words are conditional candidates. A compiled bundle or a passing
replay does not establish telescope safety or a cross-day BAO contamination bound.

The primary time comparison is the Overview's ideal calendar year: retaining
fraction `kappa` costs `1 / kappa` years. The old operating-duty scenario is an
explicit reproduction option, documented in [the pipeline guide](docs/START_HERE.md).

## Maintainer checks

```sh
make lint
make test-python
make docs-specs-check
```

On a CUDA host also run `make test-gpu`. [Test coverage](docs/TEST_COVERAGE.md)
explains the wider release checks. Preserve frozen inputs and records when
changing source; new runs must record their own source and build identities.
Do not regenerate dated evidence just to make a refactor pass.

## Citation

Use [CITATION.cff](CITATION.cff), the version from `pilot-proxy --version`, and
the actual source/build identifiers of a run. See [CHANGELOG.md](CHANGELOG.md)
for interface changes and [the survey trust contract](docs/survey_trust.md)
for provenance and recovery rules.
