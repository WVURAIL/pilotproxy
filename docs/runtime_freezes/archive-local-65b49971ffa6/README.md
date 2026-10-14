# Runtime freeze archive-local-65b49971ffa6

The exact third-party environment the September 2026 CHIME archive scan ran
with (products at tag `archive-run-source-20260901`, package source sha256
3722012957975f7d...). `requirements.lock` pins every dependency by version
and wheel hash; `runtime_manifest.json` records how the environment was
assembled; `SHA256SUMS` covers the 79 wheels.

The wheels themselves are not in this repository. They are the
246 MB `pp_runtime_freeze.tar` kept with the campaign data (WVU RAIL OneDrive,
Datasets/chime_pilots_rebuild_20260829/kit/pp_switch/), and they are the only
way to rebuild this environment exactly: PyPI has since served different
manylinux builds of the same versions, so installing from this lock file
alone fails its own hash checks. The tar's sha256 is recorded in the run
ledger (f37fe041...).
