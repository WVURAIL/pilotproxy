# First run

Install with python -m pip install -e ".[test,chime]".
Run python -m pytest -q and pilot-proxy --help.

Start with synthetic detector and estimator tests. Telescope archive access
requires its own credentials. Follow START_HERE.md for export, threshold
selection, and receiver configuration. The threshold command requires an
explicit conversion to the forecast's additive-power units.
