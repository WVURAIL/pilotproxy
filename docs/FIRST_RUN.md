# First run

Install with python -m pip install -e ".[test,chime]".
Run python -m pytest -q and pilot-proxy --help.

Start with synthetic detector and estimator tests. Telescope archive access
requires its own credentials. Follow START_HERE.md for calibration, testing
and receiver configuration. The power-limit test requires the RFIsher channel
table of the telescope, which converts the shelf estimate to the forecast's
additive-power units.
