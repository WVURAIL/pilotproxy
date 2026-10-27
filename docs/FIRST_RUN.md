# First run: from a clean checkout to a paper result

The purpose of this walkthrough is to reproduce a paper result from saved
inputs, starting from a clean checkout. It needs Git, Python 3.10 or newer with
`venv`, and access to the repositories. CUDA, telescope data, FIR, and CANFAR
are not needed. The commands assume a Linux shell.

## 1. Install and check the detector package

Start in a new working directory. If a repository is private, authenticate to
GitHub first. These credentials are distinct from the telescope archive
credentials.

```sh
git clone https://github.com/WVURAIL/pilotproxy.git
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e "pilotproxy[test]"
cd pilotproxy
pilot-proxy --help
python -m pytest tests/test_cli.py -q
pilot-proxy check-profile \
    --receiver-profile configs/receiver_profiles/reference_800mhz_pfb.json
pilot-proxy check-layout \
    --receiver-profile configs/receiver_profiles/reference_800mhz_pfb.json
```

The tests and both validation commands should pass. These checks cover the
installation and the receiver geometry and leave GPU arithmetic untested. For CUDA builds and native tests, continue with
[installation and usage](USAGE.md).

## 2. Read and rebuild one forecast channel

PilotProxy measures the pilot and assigns a threshold. RFIsher supplies the
response of the channel to residual DTV power. RadioFisher is the underlying
clean cosmological forecast. Install RFIsher alongside this checkout:

```sh
cd ..
git clone https://github.com/WVURAIL/RFIsher.git
python -m pip install -e "RFIsher[test]"
cd RFIsher
python - <<'PY'
import rfisher
row = next(r for r in rfisher.read_table()["channels"] if r["channel"] == 19)
print(f"Channel 19: response = {row['response_dv']:.0f}, a_tol = {row['a_tol']:.3g}")
PY
rfisher table --channels 19 --workers 1 --out out/first-run
rfisher figures --table out/first-run/channel_table.json --out out/first-run/figures
python -m pytest tests/test_table.py -q
```

The shipped row prints approximately `response = 8344, a_tol = 0.00012`.
The rebuild writes JSON and CSV files, and the figure command reads that new
table by its explicit path. The tests compare channel 19 with the shipped
result using documented numerical tolerances. The forecast takes longer than
reading the table, but it runs only one channel on one worker.

RFIsher installs its recorded RadioFisher revision automatically. Do not
replace that dependency with RadioFisher's latest source in this environment.
To develop RadioFisher itself, use its README in an environment of its own.

## 3. Reproduce the paper's threshold decisions

Clone the MNRAS manuscript beside the two code repositories, then run the
read-only check:

```sh
cd ..
git clone https://github.com/WVURAIL/dtv-mnras.git
cd pilotproxy
python tools/check_paper_thresholds.py --paper-root ../dtv-mnras
```

A successful check prints JSON with `"status": "pass"` and
`"channels_checked": 24`. The saved sweeps, responses, budgets, threshold
words, and calendar-year comparison of the paper are checked, together with the
frozen calculation under its recorded time convention. No data refresh or TeX
installation is needed.

These are conditional threshold calculations. This check does not cover the
measurement of the assumed temporal gain `G`, the transfer of `gamma` to a new
data population, or a live telescope run.

## 4. Trace a result and choose the next workflow

Follow [the paper/code map](PAPER_CODE_MAP.md) from each defined quantity to
its implementation and tests. For the threshold calculation, start with
`strength`, `sweep`, and `decide_guarded` in
[`characterization/threshold_rule.py`](../src/pilot_proxy/characterization/threshold_rule.py).
The [terminology reference](TERMINOLOGY.md) explains why the production score
is named a marker-to-reference ratio, and when it can be treated as an F
statistic.

- To process new observations, follow the [pipeline guide](START_HERE.md), then
  the [local](LOCAL_PROCESSING.md) or [CANFAR](CANFAR_RUNBOOK.md) runbook.
- To build the manuscript, use the README in `dtv-mnras`. Its saved data are
  distinct from the original archive products.
- To work on receiver integration, use the [operator handoff](OPERATOR_HANDOFF.md).
  Match the vendored core and native API versions before building kotekan.

Keep generated results under `out/` or another output directory. Frozen
evidence and paper inputs record the original measurements and stay in place
when new code is tested.
