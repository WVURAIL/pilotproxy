"""Stored prediction windows remain distinct from nominal RF geometry."""

import json

import numpy as np
import pytest

from pilot_proxy.products.reader import Product
from test_pilotproxy_v5 import _write_product


@pytest.mark.parametrize("bins,matched", [([126, 127, 128, 129, 130], False),
                                        ([255, 0, 1, 2, 3], True)])
def test_product_geometry_compares_scan_prediction_with_nominal_frequency(tmp_path, bins, matched):
    path = _write_product(tmp_path / "product.npz", 14, frames=6, units=2)
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    values["fine_designated_bins"] = np.array(bins, dtype=np.int64)
    contract = json.loads(str(values["detector_contract_json"]))
    contract["fine_reduction"]["designated_bins"] = bins
    values["detector_contract_json"] = np.asarray(json.dumps(contract))
    np.savez(path, **values)

    with Product(path) as product:
        geometry = product.geometry
        assert geometry.pilot_hz == 470_309_441.0
        assert geometry.centre_hz == 470_312_500.0
        assert geometry.sense == -1
        assert geometry.grid_residual_hz == 7.2421875
        # The 256-bin transform spans one 390625/128 Hz coarse-weight bin.
        assert geometry.nominal_fine_bin == 1
        assert geometry.stored_window_centre == bins[2]
        assert geometry.matches_stored_window is matched
        # A receiver-bin increment lowers RF frequency in this Nyquist zone.
        spacing = 390625.0 / 16384
        np.testing.assert_allclose(
            geometry.psd_rf_offset_hz([0, 1, 16383]),
            [3059.0, 3059.0 - spacing, 3059.0 + spacing], rtol=0, atol=1e-10,
        )
        np.testing.assert_allclose(
            geometry.psd_rf_offset_hz(geometry.nominal_psd_bin), 0.0, atol=1e-10,
        )
        np.testing.assert_allclose(
            geometry.psd_bin_of_rf_offset([0, 3059.0]),
            [3059.0 / spacing, 0], rtol=0, atol=1e-10,
        )
