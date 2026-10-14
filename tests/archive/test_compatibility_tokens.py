from pilot_proxy.archive import pipeline
from pilot_proxy.archive.inventory import (
    INVENTORY_META_SCHEMA_KEY,
    INVENTORY_META_SCHEMA_VERSION,
)
from pilot_proxy.archive.sources.cadc_inventory import logical_unit_key


def test_persisted_archive_tokens_stay_compatible():
    assert pipeline._OUTPUT_LOCK_SUFFIX == ".datatrawl.lock"
    assert INVENTORY_META_SCHEMA_KEY == "datatrawl_inventory"
    assert INVENTORY_META_SCHEMA_VERSION == 1
    assert logical_unit_key("scope", "event", "name.h5") == (
        'cadc-datatrail:["scope","event","name.h5"]'
    )
