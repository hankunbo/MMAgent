"""Multi-component native facts must reach the existing display owner unchanged."""

import copy
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.report import media, native_artifacts


def fixture(collection=True):
    artifacts = [
        dict(
            asset_code=f"ASSET:NATIVE:{index}",
            checksum=digest(f"image-{index}"),
            uri=f"saved:{index}",
            media_type="image/png",
            dimensions=[2, 2],
            extensions=dict(output_port="overlay.png", component=f"visit:{index}:fundus"),
        )
        for index in range(2 if collection else 1)
    ]
    records = [
        dict(
            schema_version="native-fundus-acceptance/1.1.0"
            if collection
            else "native-fundus-acceptance/1.0.0",
            plan=dict(target=f"cup:{index}", component=f"visit:{index}:fundus"),
            artifacts=[artifact],
        )
        for index, artifact in enumerate(artifacts)
    ]
    native = (
        dict(schema_version="native-fundus-collection/1.0.0", records=records)
        if collection
        else records[0]
    )
    reads = []

    def read(uri, maximum):
        reads.append((uri, maximum))
        assert uri in {a["uri"] for a in artifacts}
        return uri.encode()

    campaign = SimpleNamespace(
        base=SimpleNamespace(
            observations={},
            active_owner=SimpleNamespace(native=native),
            objects=SimpleNamespace(read_bounded=read),
            manifest=SimpleNamespace(checksum=digest("input")),
        ),
        accepted_observations=lambda: {},
    )
    report = dict(result=dict(case_code="CASE:PUBLIC"), requirements=[])
    return campaign, report, artifacts, reads


@pytest.mark.parametrize("collection", [False, True])
def test_native_report_media_preserves_all_component_artifacts(collection):
    campaign, report, artifacts, reads = fixture(collection)
    original = copy.deepcopy(campaign.base.active_owner.native)
    assert native_artifacts(campaign.base) == tuple(artifacts)
    images = media(campaign, report)["images"]
    assert set(images) == {a["asset_code"] for a in artifacts}
    assert reads == [(a["uri"], 4194304) for a in artifacts]
    assert all(images[a["asset_code"]]["reference"] == a for a in artifacts)
    assert campaign.base.active_owner.native == original


def test_native_collection_duplicate_binding_is_not_silently_rendered():
    campaign, report, _, reads = fixture()
    native = campaign.base.active_owner.native
    native["records"][1]["plan"] = native["records"][0]["plan"]
    with pytest.raises(ValueError, match="native_fundus_collection_binding"):
        media(campaign, report)
    assert reads == []
