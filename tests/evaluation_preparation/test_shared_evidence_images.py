"""Shared saved image bindings may share bytes, not medical/check identity."""

import copy

import pytest

from packages.evaluation_preparation.evidence_assets import attach
from packages.grounding.models import digest
from tests.evaluation_preparation.test_evidence_tree_v2 import packet as old_media


def fixture():
    media = old_media()
    image = media["images"].pop("input")
    report, checks = {"observations": []}, {}
    for key in ("one", "two"):
        ref = dict(code="ASSET:" + key, checksum=image["checksum"], version="1.0.0", role="image")
        binding = dict(
            reference=ref,
            media_type="image/png",
            width=1,
            height=1,
            coordinate_system="pixel_xy_top_left",
            location=[0, 0, 1, 1],
            preprocessing=dict(
                code="POLICY:VIEW", version="1.0.0", role="preprocessing", checksum=digest("view")
            ),
        )
        report["observations"].append(dict(image=binding))
        media["images"][key] = dict(
            image,
            reference=ref,
            title=key,
            modality="synthetic",
            laterality="OD",
            timepoint="baseline",
        )
        checks[key] = dict(
            node_id=key, sections=dict(observation=dict(image=binding), dependencies=[])
        )
    return report, dict(source_checksum="sha256:test", checks=checks), media


def test_shared_saved_view_keeps_two_links_and_one_payload_without_mutation():
    report, packet, media = fixture()
    before = copy.deepcopy((report, media))
    result = attach(packet, report, media)["assets"]
    assert len(result["items"]) == 1
    assert {x["node_id"] for x in result["links"]} == {"one", "two"}
    assert {x["asset_id"] for x in result["links"]} == {"legacy:one"}
    shared = result["items"][0]["shared_image_bindings"]
    assert shared["schema_version"] == "shared-saved-image/1.0.0"
    assert len(shared["members"]) == 2
    assert {m["reference"]["code"] for m in shared["members"]} == {"ASSET:one", "ASSET:two"}
    assert (report, media) == before


@pytest.mark.parametrize(
    "change", ["eye", "time", "preprocessing", "missing_binding", "artifact", "bad_bytes"]
)
def test_equal_bytes_do_not_merge_different_provenance(change):
    report, packet, media = fixture()
    second = media["images"]["two"]
    if change == "eye":
        second["laterality"] = "OS"
    elif change == "time":
        second["timepoint"] = "followup"
    elif change == "preprocessing":
        report["observations"][1]["image"]["preprocessing"]["checksum"] = digest("other")
    elif change == "missing_binding":
        report["observations"] = []
    elif change == "artifact":
        second["reference"] = dict(asset_code="ARTIFACT:two", checksum=second["checksum"])
    else:
        second["data_uri"] = "data:image/png;base64,bad"
    assert len(attach(packet, report, media)["assets"]["items"]) == 2


def test_caller_alias_cannot_make_unreferenced_image_belong_to_check():
    report, packet, media = fixture()
    report["observations"] = []
    media["images"].pop("two")
    media["images"]["one"]["shared_image_bindings"] = dict(
        schema_version="shared-saved-image/1.0.0", members=[dict(reference=dict(code="ASSET:two"))]
    )
    result = attach(packet, report, media)["assets"]
    assert {x["node_id"] for x in result["links"]} == {"one"}
    assert "shared_image_bindings" not in result["items"][0]


def test_size_guard_applies_after_verified_sharing_but_remains_bounded():
    report, packet, media = fixture()
    # Public labels alone must not bypass the original 32 MiB limit.
    for value in media["images"].values():
        value["note"] = "x" * (17 * 1024 * 1024)
    report["observations"] = []
    with pytest.raises(ValueError, match="asset_package_size"):
        attach(packet, report, media)


def test_unbound_duplicate_keeps_identity_and_no_observation_link():
    report, packet, media = fixture()
    third = copy.deepcopy(media["images"]["two"])
    third["reference"]["code"] = "ASSET:unattempted"
    media["images"]["unattempted"] = third
    result = attach(packet, report, media)["assets"]
    assert len(result["items"]) == 2
    missing = next(a for a in result["items"] if a["id"] == "legacy:unattempted")
    assert missing["reference"]["code"] == "ASSET:unattempted"
    assert missing["shared_display_asset_id"] == "legacy:one"
    assert missing["display_status"] == "shared_bytes_see_linked_asset"
    assert "data_uri" not in missing
    assert not any(link["asset_id"] == missing["id"] for link in result["links"])


def test_unbound_other_eye_is_not_a_redundant_preview():
    report, packet, media = fixture()
    third = copy.deepcopy(media["images"]["two"])
    third["reference"]["code"] = "ASSET:other-eye"
    third["laterality"] = "OS"
    media["images"]["other"] = third
    result = attach(packet, report, media)["assets"]
    assert result["items"][-1]["display_status"] == "ready"
    assert "data_uri" in result["items"][-1]


def test_static_shared_preview_explains_original_reference_not_new_observation():
    from packages.evaluation_preparation.evidence_tree_static import media_html

    value = dict(
        id="view:unknown",
        title="Unknown check",
        display_status="shared_bytes_see_linked_asset",
        shared_display_asset_id="view:confirmed",
        reference=dict(code="ASSET:unknown"),
        display_note="相同视图，不计为新证据。",
    )
    html = media_html(value)
    assert "view:confirmed" in html and "ASSET:unknown" in html
    assert "非已取得观察" in html and "不计为新证据" in html
    assert "<img" not in html
