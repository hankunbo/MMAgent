import json

import pytest

from packages.evaluation_preparation.cohort_recovery import gap_inventory
from packages.grounding.models import digest


@pytest.mark.parametrize("missing", ["patient_input_missing:VF", "patient_modality_not_provided"])
def test_inventory_preserves_zero_null_and_unknown_without_authorizing(tmp_path, missing):
    ids = ["nonzero", "zero", "null", "unknown", "missing", "obtained", "uninspected"]
    requirements = [dict(id=i, presentation_sections=dict(acquisition={})) for i in ids]
    requirements[4]["presentation_sections"]["acquisition"]["missing_reason"] = missing
    requirements[5]["observation"] = dict(status="obtained", reference={"code": "OBS:X"})
    report = dict(
        requirements=requirements,
        scores=dict(
            reviews=[
                dict(id=i, judgment=dict(value=v))
                for i, v in zip(ids[:3], [-0.01, 0, None], strict=True)
            ]
        ),
    )
    audit = dict(
        original_owner_verified=True,
        original_score_json_html_equal=True,
        report_checksum=digest(report),
        cumulative_requests=17,
        unconfirmed_phase_requests=[dict(target="unknown", request_id="old-unknown")],
    )
    rp, ep = tmp_path / "report.json", tmp_path / "evidence.json"
    rp.write_text(json.dumps(report))
    ep.write_text(json.dumps(dict(cases=dict(case=dict(audit=audit)))))
    result = gap_inventory(rp, ep, "case")
    assert [r["status"] for r in result["rows"]] == [
        "reviewed_nonzero",
        "reviewed_neutral",
        "reviewed_null",
        "unknown_attempt",
        "patient_input_missing",
        "obtained_without_current_review",
        "requires_owner_inspection",
    ]
    assert all(r["dispatch_authorized"] is False for r in result["rows"])
    assert result["cumulative_requests"] == 17 and result["new_api_requests"] == 0
    assert json.loads(rp.read_text()) == report
    report["scores"]["reviews"][1]["judgment"]["value"] = 0.001
    rp.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="report_audit_binding"):
        gap_inventory(rp, ep, "case")


@pytest.mark.parametrize("remote_unknown", [False, True])
def test_saved_unscored_observation_is_not_unsent_or_replay_permission(tmp_path, remote_unknown):
    observation = dict(
        status="unknown",
        method="live_image_observation",
        reference=dict(code="OBS:OCT", version="2.0.0", checksum=digest("observation")),
        value="Visible inner band; independent layer boundaries unclear.",
    )
    report = dict(
        requirements=[
            dict(
                id="oct",
                observation=None,
                unscored_observation=observation,
                presentation_sections=dict(acquisition=dict(missing_reason="not_yet_observed")),
            )
        ],
        scores=dict(reviews=[]),
    )
    audit = dict(
        original_owner_verified=True,
        original_score_json_html_equal=True,
        report_checksum=digest(report),
        cumulative_requests=4,
        unconfirmed_phase_requests=[dict(target="oct", request_id="old-unknown")]
        if remote_unknown
        else [],
    )
    rp, ep = tmp_path / "report.json", tmp_path / "evidence.json"
    rp.write_text(json.dumps(report))
    ep.write_text(json.dumps(dict(cases=dict(case=dict(audit=audit)))))
    before = rp.read_bytes()
    result = gap_inventory(rp, ep, "case")
    row = result["rows"][0]
    assert result["schema_version"] == "terminal-case-gap-inventory/1.1.0"
    assert row["status"] == "saved_observation_requires_interpretation"
    assert row["observation_reference"] == observation["reference"]
    assert row["observation_status"] == "unknown"
    assert row["observation_pointer"] == "/requirements/0/unscored_observation"
    assert row["observation_checksum"] == digest(observation)
    assert row["unknown_request_ids"] == (["old-unknown"] if remote_unknown else [])
    assert row["dispatch_authorized"] is False
    assert (
        row["missing_reason"] == "not_yet_observed"
    )  # old wording preserved, not treated as truth
    assert rp.read_bytes() == before
