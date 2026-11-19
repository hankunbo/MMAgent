"""Staged contract coverage; not a new clinical or real API result."""

import json
from types import SimpleNamespace as NS

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import alternative_acquisition as a
from packages.multimodal_research import campaign_renewal as c
from tests.evaluation_preparation.test_exact_acquisition_budget import configuration, valid_plan


def plan():
    p = dict(
        valid_plan(),
        schema_version=a.VERSION,
        mode="acquisition",
        review_context_scope=c.RETAINED_CONTEXT_SCOPE,
        seconds=5400,
        acquisition_targets=["check"],
        max_new_requests=3,
        max_model_dispatches=3,
        new_public_search_allowed=False,
        alternative_view={"original": "proof"},
        reuse_assessment_ids=["old"],
    )
    p["acquisition_selection"] = dict(
        schema_version="bounded-acquisition-selection/1.1.0",
        eligible=["check"],
        selected=["check"],
        deferred=[],
        eligibility_owner="MultimodalCampaign.available_targets",
    )
    return p


def test_explicit_new_version_preserves_original_exact_budget():
    p = plan()
    c.validate_exact_acquisition_budget(p, configuration())
    assert c.acquisition_review_ids(p) == ("check", "old")
    assert a.VERSION in c.RETAINED_ACQUISITION_VERSIONS


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": c.RETAINED_ACQUISITION_VERSION},
        {"mode": "review"},
        {"acquisition_targets": ["a", "b"]},
        {"seconds": 5401},
        {"max_new_requests": 4},
        {"new_public_search_allowed": True},
        {"unknown_acquisition": {"request": "unknown"}},
        {"review_clarification": {"assessment_ids": ["old"]}},
        {"alternative_view": {"changed": True}},
    ],
)
def test_changed_scope_cannot_authorize_alternative(monkeypatch, change):
    monkeypatch.setattr(a, "proof", lambda parent, target: {"original": "proof"})
    p = plan()
    assert a.check(NS(), p) == ["check"]
    p.update(change)
    with pytest.raises(ValueError, match="alternative_view_plan_binding"):
        a.check(NS(), p)


def test_old_unattempted_selector_still_excludes_unknown_and_confirmed():
    p = NS(
        initial_targets=["old", "unknown", "fresh"],
        base=NS(
            ledger=[
                dict(id=x, component="image", acquisition_method="vision")
                for x in ["old", "unknown", "fresh"]
            ]
        ),
    )
    assert c.unattempted_targets(p, {"unknown"}, {"old": object()}) == ["fresh"]


def test_explicit_method_refuses_mixed_flags_before_owner_access(tmp_path):
    with pytest.raises(ValueError, match="alternative_view_explicit_bounded_scope"):
        c.prepare(
            facts=tmp_path,
            previous=tmp_path,
            output=tmp_path / "new",
            mode="acquisition",
            authorization="test",
            exact_request_budget=True,
            request_limit=3,
            acquisition_target=["check"],
            alternative_view_method=a.METHOD,
            definition_increment_only=True,
        )
    assert not (tmp_path / "new").exists()


def view_owner(tmp_path):
    from db.runtime.repository.case_pipeline import identity
    from packages.object_store import checksum

    raw = b"synthetic-contract-payload-not-a-medical-image"
    root_code = identity("ASSET", "alternative-view-test-root")
    proof = dict(
        method=a.METHOD,
        target="check",
        source_checksum=checksum(raw),
        native_checksum=digest("native"),
        case_code=identity("CASE", "alternative-view-test-case"),
        native_artifact=dict(
            asset_code=identity("ASSET", "alternative-view-test-native"),
            laterality="LEFT",
            checksum=digest("native"),
            derived_from=[root_code],
            provenance=dict(input_checksums=[checksum(raw)]),
        ),
    )
    metadata = dict(
        method=a.METHOD,
        source_checksum=proof["source_checksum"],
        native_checksum=proof["native_checksum"],
        checksum=checksum(raw),
        width=1,
        height=1,
    )
    (tmp_path / "alternative-view").mkdir()
    (tmp_path / "alternative-view/view.png").write_bytes(raw)
    (tmp_path / "alternative-view/view.json").write_text(json.dumps(metadata))
    bundles = {}
    payloads = {}
    writes = []
    checks = []

    def register(values, key):
        if key in bundles:
            assert bundles[key] == values
        else:
            bundles[key] = values
            writes.append(key)

    def promote(raw, expected):
        from packages.object_store import checksum

        assert checksum(raw) == expected
        uri = "object://knowledge/objects/alternative-test"
        payloads[uri] = raw
        return uri

    artifact = proof["native_artifact"]
    base = NS(
        check=lambda: checks.append("current"),
        root=NS(
            asset_code=artifact["derived_from"][0],
            checksum=artifact["provenance"]["input_checksums"][0],
        ),
        runtime=NS(
            read_observation_bundle=lambda key: bundles.get(key),
            register_observation_bundle=register,
        ),
        objects=NS(
            stage=lambda raw: raw, promote=promote, read_bounded=lambda uri, limit: payloads[uri]
        ),
    )
    owner = a.AlternativeView(
        NS(base=base),
        dict(
            alternative_view=proof,
            alternative_view_metadata_checksum=digest(metadata),
            started_at="2026-10-27T18:00:00+00:00",
        ),
        tmp_path,
    )
    return owner, bundles, writes, checks


def test_registered_view_is_idempotent_and_exact_derivation(tmp_path):
    owner, bundles, writes, checks = view_owner(tmp_path)
    with pytest.raises(ValueError, match="registered_artifact_required"):
        owner.view()
    owner.ensure()
    first = owner.view()
    owner.ensure()
    assert owner.view() == first
    assert len(writes) == 1 and len(checks) >= 4
    artifact = bundles[owner.key][0]
    assert tuple(artifact.derived_from) == (
        owner.parent.base.root.asset_code,
        owner.spec["native_artifact"]["asset_code"],
    )
    assert first[0].preprocessing.code == "POLICY:ALTERNATIVE_VIEW"
    assert first[1] == b"synthetic-contract-payload-not-a-medical-image"


def test_view_bytes_tamper_rejected_without_register(tmp_path):
    owner, _, writes, _ = view_owner(tmp_path)
    (tmp_path / "alternative-view/view.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="material_changed"):
        owner.ensure()
    assert writes == []


def test_current_source_revocation_rejected_before_register(tmp_path):
    owner, _, writes, _ = view_owner(tmp_path)

    def revoked():
        raise ValueError("current_source_withdrawn")

    owner.parent.base.check = revoked
    with pytest.raises(ValueError, match="current_source_withdrawn"):
        owner.ensure()
    assert writes == []


def test_immutable_saved_artifact_metadata_mismatch_rejected(tmp_path):
    owner, bundles, _, _ = view_owner(tmp_path)
    owner.ensure()
    artifact = bundles[owner.key][0]
    bundles[owner.key] = (artifact.model_copy(update={"extensions": {"validated": True}}),)
    with pytest.raises(ValueError, match="artifact_binding"):
        owner.view()


def test_report_keeps_historical_observation_as_exact_provenance_not_current_image(tmp_path):
    from copy import deepcopy

    from packages.evaluation_preparation.shared_evidence_images import saved_bindings

    owner, _, _, _ = view_owner(tmp_path)
    owner.ensure()
    current = owner.view()[0].model_dump(mode="json")
    old = deepcopy(current)
    old["reference"]["checksum"] = digest("old-view")
    original = dict(image=old, status="unknown", reference={"code": "OBS:OLD"})
    owner.spec["original"] = dict(
        observation=original,
        observation_checksum=digest(original),
        run_id="RUN:OLD",
        request_id="request:old",
        target=owner.target,
    )
    campaign = NS(base=owner.parent.base, protocol={"alternative_view": owner.spec})
    campaign.base.alternative_view = owner
    report = dict(
        requirements=[
            dict(
                id=owner.target,
                observation=dict(image=current),
                execution_binding={},
                presentation_sections={},
            )
        ]
    )
    before = deepcopy(owner.spec)
    a.report_binding(campaign, report)
    bindings = saved_bindings(report)
    assert bindings[current["reference"]["code"]][0] == current
    history = report["alternative_method"]["proof"]["original"]
    assert history["observation_checksum"] == digest(original)
    assert history["observation_location"] == dict(
        protocol_checksum=digest(campaign.protocol),
        pointer="/alternative_view/original/observation",
    )
    assert owner.spec == before and "observation" not in history
    assert report["alternative_method"]["proof_checksum"] == digest(before)
    report["unexpected_current_observation"] = dict(image=old)
    with pytest.raises(ValueError, match="asset_shared_reference_conflict"):
        saved_bindings(report)


def test_alternative_recovery_never_accepts_scoring_or_provider_drift():
    from packages.multimodal_research.report_recovery import alternative_report_delta

    before = {
        "packages/multimodal_research/alternative_acquisition.py": "old",
        "packages/multimodal_research/report_recovery.py": "old",
        "packages/evidence_scoring/continuous.py": "original",
    }
    changed = {**before, "packages/multimodal_research/alternative_acquisition.py": "new"}
    assert alternative_report_delta(before, changed) == [
        "packages/multimodal_research/alternative_acquisition.py"
    ]
    assert alternative_report_delta(before, before) == []
    with pytest.raises(ValueError, match="implementation_scope"):
        alternative_report_delta(
            before, {**changed, "packages/evidence_scoring/continuous.py": "changed"}
        )
    with pytest.raises(ValueError, match="implementation_scope"):
        alternative_report_delta(before, {**changed, "new_uncovered_owner.py": "new"})


def test_tiled_method_requires_explicit_version_and_same_budget(monkeypatch):
    p = dict(plan(), schema_version=a.TILED_VERSION)
    calls = []

    def checked(parent, target, *, method=a.METHOD):
        calls.append(method)
        return {"original": "proof"}

    monkeypatch.setattr(a, "proof", checked)
    assert a.check(NS(), p) == ["check"]
    assert calls == [a.TILED_METHOD]
    c.validate_exact_acquisition_budget(p, configuration())
    assert a.TILED_VERSION in c.RETAINED_ACQUISITION_VERSIONS
    with pytest.raises(ValueError, match="alternative_view_plan_binding"):
        a.check(NS(), dict(p, max_new_requests=4))


def ancestor_fixture(tmp_path):
    from packages.llm_provider.models import ObservationInput

    observation = ObservationInput(
        reference=dict(role="observation", code="OBS:TEST", version="1.0.0", checksum=digest("o")),
        status="unknown",
        method="live_image_observation",
        unit="qualitative",
        value="visible part",
    )
    base = NS(manifest=NS(checksum=digest("input")), task=NS(checksum=digest("task")))

    def make(name, phases, parent=None):
        directory = tmp_path / name
        directory.mkdir()
        result = dict(complete=True, phases=phases)
        (directory / "result.json").write_text(json.dumps(result))
        return NS(
            output=directory,
            parent=parent,
            base=base,
            accepted_observations=lambda: {"check": observation},
            documents=lambda rid: NS(read=lambda key: (result,)),
        )

    old = make("old", [dict(target="check", run_id="vision"), dict(target=None, run_id="review")])
    latest = make("latest", [dict(target=None, run_id="next-review")], old)
    return latest, old, observation


def test_ancestor_lookup_matches_current_fact_and_only_follows_parent_chain(tmp_path):
    parent, old, _ = ancestor_fixture(tmp_path)
    assert a.observation_origin(parent, "check") is old
    parent.parent = None
    with pytest.raises(ValueError, match="confirmed_ancestor_not_found"):
        a.observation_origin(parent, "check")


def test_ancestor_lookup_rejects_observation_or_input_drift(tmp_path):
    parent, old, observation = ancestor_fixture(tmp_path)
    old.accepted_observations = lambda: {
        "check": observation.model_copy(update={"value": "changed"})
    }
    with pytest.raises(ValueError, match="ancestor_observation_changed"):
        a.observation_origin(parent, "check")
    old.base = NS(manifest=NS(checksum=digest("changed")), task=parent.base.task)
    with pytest.raises(ValueError, match="ancestor_input_or_task_changed"):
        a.observation_origin(parent, "check")


def test_ancestor_lookup_never_skips_newer_target_attempt(tmp_path):
    parent, old, observation = ancestor_fixture(tmp_path)
    old.accepted_observations = lambda: {"check": observation.model_copy(update={"value": "other"})}
    parent.parent.parent = old
    with pytest.raises(ValueError, match="ancestor_observation_changed"):
        a.observation_origin(parent, "check")


def test_tiled_artifact_reports_its_method_without_promoting_native_measurement(tmp_path):
    owner, bundles, _, _ = view_owner(tmp_path)
    owner.spec["method"] = a.TILED_METHOD
    meta_path = owner.output / "view.json"
    meta = json.loads(meta_path.read_text())
    meta.update(method=a.TILED_METHOD, native_localization_used=False)
    meta_path.write_text(json.dumps(meta))
    owner.plan["alternative_view_metadata_checksum"] = digest(meta)
    owner.ensure()
    artifact = bundles[owner.key][0]
    assert artifact.extensions["method"] == a.TILED_METHOD
    assert artifact.extensions["metadata"]["native_localization_used"] is False
    assert not artifact.measurements


def test_public_cli_dispatches_explicit_tiled_method_to_original_prepare(monkeypatch):
    from apps.multimodal_case import main

    calls = []

    def prepare(**kwargs):
        calls.append(kwargs)
        return dict(prepared=False, synthetic_cli_binding_only=True)

    monkeypatch.setattr(c, "prepare", prepare)
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["multimodal_case"]
        + [
            "renewal-prepare",
            "--facts",
            "old",
            "--previous",
            "old",
            "--output",
            "new",
            "--mode",
            "acquisition",
            "--authorization",
            "synthetic CLI test",
            "--acquisition-target",
            "check",
            "--exact-request-budget",
            "--request-limit",
            "3",
            "--alternative-view-method",
            a.TILED_METHOD,
        ],
    )
    assert main() == 0
    assert calls[0]["alternative_view_method"] == a.TILED_METHOD
    assert calls[0]["acquisition_target"] == ["check"]


def test_partial_ancestor_does_not_erase_confirmed_phase_but_current_parent_must_complete(
    tmp_path, monkeypatch
):
    parent, origin, _ = ancestor_fixture(tmp_path)
    path = origin.output / "result.json"
    raw = json.loads(path.read_text())
    raw["complete"] = False
    raw["phases"][0]["accepted"] = False
    path.write_text(json.dumps(raw))
    current = parent.output / "result.json"
    value = json.loads(current.read_text())
    value["complete"] = True
    current.write_text(json.dumps(value))
    monkeypatch.setattr(a, "observation_origin", lambda *_: origin)
    # A partial batch is allowed, but an unconfirmed phase is still rejected.
    with pytest.raises(ValueError, match="alternative_view_confirmed_phase_required"):
        a.original_observation(parent, "check", ancestor_allowed=True)
    value["complete"] = False
    current.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="alternative_view_completed_parent_required"):
        a.original_observation(parent, "check", ancestor_allowed=True)
