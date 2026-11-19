"""Current aggregate binding, history protection and original lease boundaries."""

import json
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.accepted_aggregation import (
    ACCEPTANCE_KEY,
    GRANT_KEY,
    AggregationPermission,
)
from packages.grounding.models import digest
from packages.multimodal_research import alternative_acquisition as a
from packages.multimodal_research import campaign_renewal as c
from packages.multimodal_research import execution_lease as lease
from tests.expert_agent.test_accepted_aggregation import accepted_origin, aggregate_fixture


def original_aggregate(tmp_path, monkeypatch):
    f = accepted_origin(monkeypatch, tmp_path)
    combined = aggregate_fixture(f)
    permission, submission = AggregationPermission.prepare(
        owner=combined.owner,
        documents=combined.documents,
        requests=combined.requests,
        config=f.config,
    )
    service = ResearchAcceptance(
        owner=combined.owner,
        documents=combined.documents,
        requests=combined.requests,
        live_permission=permission,
    )
    service.submit(submission, key=ACCEPTANCE_KEY)
    assert service.accept(key=ACCEPTANCE_KEY).status == "accepted"
    state = combined.state
    manifest = dict(state_checksum=digest(state), submission_checksum=digest(submission))
    docs = combined.documents
    docs.write("accepted-aggregation-manifest", (manifest,))
    docs.write("dynamic-phase-input", (state.model_dump(mode="json"),))
    result = dict(complete=True, phases=[dict(run_id=docs.run_id, target=None)])
    docs.write("multimodal-case-result", (result,))
    path = tmp_path / "aggregate"
    path.mkdir()
    (path / "result.json").write_text(json.dumps(result))
    baseline = NS(
        state=state,
        submission=submission,
        manifest=manifest,
        documents=docs,
        path=path,
        grant=docs.read(GRANT_KEY)[0],
    )
    parent = NS(
        aggregation_baseline=baseline,
        base=NS(),
        config=f.config,
        accepted_observations=lambda: {},
        sessions=docs.sessions,
    )

    class ProjectedPhase:
        def __init__(self, base, observation_reader):
            assert base is parent.base

        def project(self, *, run_id, config):
            assert run_id == docs.run_id and config == f.config
            return state, None

        def verify(self, current):
            # The old origins still pass through the original current owner.
            combined.source.current(current)

    monkeypatch.setattr("packages.multimodal_research.campaign.Phase", ProjectedPhase)
    return parent, combined, f


def test_current_original_aggregate_can_prove_only_unscored_target(tmp_path, monkeypatch):
    p, combined, f = original_aggregate(tmp_path, monkeypatch)
    before = deepcopy(combined.documents.saved)
    v = a.aggregation_origin(p, "new-unscored")
    assert v["original_judgments_not_replaced"]
    assert v["submission_checksum"] == digest(p.aggregation_baseline.submission)
    assert combined.documents.saved == before
    target = p.aggregation_baseline.submission.candidate.items[0].assessment_id
    with pytest.raises(ValueError, match="must_not_replace_accepted_judgment"):
        a.aggregation_origin(p, target)


@pytest.mark.parametrize(
    "fault", ["state", "submission", "manifest", "input", "result", "control", "grant", "source"]
)
def test_current_aggregate_drift_is_not_a_reusable_authorization(tmp_path, monkeypatch, fault):
    p, combined, f = original_aggregate(tmp_path, monkeypatch)
    b = p.aggregation_baseline
    d = b.documents
    if fault == "state":
        b.state = b.state.model_copy(update={"config_checksum": digest("changed")})
    elif fault == "submission":
        b.manifest["submission_checksum"] = digest("changed")
    elif fault == "manifest":
        d.saved["accepted-aggregation-manifest"] = ({"changed": True},)
    elif fault == "input":
        d.saved["dynamic-phase-input"] = ({"changed": True},)
    elif fault == "result":
        (b.path / "result.json").write_text(json.dumps(dict(complete=False)))
    elif fault == "control":
        current = p.sessions.read(d.run_id)
        p.sessions = NS(read=lambda _: current.model_copy(update={"cancelled": True}))
    elif fault == "grant":
        d.saved[GRANT_KEY] = ({"changed": True},)
    else:
        f.state = f.state.model_copy(update={"config_checksum": digest("withdrawn")})
    before = deepcopy(d.saved)
    with pytest.raises(ValueError):
        a.aggregation_origin(p, "new-unscored")
    assert d.saved == before


def aggregate_plan(tmp_path):
    from tests.evaluation_preparation.test_execution_lease import aggregate_fixture as lease_fixture

    campaign = lease_fixture(tmp_path)
    campaign.protocol.update(
        schema_version=a.AGGREGATE_VERSION, alternative_view={"method": a.TILED_METHOD}
    )
    return campaign


def test_new_aggregate_uses_original_runtime_lease_without_legacy_upgrade(tmp_path):
    campaign = aggregate_plan(tmp_path)
    old = deepcopy(campaign.protocol)
    lease.declare_aggregate(campaign.protocol, campaign.parent)
    lease.validate_aggregate(campaign)
    assert campaign.protocol["execution_lease_profile"] == lease.PROFILE
    assert "execution_lease_profile" not in old
    assert a.AGGREGATE_VERSION not in c.EXACT_ACQUISITION_VERSIONS
    assert a.AGGREGATE_VERSION not in c.RETAINED_ACQUISITION_VERSIONS


@pytest.mark.parametrize(
    "change",
    [
        {"alternative_view": None},
        {"acquisition_increment_only": False},
        {"review_target_ids": ["wrong"]},
        {"accepted_origin_checksum": "changed"},
        {"schema_version": "multimodal-campaign-renewal/1.26.0"},
    ],
)
def test_aggregate_alternative_lease_rejects_mixed_scope(tmp_path, change):
    campaign = aggregate_plan(tmp_path)
    campaign.protocol.update(change)
    with pytest.raises(ValueError, match="aggregate_declaration_binding"):
        lease.declare_aggregate(campaign.protocol, campaign.parent)
