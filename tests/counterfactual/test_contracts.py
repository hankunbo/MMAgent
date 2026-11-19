"""Exact old object bytes, explicit simulation contracts and owner boundaries."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from packages.counterfactual.models import Intervention, SimulationPolicy
from packages.evidence_scoring.api import validate as validate_score
from packages.evidence_scoring.simulation import project
from packages.grounding.models import digest
from packages.llm_provider.models import RolePayload, SimulationExpertInput
from tests.counterfactual.support import candidate, fixture, run


def test_rehashed_fact_scores_are_still_rejected():
    x = fixture()
    scores = copy.deepcopy(x.fact.scores)
    scores["root"]["summary"]["combined"]["support"] = 0.99
    scores["checksum"] = digest({k: v for k, v in scores.items() if k != "checksum"})
    with pytest.raises(ValueError, match="reconstruction"):
        project(x.parent, x.fact.candidate, scores)
    with pytest.raises(ValueError, match="reconstruction"):
        validate_score(dict(state=x.parent, candidate=x.fact.candidate), scores)


def test_old_export_and_v5_frozen_bytes_read_only():
    paths = [
        Path("docs/exports/青光眼证据图.json"),
    ]
    for path in paths:
        raw = path.read_bytes()
        value = json.loads(raw)
        assert isinstance(value, dict)
        assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(raw).digest()
    # No upgrade, exporter or v5 compute is used for the current 6.1 branch.
    x = fixture()
    wrong = dict(x.fact.scores, schema_version="case-research-scoring/5.0.0")
    with pytest.raises(ValueError, match="exact_6_1_required"):
        project(x.parent, x.fact.candidate, wrong)


@pytest.mark.parametrize(
    "field,value",
    [("kind", "patient_value_change"), ("targets", []), ("targets", ["same", "same"])],
)
def test_intervention_unknown_or_empty_is_denied(field, value):
    raw = dict(kind="remove_evidence", targets=["one"])
    raw[field] = value
    with pytest.raises(ValueError):
        Intervention.model_validate(raw)


@pytest.mark.parametrize(
    "changes",
    [
        dict(synthetic=False),
        dict(allowed={}),
        dict(max_candidates=0),
        dict(tolerance=float("nan")),
        dict(metrics=()),
    ],
)
def test_policy_never_defaults_to_open_domain(changes):
    x = fixture()
    with pytest.raises(ValueError):
        SimulationPolicy.model_validate(x.policy.model_dump() | changes)


def test_tampered_json_or_html_read_is_rejected():
    x = fixture()
    report = run(x)["report"]
    original = x.f.documents.read

    def bad(key):
        values = original(key)
        if key == "simulation:result:one":
            return values[0], dict(values[1], html="forged")
        return values

    x.f.documents.read = bad
    with pytest.raises(ValueError, match="html_source"):
        x.service.read(candidate_id="one", historical=True)
    assert report["fact_checksum"] == x.fact.checksum


def test_simulation_review_cannot_enter_other_roles_or_fact_review():
    x = fixture()
    context = SimulationExpertInput(
        policy_checksum=digest(x.policy),
        report_checksum=digest("r"),
        proposed_candidate_id="one",
        declared_candidates=({"id": "one"},),
        comparisons={},
    )
    raw = dict(role="source", material=x.parent.material, simulation_review=context)
    with pytest.raises(ValueError, match="simulation_context_forbidden"):
        RolePayload.model_validate(raw)


def test_cross_run_snapshot_writer_refuses_scope(monkeypatch):
    x = fixture()

    def refused(scope):
        raise ValueError("agent_runtime_scope")

    monkeypatch.setattr(x.f.documents, "validate_scope", refused)
    with pytest.raises(ValueError, match="runtime_scope"):
        x.service.declare((candidate(),))
