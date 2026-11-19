"""Archive boundary contracts; isolated original-owner PG lives in evaluation tests."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.memory.scope_archive import check_phase_input
from packages.multimodal_research.input_scope import accepted_fact
from packages.multimodal_research.scope_selection import storage_identity
from tests.agent_provider.test_input_scope import fixture, generate


@pytest.mark.parametrize(
    "field", [None, "case_code", "run_id", "input_checksum", "visible_checksum"]
)
def test_phase_and_case_scopes_compare_their_actual_binding_fields(field):
    f = fixture()
    request = SimpleNamespace(
        scope=SimpleNamespace(case_code=f.scope.case_code),
        input_checksum=f.scope.input_checksum,
        visible_checksum=f.scope.visible_checksum,
    )
    context = SimpleNamespace(run_id=f.scope.run_id, request=request)
    if field == "case_code":
        request.scope.case_code = "CASE:OTHER"
    elif field == "run_id":
        context.run_id = "RUN:OTHER"
    elif field:
        setattr(request, field, digest("other"))
    if field:
        with pytest.raises(ValueError, match="memory_scope_phase_input_binding"):
            check_phase_input(f.reader.state, context)
    else:
        check_phase_input(f.reader.state, context)


@pytest.mark.parametrize(
    "fault", [None, "task", "run", "unaccepted", "score", "pending", "running", "receipt"]
)
def test_shared_fact_requires_original_acceptance_and_no_disease_vote(fault):
    f = fixture()
    f.acceptor.submit(generate(f), key="scope")
    f.acceptor.accept(key="scope")
    values = deepcopy(f.acceptor.read(key="scope"))
    phase = dict(run_id=f.scope.run_id, accepted=True, acceptance_key="scope")
    task = f.task.model_dump(mode="json")
    control = SimpleNamespace(pending=None, phase="stopped")
    if fault == "task":
        task["component_id"] = "other"
    elif fault == "run":
        phase["run_id"] = "RUN:OTHER"
    elif fault == "unaccepted":
        phase["accepted"] = False
    elif fault == "score":
        values[2]["scores"] = {"invented_vote": 1}
    elif fault == "pending":
        control.pending = object()
    elif fault == "running":
        control.phase = "running"
    elif fault == "receipt":
        values[3]["records_checksum"] = digest("other")
    if fault:
        with pytest.raises(ValueError):
            accepted_fact(values, f.reader.state, phase, task, control)
    else:
        fact = accepted_fact(values, f.reader.state, phase, task, control)
        assert fact["disease_votes"] == 0
        assert fact["records_checksum"] == digest(values[0])
        assert fact["regions"] == values[0]["candidate"]["regions"]


def test_numeric_location_is_bound_to_original_campaign_and_exact_plan():
    plan = dict(
        schema_version="input-scope-method-plan/1.1.0",
        storage_basis="original_input_capabilities_without_visual_survey",
        scope_records=[],
        rows=[],
    )
    first = storage_identity(plan, "original")
    assert first == storage_identity(deepcopy(plan), "original")
    assert first[0] != storage_identity(plan, "other-campaign")[0]
    assert first != storage_identity(dict(plan, rows=[{"id": "new"}]), "original")
    with pytest.raises(ValueError, match="requires_accepted_survey"):
        storage_identity(dict(plan, schema_version="input-scope-method-plan/1.0.0"), "original")
    with pytest.raises(ValueError, match="requires_accepted_survey"):
        storage_identity(dict(plan, storage_basis="unverified"), "original")


def test_visual_location_uses_original_scope_run_and_explicit_version():
    plan = dict(
        schema_version="input-scope-method-plan/1.0.0",
        scope_records=[dict(phase=dict(run_id="RUN:SCOPE"))],
    )
    assert storage_identity(plan, "original") == (
        "RUN:SCOPE",
        "input-scope-method-plan:" + digest(plan)[7:],
    )
    with pytest.raises(ValueError, match="requires_accepted_survey"):
        storage_identity(dict(plan, schema_version="input-scope-method-plan/9.0.0"), "original")


@pytest.mark.parametrize("fault", [None, "request", "original", "claim", "task", "score"])
def test_recovered_scope_archive_keeps_original_failed_phase_and_new_admission(fault, version=None):
    from packages.evidence_details.scope_finalization import ACCEPTANCE_KEY, GRANT_KEY
    from packages.memory.scope_archive import survey_origin
    from packages.multimodal_research.capability_window import claim
    from packages.multimodal_research.scope_successor import VERSION, claim_key

    entry = dict(
        target="input-scope:example", task={"component_id": "image"}, view_checksum=digest("view")
    )
    old_phase = dict(run_id="RUN:OLD", target=entry["target"], accepted=False)
    old = dict(
        input_scope={"provider_checksum": digest("old-scope-provider")},
        capability_transition={"scope_plan": {"entries": [entry]}},
    )
    grant = dict(run_id="RUN:OLD", original_request_checksum=digest("confirmed-request"))
    phase = dict(
        schema_version="recorded-input-scope-result/1.0.0",
        run_id="RUN:OLD",
        target=entry["target"],
        original_phase=old_phase,
        original_phase_checksum=digest(old_phase),
        acceptance_key=ACCEPTANCE_KEY,
        grant_checksum=digest(grant),
        original_request_checksum=grant["original_request_checksum"],
        new_api_requests=0,
        new_disease_scores=0,
        campaign_resumed=False,
    )
    saved = dict(phase=phase, disease_votes=0)
    proof = dict(
        protocol_checksum=digest(old),
        reused=[dict(target=entry["target"], fact=saved, fact_checksum=digest(saved))],
    )
    protocol = dict(schema_version=version or VERSION, scope_reuse=proof, output="/new")
    docs = {
        "dynamic-phase-result": [old_phase],
        "dynamic-campaign": [old],
        "recorded-input-scope:result:1": [phase],
        GRANT_KEY: [grant],
        claim_key(proof): [claim(protocol, "/new")],
    }
    docs = deepcopy(docs)
    if fault == "request":
        docs[GRANT_KEY][0]["original_request_checksum"] = digest("other")
    elif fault == "original":
        docs["dynamic-phase-result"][0]["accepted"] = True
    elif fault == "claim":
        docs[claim_key(proof)] = []
    elif fault == "task":
        entry = dict(entry, task={"component_id": "other"})
    elif fault == "score":
        docs["recorded-input-scope:result:1"][0]["new_disease_scores"] = 1

    def read(key, **kwargs):
        return docs[key]

    if fault:
        with pytest.raises(ValueError, match="memory_scope_recovery"):
            survey_origin(read, protocol, entry, saved)
    else:
        assert (
            survey_origin(read, protocol, entry, saved) == old["input_scope"]["provider_checksum"]
        )
        assert docs["dynamic-phase-result"][0]["accepted"] is False


@pytest.mark.parametrize("fault", [None, "request", "original", "claim", "task", "score"])
def test_current_recovered_scope_archive_keeps_bound_original_lineage(fault):
    from packages.multimodal_research.scope_successor import CURRENT_VERSION

    test_recovered_scope_archive_keeps_original_failed_phase_and_new_admission(
        fault, CURRENT_VERSION
    )
