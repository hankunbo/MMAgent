"""The inner definition guard honors only the exact owner-verified repair pair."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS

import pytest

from packages.grounding.models import digest
from packages.llm_provider.provider import ProviderError
from packages.multimodal_research.definition_expansion import DefinitionExpansion


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "no_grant",
        "old_grant",
        "failure_changed",
        "closure_changed",
        "repair_failed",
        "cancelled",
        "expired",
    ],
)
def test_preserved_markers_are_not_global_bypass(tmp_path, fault):
    failure, closure = (
        {"error_code": "stage_previous_failure_read_only"},
        {"primary": "local_owner_error"},
    )
    (tmp_path / "stage-failure.json").write_text(json.dumps(failure))
    (tmp_path / "stage-closure-error.json").write_text(json.dumps(closure))
    grant = dict(
        schema_version="confirmed-material-local-repair/1.6.0",
        original_failure_checksum=digest(failure),
        original_closure_checksum=digest(closure),
    )
    c = NS(
        output=tmp_path,
        local_material_repair=grant,
        protocol=dict(deadline=(datetime.now(UTC) + timedelta(minutes=10)).isoformat()),
    )
    if fault == "no_grant":
        c.local_material_repair = {}
    elif fault == "old_grant":
        grant["schema_version"] = "confirmed-material-local-repair/1.5.0"
    elif fault == "failure_changed":
        (tmp_path / "stage-failure.json").write_text("{}")
    elif fault == "closure_changed":
        (tmp_path / "stage-closure-error.json").write_text("{}")
    elif fault == "repair_failed":
        (tmp_path / "local-repair-failure.json").write_text("{}")
    elif fault == "cancelled":
        (tmp_path / "cancel").touch()
    elif fault == "expired":
        c.protocol["deadline"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    owner = object.__new__(DefinitionExpansion)
    owner.campaign = c
    if fault is None:
        assert owner.fast_control() is None
    else:
        with pytest.raises(ProviderError):
            owner.fast_control()


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("fault", [None, "reserved", "budget_changed", "extra"])
def test_budget_initialization_is_not_a_dispatch_but_reservations_still_reject(
    monkeypatch, tmp_path, fault, container
):
    from packages.multimodal_research import pending_review as p

    docs, ctx = (
        NS(run_id="run", read=lambda _: None),
        NS(
            run_id="run",
            request=NS(
                input_checksum="input",
                task=NS(checksum="task"),
                visible_checksum="visible",
                scope=NS(case_code="case"),
            ),
        ),
    )
    events = [
        dict(
            schema_version="agent-budget/1.0.0",
            binding=dict(run_id="run", deadline="until", max_requests=40),
        )
    ]
    if fault == "reserved":
        events.append(dict(phase="reserved", request_id="unknown"))
    elif fault == "budget_changed":
        events[0]["binding"]["max_requests"] = 400
    elif fault == "extra":
        events[0]["unknown"] = True
    monkeypatch.setattr(p, "projection_context", lambda *args, **kwargs: (docs, ctx))
    monkeypatch.setattr(
        "packages.research_runtime.agent_records.RuntimeAgentJournal",
        lambda _: NS(read=lambda: container(events)),
    )
    c = NS(
        protocol=dict(input="input", task="task", deadline="until"),
        config=NS(max_requests=40),
        base=NS(visible_value=NS(checksum="visible"), binding=dict(case_code="case")),
        sessions=NS(read=lambda _: None),
        phase_path=lambda _: tmp_path,
    )
    if fault is None:
        assert p.partition_projection_unstarted(c, definition=True, budget_binding_only=True) == (
            docs,
            ctx,
        )
    else:
        with pytest.raises(ValueError, match="budget_binding"):
            p.partition_projection_unstarted(c, definition=True, budget_binding_only=True)
