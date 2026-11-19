"""07C independent replay/report boundary checks using immutable synthetic facts."""

import copy

import pytest

from apps.graph_versioning import GraphVersionAuditContext, build_graph_version_commit
from db.audit.repository import AuditEventRecord
from db.runtime.repository.case_pipeline import identity
from packages.contracts import GraphDelta
from packages.deterministic_pipeline.reporting import accepted_validations
from packages.graph_engine import (
    GraphEngineError,
    canonical_checksum,
    plan_graph_delta,
    transition_reset_reference,
)
from tests.deterministic_pipeline.test_reporting import facts as facts


def transition(f, graph, target, before, after, *, reset_to=None):
    raw = f.validations[0]["item"].graph_delta_candidate.model_dump(mode="json")
    reasons = [str(graph.provenance.rule_versions[0].code)]
    refs = [graph.provenance.rule_versions[0].model_dump(mode="json")]
    if reset_to:
        markers = [
            "RULE:TRANSITION_RESET_V1",
            "RULE:TRANSITION_RESET_REQUIRED",
            "RULE:TRANSITION_RESET_OP_000_DERIVED_" + reset_to,
        ]
        reasons.extend(markers)
        refs.extend(transition_reset_reference(r).model_dump(mode="json") for r in markers)
    raw.update(
        base_version=graph.graph_version,
        proposed_by="RULE_ENGINE",
        proposal_kind="GRAPH_DELTA_VALIDATION",
        idempotency_key=identity("TASK", "07c:" + str(graph.graph_version)),
        reason_codes=reasons,
        operations=[
            dict(
                op="UPDATE_NODE_STATUS",
                node_code=target,
                from_status=before,
                to_status=after,
                source="DETERMINISTIC_RULE",
            )
        ],
    )
    raw["provenance"].update(actor="RULE_ENGINE", rule_versions=refs)
    delta = GraphDelta.model_validate(raw)
    plan = plan_graph_delta(graph, delta)
    commit = build_graph_version_commit(
        graph,
        plan,
        GraphVersionAuditContext(
            event_id=identity("MESSAGE", "07c:" + str(graph.graph_version)),
            correlation_code="SOURCE:DETERMINISTIC_PIPELINE",
            occurred_at=graph.provenance.created_at,
        ),
    )
    event = AuditEventRecord.model_validate(
        commit.command.audit_event.model_dump(mode="python")
        | {
            "event_sequence": max(e.event_sequence for e in f.audits) + graph.graph_version,
            "recorded_at": graph.provenance.created_at,
            "payload_checksum": canonical_checksum(commit.command.audit_event.payload),
            "event_checksum": canonical_checksum(commit.command.audit_event),
        }
    )
    return commit.command.graph, event


def test_reset_and_same_final_status_do_not_resurrect_old_validation(facts):
    f = facts
    graph = f.graph
    audits = list(f.audits)
    # Each status change is accepted by the actual 03 plan/apply boundary, not a
    # rewritten graph dict. This is a synthetic history, never medical reasoning.
    for before, after, reset_to in [
        ("SUPPORTED", "PENDING", "EXCLUDED"),
        ("PENDING", "EXCLUDED", None),
        ("EXCLUDED", "PENDING", "SUPPORTED"),
        ("PENDING", "SUPPORTED", None),
    ]:
        graph, event = transition(f, graph, f.target, before, after, reset_to=reset_to)
        audits.append(event)
        f.snapshots[graph.graph_version] = graph
        assert accepted_validations(graph, f.events, audits, f.read) == []
    assert next(n for n in graph.nodes if n.node_code == f.target).status.value == "SUPPORTED"
    assert accepted_validations(f.graph, f.events, f.audits, f.read) == f.validations


@pytest.mark.parametrize("field", ["base_checksum", "result_checksum"])
def test_rehashed_accepted_package_cannot_replace_runtime_snapshot(facts, field):
    f = facts
    audits = copy.deepcopy(f.audits)
    index = next(i for i, e in enumerate(audits) if e.payload.get("validation_checksum"))
    payload = copy.deepcopy(audits[index].payload)
    event = payload["accepted_delta_event"]
    event[field] = "sha256:" + "a" * 64
    event["event_checksum"] = canonical_checksum(
        {k: v for k, v in event.items() if k != "event_checksum"}
    )
    audits[index] = audits[index].model_copy(update={"payload": payload})
    # Recomputed outer hashes cannot grant authority. A canonical parser refusal
    # and an empty accepted projection are both safe outcomes at this boundary.
    try:
        accepted = accepted_validations(f.graph, f.events, audits, f.read)
    except GraphEngineError:
        accepted = []
    assert accepted == []
