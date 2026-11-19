"""Pure projections over immutable synthetic 07B facts; no PG, tool or model calls."""

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from db.audit.repository import AuditEventRecord
from packages.contracts import GECEGInstance
from packages.deterministic_pipeline.reporting import accepted_validations, observation_effects
from packages.deterministic_pipeline.service import DeterministicCasePipeline
from packages.graph_engine import (
    accepted_delta_event_from_payload,
    canonical_checksum,
    replay_accepted_deltas,
    replay_artifact_binding,
)
from packages.tool_execution.models import WorkflowResult
from packages.tool_workflow import ExecutionPlan

ROOT = Path(__file__).resolve().parents[2] / "docs/acceptance/07b/synthetic-final"


@pytest.fixture
def facts():
    events = json.loads((ROOT / "trace.json").read_text())
    audits = [
        AuditEventRecord.model_validate_json(json.dumps(e))
        for e in json.loads((ROOT / "audit.json").read_text())
    ]
    graph = GECEGInstance.model_validate_json(
        json.dumps(next(e["graph"] for e in events if e["kind"] == "instantiation"))
    )
    snapshots = {graph.graph_version: graph}
    for event in audits:
        if event.event_type.value != "GRAPH_VERSION_APPENDED":
            continue
        if event.payload.get("payload_version") == "03_ARTIFACT_BINDING_V1":
            graph = replay_artifact_binding(graph, event.payload)
        else:
            graph = replay_accepted_deltas(
                graph, (accepted_delta_event_from_payload(event.payload["accepted_delta_event"]),)
            )
        snapshots[graph.graph_version] = graph
    assert graph.model_dump(mode="json") == json.loads((ROOT / "graph.json").read_text())

    def read(code, version):
        return snapshots[version]

    validations = accepted_validations(graph, events, audits, read)
    assert len(validations) == 1
    workflow = WorkflowResult.model_validate_json(
        json.dumps(next(e["workflow"] for e in events if e["kind"] == "execution"))
    )
    plan = ExecutionPlan.from_bundle_json(
        json.dumps(next(e["bundle"] for e in events if e["kind"] == "plan"))
    )
    return SimpleNamespace(
        graph=graph,
        events=events,
        audits=audits,
        snapshots=snapshots,
        read=read,
        validations=validations,
        step=workflow.steps[0],
        plan=plan,
        target=validations[0]["item"].target_node_code,
    )


def effect(f, **kwargs):
    args = dict(
        graph=f.graph,
        target=f.target,
        step=f.step,
        plan=f.plan,
        validations=f.validations,
        synthetic=True,
    )
    args.update(kwargs)
    return observation_effects(**args)


@pytest.mark.parametrize(
    "field", ["source_snapshot", "item_checksum", "binding_checksum", "payload_version"]
)
def test_unaccepted_or_stale_bound_trace_never_enters_report(facts, field):
    f = facts
    events = copy.deepcopy(f.events)
    bound = next(e["validation"] for e in events if e["kind"] == "validation")
    if field == "source_snapshot":
        bound[field]["graph_version"] -= 1
    else:
        bound[field] = "obsolete"
    assert accepted_validations(f.graph, events, f.audits, f.read) == []
    assert accepted_validations(f.graph, f.events, (), f.read) == []
    # A valid acceptance later than the requested report graph is not current evidence.
    source = f.validations[0]["reference"]["source_snapshot"]["graph_version"]
    assert accepted_validations(f.snapshots[source], f.events, f.audits, f.read) == []


@pytest.mark.parametrize("field", ["case_code", "run_id", "graph_code"])
def test_other_scope_audit_cannot_authorize_report(facts, field):
    f = facts
    audits = [e.model_copy(update={field: (getattr(e, field) or "") + "OTHER"}) for e in f.audits]
    assert accepted_validations(f.graph, f.events, audits, f.read) == []


def test_report_retains_exact_accepted_sources_and_deduplicates_trace(facts):
    f = facts
    assert accepted_validations(f.graph, f.events * 2, f.audits, f.read) == f.validations
    value = effect(f)
    assert value["clinical_effect"] == "SUPPORTS"
    ref = value["evidence_validations"][0]["validations"][0]
    assert ref["validation_checksum"] == canonical_checksum(f.validations[0]["item"])
    assert ref["source_snapshot"]["graph_version"] < f.graph.graph_version
    assert ref["rule_refs"] and ref["audit_event_id"]
    assert ref["accepted_graph"]["graph_version"] == ref["source_snapshot"]["graph_version"] + 1
    assert effect(f, target=f.graph.goal_node_code)["clinical_effect"] == "NONE"


@pytest.mark.parametrize(
    "change",
    [
        "asset_identity",
        "asset_checksum",
        "asset_case",
        "asset_run",
        "step_identity",
        "validation_run",
        "shadow",
        "qc",
        "real_nondecisive",
    ],
)
def test_evidence_identity_and_eligibility_are_not_upgraded(facts, change):
    f = facts
    step, plan, validations, synthetic = f.step, f.plan, f.validations, True
    artifact = step.result.artifacts[0]
    if change.startswith("asset_"):
        if change == "asset_identity":
            artifact = artifact.model_copy(update={"asset_code": artifact.asset_code + "OTHER"})
        elif change == "asset_checksum":
            artifact = artifact.model_copy(update={"checksum": "sha256:" + "0" * 64})
        else:
            field = "case_code" if change == "asset_case" else "run_id"
            artifact = artifact.model_copy(
                update={
                    "scope": artifact.scope.model_copy(
                        update={field: getattr(artifact.scope, field) + "OTHER"}
                    )
                }
            )
        step = step.model_copy(
            update={"result": step.result.model_copy(update={"artifacts": (artifact,)})}
        )
    elif change == "step_identity":
        step = step.model_copy(update={"elapsed_seconds": step.elapsed_seconds + 1})
    elif change == "validation_run":
        item = validations[0]["item"]
        validations = [
            dict(
                validations[0],
                item=item.model_copy(
                    update={
                        "scope": item.scope.model_copy(
                            update={"run_id": item.scope.run_id + "OTHER"}
                        )
                    }
                ),
            )
        ]
    elif change == "shadow":
        plan = SimpleNamespace(document=SimpleNamespace(purpose="shadow"))
    elif change == "qc":
        from packages.contracts import QCStatus

        step = step.model_copy(
            update={"result": step.result.model_copy(update={"qc_status": QCStatus.FAIL})}
        )
    else:
        synthetic = False
    assert (
        effect(f, step=step, plan=plan, validations=validations, synthetic=synthetic)[
            "clinical_effect"
        ]
        == "NONE"
    )


@pytest.mark.parametrize(
    "effects", [("SUPPORTS", "REFUTES"), ("REFUTES", "SUPPORTS"), ("SUPPORTS", "SUPPORTS")]
)
@pytest.mark.parametrize("artifact_count", [1, 2])
def test_multiple_validation_details_have_no_last_item_aggregation(facts, effects, artifact_count):
    # Projection-only inputs. Acceptance filtering is independently exercised above;
    # these unit values are never submitted as medical rules, Audit or graph facts.
    from packages.contracts import ClinicalEffect

    f = facts
    artifacts = list(f.step.result.artifacts)
    if artifact_count == 2:
        artifacts.append(
            artifacts[0].model_copy(update={"asset_code": artifacts[0].asset_code + "SECOND"})
        )
    step = f.step.model_copy(
        update={"result": f.step.result.model_copy(update={"artifacts": tuple(artifacts)})}
    )
    values = []
    for index, name in enumerate(effects):
        item = f.validations[0]["item"]
        asset = artifacts[index % artifact_count]
        item = item.model_copy(
            update={
                "clinical_effect": ClinicalEffect(name),
                "evidence_bindings": (asset.asset_code,),
                "provenance": item.provenance.model_copy(
                    update={"input_checksums": (canonical_checksum(step), asset.checksum)}
                ),
            }
        )
        values.append(
            dict(
                item=item,
                reference=dict(
                    f.validations[0]["reference"], validation_code=f"VALIDATION:TEST_{index}"
                ),
            )
        )
    result = effect(f, step=step, validations=values)
    assert sum(len(d["validations"]) for d in result["evidence_validations"]) == 2
    expected = effects[0] if artifact_count == 1 and len(set(effects)) == 1 else None
    assert result["clinical_effect"] == expected
    if expected is None:
        assert result["clinical_effect_status"] == "per_evidence_only"
    assert effect(f, step=step, validations=list(reversed(values)))["clinical_effect"] == expected
    if artifact_count == 2:
        partial = effect(f, step=step, validations=values[:1])
        assert partial["clinical_effect"] is None
        assert partial["evidence_validations"][1]["validations"] == []


@pytest.mark.parametrize("offset,expired", [(-1, False), (0, True), (1, True)])
def test_deadline_boundary_uses_original_start(monkeypatch, offset, expired):
    from tests.deterministic_pipeline.test_budget_attribution import freeze_case_clock

    started = datetime(2026, 9, 12, tzinfo=UTC)
    state = SimpleNamespace(started_at=started)
    config = SimpleNamespace(policy=SimpleNamespace(max_seconds=60))
    service = object.__new__(DeterministicCasePipeline)
    freeze_case_clock(monkeypatch, started + timedelta(seconds=60, microseconds=offset))
    assert service._deadline_reached(state, config) is expired
    assert service._deadline(state, config) == started + timedelta(seconds=60)


def test_legacy_report_is_read_without_rewriting_bytes(tmp_path):
    from packages.dataset_adapters.models import ObjectReference
    from packages.object_store import LocalObjectStore, checksum

    raw = (ROOT / "report.json").read_bytes()
    store = LocalObjectStore(tmp_path / "objects")
    stored = store.promote(store.stage(raw), checksum(raw))
    service = object.__new__(DeterministicCasePipeline)
    service.objects = store
    ref = ObjectReference(uri=stored, checksum=checksum(raw), byte_size=len(raw))
    assert service._read(ref)["schema_version"] == "deterministic-case-report/1.0.0"
    assert store.read(ref.uri) == raw
