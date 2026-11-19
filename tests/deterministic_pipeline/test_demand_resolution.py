"""P3 anonymous fixtures, real Published admission and isolated Runtime transactions."""

import pytest

from packages.deterministic_pipeline.demands import DemandResolver
from packages.deterministic_pipeline.models import PipelineConfig
from packages.grounding.models import seal
from tests.deterministic_pipeline.test_pipeline import invoke


def unmapped(p):
    p.service.demand_resolver = DemandResolver(
        knowledge=p.database.reader, catalogs=p.service.mappings.catalogs
    )
    p.config = seal(
        PipelineConfig,
        **(
            p.config.model_dump(exclude={"checksum"})
            | {
                "mapping_refs": (),
                "service_refs": (*p.config.service_refs, p.service.demand_resolver.reference),
            }
        ),
    )
    p.service.configs = {p.config.checksum: p.config}
    return p


def test_published_without_mapping_has_run_and_no_execution(assembled_pipeline):
    p = unmapped(assembled_pipeline)
    # The existing executor is deliberately present: absent Mapping is not absent Tool.
    report = invoke(p, "unresolved")
    assert report["run_id"] is not None
    assert report["actions"] == 0
    assert report["observations"] == []


def counts(p):
    from tests.database.knowledge_support_04a import sql

    return tuple(
        sql(p.database.admin, query, (p.case.case_code,))[0][0]
        for query in (
            "SELECT count(*) FROM runtime.run WHERE case_code=%s",
            "SELECT count(*) FROM runtime.graph_snapshot WHERE case_code=%s",
            "SELECT count(*) FROM runtime.run_knowledge_lock l JOIN runtime.run r USING(run_id) "
            "WHERE r.case_code=%s",
            "SELECT count(*) FROM runtime.run_grounding_lock l JOIN runtime.run r USING(run_id) "
            "WHERE r.case_code=%s",
            "SELECT count(*) FROM audit.event WHERE case_code=%s",
        )
    )


def test_unresolved_retry_replay_immutable_and_zero_execution(assembled_pipeline, monkeypatch):
    import pytest

    from db.runtime.repository.ports import IdempotencyConflictError

    p = unmapped(assembled_pipeline)

    def forbidden(*args, **kwargs):
        raise AssertionError("unresolved branch touched execution or Mapping lock")

    for name in ("query", "compile", "executor_for"):
        monkeypatch.setattr(p.tools, name, forbidden)
    monkeypatch.setattr(p.service.locks, "prepare", forbidden)
    monkeypatch.setattr(p.service.locks, "read", forbidden)
    report = invoke(p)
    assert report["stop_reason"] == "demands_unresolved"
    assert report["demand_resolution"]["state"] == "unresolved"
    assert report["demand_propagations"]
    assert report["observations"] == [] and report["actions"] == 0
    assert counts(p)[0:4:2] == (1, 1)
    assert counts(p)[3] == 0
    assert p.service.replay(report["run_id"]).model_dump(mode="json") == report["graph"]
    before = counts(p)
    state = p.runtime.read_request(p.runtime.key(p.case.case_code, "complete"))
    assert invoke(p) == report
    assert counts(p) == before
    assert (
        p.runtime.read_request(p.runtime.key(p.case.case_code, "complete")).started_at
        == state.started_at
    )
    altered = seal(
        type(state.demand_resolution),
        **(
            state.demand_resolution.model_dump(exclude={"checksum"})
            | {"resolver_ref": p.tools.reference}
        ),
    )
    with pytest.raises(ValueError, match="resolution_immutable"):
        p.runtime.checkpoint(
            p.runtime.key(p.case.case_code, "complete"), state, demand_resolution=altered
        )
    changed = p.case.model_copy(update={"patient_hash": "sha256:" + "d" * 64})
    with pytest.raises(IdempotencyConflictError):
        p.service.run_synthetic(
            changed, pack_ref=p.config.pack_ref, config_checksum=p.config.checksum, key="complete"
        )
    assert counts(p) == before
    assert not list(p.objects.root.parent.glob("attempts/*/*/process.json"))
    from tests.database.knowledge_support_04a import sql

    assert (
        sql(
            p.database.admin,
            "SELECT count(*) FROM runtime.tool_attempt WHERE run_id=%s",
            (report["run_id"],),
        )[0][0]
        == 0
    )
    import json

    print(
        "P3_ACCEPTANCE_JSON:"
        + json.dumps(
            {
                "report": report,
                "trace": p.service.trace(p.case.case_code, "complete"),
                "initial_graph": p.runtime.get_graph_snapshot(
                    report["graph"]["graph_code"], 1
                ).model_dump(mode="json"),
                "audit": [
                    e.model_dump(mode="json")
                    for e in p.audit_reader.list_correlation("SOURCE:DETERMINISTIC_PIPELINE")
                    if e.case_code == p.case.case_code
                ],
                "counts_run_graph_knowledge_grounding_audit": before,
                "tool_attempts": 0,
                "replay_matches": True,
            },
            sort_keys=True,
        )
    )


def test_unresolved_real_sql_audit_conflict_rolls_back_every_run_fact(
    assembled_pipeline, monkeypatch
):
    import pytest

    p = unmapped(assembled_pipeline)
    original = p.runtime._audit

    def conflict(c, key, state, event_type, payload):
        if event_type == "CONFIG_VERSION_LOCKED":
            # Collide with REQUEST_RECEIVED through the real SQL bridge in this transaction.
            state = state.model_copy(update={"revision": 0})
        return original(c, key, state, event_type, payload)

    monkeypatch.setattr(p.runtime, "_audit", conflict)
    from psycopg.errors import UniqueViolation

    with pytest.raises(UniqueViolation):
        invoke(p)
    assert counts(p)[:4] == (0, 0, 0, 0)
    state = p.runtime.read_request(p.runtime.key(p.case.case_code, "complete"))
    assert state.run_id is None and state.demand_resolution is None
    monkeypatch.setattr(p.runtime, "_audit", original)
    report = invoke(p)
    assert report["run_id"] and counts(p)[0] == 1
    assert (
        p.runtime.read_request(p.runtime.key(p.case.case_code, "complete")).started_at
        == state.started_at
    )


@pytest.mark.parametrize("qc", ["UNASSESSED", "PENDING"])
def test_missing_inputs_unknown_scope_and_unassessed_qc_are_separate(assembled_pipeline, qc):
    from packages.contracts import CaseBundle, Laterality, QCStatus
    from tests.disease_packs.support import anonymous_case, code

    p = unmapped(assembled_pipeline)
    source_case = anonymous_case(p.data["pack"].definition.target_node_code)
    source_assets = tuple(
        a.model_copy(
            update={
                "asset_code": code("ASSET"),
                "qc_status": QCStatus(qc),
                "scope": a.scope.model_copy(update={"case_code": source_case.case_code}),
            }
        )
        for a in p.case.artifacts
    )
    source_case = CaseBundle.model_validate_json(
        source_case.model_copy(update={"artifacts": source_assets}).model_dump_json()
    )
    p.runtime.create_case_bundle(source_case, code("TASK"))
    p.case = source_case
    first = invoke(p)
    row = first["demand_resolution"]["demands"][0]
    assert row["input_state"] == "present"
    assert row["availability"] == "UNAVAILABLE"
    assert "input_qc_" + qc.lower() in row["reasons"]
    case = anonymous_case(p.data["pack"].definition.target_node_code)
    case = CaseBundle.model_validate_json(
        case.model_copy(
            update={
                "laterality": Laterality.UNKNOWN,
            }
        ).model_dump_json()
    )
    p.runtime.create_case_bundle(case, code("TASK"))
    p.case = case
    report = invoke(p)
    row = report["demand_resolution"]["demands"][0]
    assert row["input_state"] == "missing" and row["applicability"] == "UNKNOWN"
    assert "patient_input_missing" in row["reasons"] and "applicability_unknown" in row["reasons"]
    assert not any(n["status"] in ("SUPPORTED", "REFUTED") for n in report["graph"]["nodes"])


def test_catalog_failure_is_not_unknown_producer_or_capability_gap(assembled_pipeline):
    p = unmapped(assembled_pipeline)

    class Broken:
        def read_snapshot(self, ref):
            raise OSError("private connection detail must not appear")

    p.service.demand_resolver.catalogs = Broken()
    p.service.tools = None
    report = invoke(p)
    assert report["run_id"]
    assert report["stop_reason"] == "environment_error"
    assert report["demand_resolution"]["catalog_state"] == "unreadable"
    assert "producer_unresolved" not in report["gaps"]
    assert "private connection" not in str(report)
    assert report["product_demands"] == [] and report["tool_suggestions"] == []


def test_catalog_version_conflict_stops_without_execution(assembled_pipeline):
    p = unmapped(assembled_pipeline)
    catalogs = p.service.demand_resolver.catalogs
    snapshot = catalogs.read_snapshot(p.config.catalog_ref)
    changed = seal(
        type(snapshot), **(snapshot.model_dump(exclude={"checksum"}) | {"version": "99.0.0"})
    )

    class WrongVersion:
        def read_snapshot(self, ref):
            return changed

    p.service.demand_resolver.catalogs = WrongVersion()
    report = invoke(p)
    assert report["demand_resolution"]["catalog_state"] == "version_conflict"
    assert report["stop_reason"] == "environment_error" and report["actions"] == 0


def test_unknown_producer_never_invents_capability(assembled_pipeline):
    from packages.grounding.synthetic import SyntheticCatalogReader, fixture

    p = unmapped(assembled_pipeline)
    _, _, catalog, _, _ = fixture()
    p.service.demand_resolver.catalogs = SyntheticCatalogReader([catalog])
    p.config = seal(
        PipelineConfig,
        **(p.config.model_dump(exclude={"checksum"}) | {"catalog_ref": catalog.reference()}),
    )
    p.service.configs = {p.config.checksum: p.config}
    report = invoke(p)
    row = report["demand_resolution"]["demands"][0]
    assert row["producer_refs"] == [] and "producer_unresolved" in row["reasons"]
    assert "capability_code" not in row
    assert report["tool_suggestions"] == []
    assert report["demand_resolution"]["extension_enqueued"] is False


def test_mapped_query_without_executor_does_not_claim_absent_tools(assembled_pipeline):
    p = assembled_pipeline
    p.service.tools = None
    report = invoke(p)
    assert report["run_id"] and report["demand_resolution"]["state"] == "mapped"
    assert "execution_dependency_unavailable" in report["gaps"]
    assert report["actions"] == 0 and report["observations"] == []
    assert counts(p)[3] == 1


def test_withdrawn_mapping_cannot_downgrade(assembled_pipeline):
    import pytest

    from packages.grounding import GroundingError, State
    from tests.disease_packs.support import code

    p = assembled_pipeline
    writer = p.service.mappings
    original = writer.get_exact(p.config.mapping_refs[0]).record
    record = seal(
        type(original), **(original.model_dump(exclude={"checksum"}) | {"code": code("BINDING")})
    )
    writer.register_draft(record, p.config.context, request_key=code("TASK"))
    writer.machine_review(record.reference(), p.config.context, request_key=code("TASK"))
    writer.invalidate(record.reference(), state=State.WITHDRAWN, request_key=code("TASK"))
    p.config = seal(
        PipelineConfig,
        **(p.config.model_dump(exclude={"checksum"}) | {"mapping_refs": (record.reference(),)}),
    )
    p.service.configs = {p.config.checksum: p.config}
    p.service.demand_resolver = DemandResolver(
        knowledge=p.database.reader, catalogs=writer.catalogs
    )
    with pytest.raises(GroundingError):
        invoke(p)
    assert counts(p)[:4] == (0, 0, 0, 0)
    assert (
        p.runtime.read_request(p.runtime.key(p.case.case_code, "complete")).demand_resolution
        is None
    )


def test_report_promotion_failure_preserves_run_and_resumes(assembled_pipeline, monkeypatch):
    import pytest

    p = unmapped(assembled_pipeline)
    put = p.service._put

    def fail(value):
        if (
            isinstance(value, dict)
            and value.get("schema_version") == "deterministic-case-report/2.0.0"
        ):
            put(value)
            raise OSError("injected promotion acknowledgement loss")
        return put(value)

    monkeypatch.setattr(p.service, "_put", fail)
    with pytest.raises(OSError):
        invoke(p)
    before = counts(p)
    monkeypatch.setattr(p.service, "_put", put)
    report = invoke(p)
    assert counts(p)[:4] == before[:4]
    assert report["actions"] == 0
    assert p.service.replay(report["run_id"]).model_dump(mode="json") == report["graph"]


def test_initial_graph_checksum_conflict_is_atomic(assembled_pipeline, monkeypatch):
    p = unmapped(assembled_pipeline)
    start = p.runtime.start_run

    def conflict(*args, **kwargs):
        resolution = kwargs["demand_resolution"]
        kwargs["demand_resolution"] = seal(
            type(resolution),
            **(
                resolution.model_dump(exclude={"checksum"})
                | {"graph_checksum": "sha256:" + "f" * 64}
            ),
        )
        return start(*args, **kwargs)

    monkeypatch.setattr(p.runtime, "start_run", conflict)
    with pytest.raises(ValueError, match="resolution_scope_conflict"):
        invoke(p)
    assert counts(p)[:4] == (0, 0, 0, 0)
    monkeypatch.setattr(p.runtime, "start_run", start)
    report = invoke(p)
    assert report["run_id"] and report["actions"] == 0


@pytest.fixture(scope="module", params=["AND_CHILD", "OR_CHILD"])
def shared_mapped(database, tmp_path_factory, request):
    from unittest.mock import patch

    from packages.contracts import EdgeType
    from tests.deterministic_pipeline.conftest import shared_mapped as base
    from tests.disease_packs import support

    original = support.assembled

    def selected(*args, **kwargs):
        return original(*args, **(kwargs | {"edge_type": EdgeType(request.param)}))

    with patch.object(support, "assembled", selected):
        return base.__wrapped__(database, tmp_path_factory)


@pytest.mark.parametrize("assembled_pipeline", ["shared"], indirect=True)
def test_reviewed_and_or_propagates_without_fake_validation(assembled_pipeline, shared_mapped):
    p = unmapped(assembled_pipeline)
    report = invoke(p)
    assert len(report["demand_resolution"]["demands"]) == 2
    trace = report["demand_propagations"]
    assert trace and trace[-1]["propagation"]["stop_suggestion"]
    gates = trace[-1]["package"]["gates"]
    assert len(gates) == 1 and gates[0]["operator"] in ("AND", "OR")
    events = p.service.trace(p.case.case_code, "complete")
    assert not any(e["kind"] in ("plan", "execution", "validation") for e in events)
    assert p.service.replay(report["run_id"]).model_dump(mode="json") == report["graph"]
    accepted = [
        e
        for e in p.audit_reader.list_correlation("SOURCE:DETERMINISTIC_PIPELINE")
        if e.run_id == report["run_id"] and e.event_type.value == "GRAPH_VERSION_APPENDED"
    ]
    assert len(accepted) == report["graph"]["graph_version"] - 1
    assert accepted or all(
        e["propagation"]["candidate_delta"] is None
        and e["delta_reason"] == "no_status_change_required_by_reviewed_propagation"
        for e in trace
    )
    print(
        {
            "gate": gates[0]["operator"],
            "graph_version": report["graph"]["graph_version"],
            "accepted_events": len(accepted),
            "actions": report["actions"],
            "demands": len(report["demand_resolution"]["demands"]),
        }
    )


def test_missing_dependency_is_explicit_environment_failure(assembled_pipeline):
    p = unmapped(assembled_pipeline)
    p.service.demand_resolver = None
    report = invoke(p)
    assert report["stop_reason"] == "environment_error"
    assert report["run_id"] is None
    assert report["gaps"] == ["demand_resolver_unavailable"]


def test_query_and_execution_are_optional_for_explicit_unresolved_branch(assembled_pipeline):
    p = unmapped(assembled_pipeline)
    p.service.mappings = p.service.tools = p.service.queries = p.service.locks = None
    report = invoke(p)
    assert report["stop_reason"] == "demands_unresolved"
    assert report["demand_resolution"]["demands"][0]["producer_refs"]
    assert report["actions"] == 0


def test_new_mapping_requires_new_config_and_new_request(assembled_pipeline):
    from db.runtime.repository.ports import IdempotencyConflictError

    p = assembled_pipeline
    original = p.config
    unmapped(p)
    stopped = invoke(p)
    state = p.runtime.read_request(p.runtime.key(p.case.case_code, "complete"))
    p.service.configs[original.checksum] = original
    p.config = original
    with pytest.raises(IdempotencyConflictError):
        invoke(p)
    assert p.runtime.read_request(p.runtime.key(p.case.case_code, "complete")) == state
    # New exact request retains Mapping; a separate query port suppresses execution.
    p.service.tools = None
    new = invoke(p, "new-mapped-request")
    assert new["run_id"] != stopped["run_id"]
    assert new["demand_resolution"]["state"] == "mapped"
    assert p.runtime.read_request(p.runtime.key(p.case.case_code, "complete")) == state


def test_unknown_applicability_cannot_be_derivable_or_bind_evidence():
    from dataclasses import replace

    from packages.contracts import DataAvailability, NodeType
    from packages.graph_engine import Applicability, NodeResolution
    from packages.graph_engine.engine import GraphEngineError, _validate_resolution

    resolution = NodeResolution(
        "EVIDENCE:TEST", Applicability.UNKNOWN, DataAvailability.UNAVAILABLE
    )
    _validate_resolution(resolution, NodeType.EVIDENCE, set())
    with pytest.raises(GraphEngineError, match="unknown applicability"):
        _validate_resolution(
            replace(resolution, availability=DataAvailability.DERIVABLE), NodeType.EVIDENCE, set()
        )
