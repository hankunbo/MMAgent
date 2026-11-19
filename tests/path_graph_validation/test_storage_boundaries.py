"""Independent synthetic review probes; no patient/model/network/production writes."""

from threading import Event

import pytest

from packages.path_graph_validation import PathGraphValidation
from tests.expert_agent.pg_support import runtime_fixture
from tests.expert_agent.support import expert_reply
from tests.expert_agent.test_pg_boundaries import database as database
from tests.path_graph_validation.support import accepted
from tests.path_graph_validation.test_expert import ready


def test_cancel_during_new_binding_failure_is_not_lost(monkeypatch):
    f, h, service, summary, args = ready([expert_reply()], {"required": True})
    signal = Event()
    before_requests = f.budget.used
    original = f.documents.write

    def broken(key, values):
        if key == "review:" + args["event"].code + ":path-graph-binding":
            signal.set()
            raise OSError("synthetic_09b_binding_write_failure")
        return original(key, values)

    with monkeypatch.context() as m:
        m.setattr(f.documents, "write", broken)
        with pytest.raises(OSError, match="synthetic_09b_binding_write_failure"):
            service.review(**args, cancel=signal)
    cancelled_after_failure = h.planner.read(f.binding.run_id).cancelled
    requests_after_failure = f.budget.used
    signal.clear()
    result = service.review(**args)
    print(
        dict(
            cancelled_after_failure=cancelled_after_failure,
            requests_before=before_requests,
            requests_after_failure=requests_after_failure,
            requests_after_retry=f.budget.used,
            retry_status=result["result"]["status"],
        )
    )
    assert cancelled_after_failure, "Observed cancellation lost before delegated ResearchReview.run"
    assert f.budget.used == requests_after_failure, "Cancelled retry must not issue a new request"


def test_cross_scope_report_rejected_before_pg_bundle_write(database, tmp_path, monkeypatch):
    pg = runtime_fixture(tmp_path)
    other = accepted()
    scope = other.parent.material.scope
    assert (scope.case_code, scope.run_id) != (pg.binding.case_code, pg.binding.run_id)
    # Existing Runtime owner has enough information to reject this wrong assembly.
    with pytest.raises(ValueError, match="agent_runtime_scope"):
        pg.documents.validate_scope(scope)
    original = pg.repo.get_case_bundle(pg.binding.case_code)
    service = PathGraphValidation(reader=other.reader, documents=pg.documents)
    error = None
    try:
        service.inspect(key="review-cross-scope")
    except ValueError as exc:
        error = str(exc)
    saved = pg.documents.read("path-graph:review-cross-scope")
    after = pg.repo.get_case_bundle(pg.binding.case_code)
    print(
        dict(
            error=error,
            saved_documents=len(saved or ()),
            artifact_delta=len(after.artifacts) - len(original.artifacts),
            payload_matches_destination=False
            if saved
            and (saved[0]["case_code"], saved[0]["run_id"])
            != (pg.binding.case_code, pg.binding.run_id)
            else True,
        )
    )
    assert error is not None, "Foreign case/run report was stored in this Runtime bundle"
    assert saved is None and after == original
    storage_matrix(pg, monkeypatch, database)


def memory_source(binding):
    from packages.path_graph_validation import ResearchPathReader
    from tests.agent_provider.support import review_from_wire, submission
    from tests.expert_agent.support import setup

    f = setup([review_from_wire], binding=binding)
    parent = f.reader.state
    candidate = f.ports.review.call(parent.material.scope, deadline=binding.deadline, round_id="A")
    f.acceptance.submit(submission(f.reader, candidate, f.budget), key="source")
    assert f.acceptance.accept(key="source").status == "accepted"
    return ResearchPathReader(acceptance=f.acceptance, parent=parent, key="source")


def storage_matrix(pg, monkeypatch, database):
    """One owned PG target, multiple rejected owners; no repeated cluster bootstrap."""
    from packages.grounding.models import digest
    from packages.research_runtime.agent_records import RuntimeDocuments
    from tests.database.knowledge_support_04a import sql
    from tests.path_graph_validation.pg_boundaries_support import ready_pg

    for change in (
        {"run_id": pg.binding.run_id + "OTHER"},
        {"case_code": pg.binding.case_code + "OTHER"},
        {"graph": pg.binding.graph.model_copy(update={"code": "ASSET:WRONG_GRAPH"})},
        {},  # Correct parent, but an accepted graph in a different document owner.
    ):
        reader = memory_source(pg.binding.model_copy(update=change))
        target = PathGraphValidation(reader=reader, documents=pg.documents)
        before = pg.repo.get_case_bundle(pg.binding.case_code)
        staged = []
        stage = pg.store.stage

        def counted(raw, staged=staged, stage=stage):
            staged.append(1)
            return stage(raw)

        with monkeypatch.context() as m:
            m.setattr(pg.store, "stage", counted)
            with pytest.raises(ValueError, match="agent_runtime_(scope|graph)"):
                target.inspect(key="wrong-destination")
        assert not staged
        assert pg.documents.read("path-graph:wrong-destination") is None
        assert pg.repo.get_case_bundle(pg.binding.case_code) == before
        # Read paths must ask the same real destination owner, even for history.
        value = reader().model_dump(mode="json")
        from packages.path_graph_validation import render_path_graph_html

        saved = (
            value,
            dict(
                schema_version="path-graph-html/1.0.0",
                summary_checksum=value["checksum"],
                html=render_path_graph_html(value),
            ),
        )
        with monkeypatch.context() as m:
            m.setattr(pg.documents, "read", lambda key, saved=saved: saved)
            for historical in (False, True):
                with pytest.raises(ValueError, match="agent_runtime_(scope|graph)"):
                    target.read(key="wrong-destination", historical=historical)

    # A synthetic legacy bad bundle proves historical=True cannot excuse bad ownership.
    foreign = accepted()
    saved = foreign.service.inspect(key="foreign")
    with pg.documents.session():
        pg.documents.write(
            "path-graph:legacy-foreign", foreign.f.documents.read("path-graph:foreign")
        )
    validator = PathGraphValidation(reader=foreign.reader, documents=pg.documents)
    before = pg.documents.read("path-graph:legacy-foreign")
    assert before[0] == saved
    for historical in (False, True):
        with pytest.raises(ValueError, match="agent_runtime_scope"):
            validator.read(key="legacy-foreign", historical=historical)
    assert pg.documents.read("path-graph:legacy-foreign") == before

    f, h, service, summary, args = ready_pg(pg)
    before = pg.repo.get_case_bundle(pg.binding.case_code)
    with pytest.raises(ValueError, match="agent_runtime_scope"):
        service.review(**{**args, "key": "legacy-foreign"})
    assert pg.repo.get_case_bundle(pg.binding.case_code) == before and f.budget.used == 1
    clone = RuntimeDocuments(
        repo=pg.repo,
        contexts=pg.documents.contexts,
        sessions=pg.documents.sessions,
        store=pg.store,
        run_id=pg.binding.run_id,
        root=pg.root,
    )
    service.documents = clone  # Same real owner through a different Python instance is legal.
    assert service.inspect(key="summary") == summary
    assert service.read(key="summary") == summary
    assert service.read(key="summary", historical=True) == summary
    original = pg.repo.get_case_bundle(pg.binding.case_code)
    # A real registered alternate root in the same case is not the summary's lineage.
    alternate = pg.root.model_copy(update={"asset_code": "ASSET:OID_OTHER_ROOT"})
    pg.repo.register_observation_bundle((alternate,), "TASK:OID_OTHER_ROOT")
    wrong_root = RuntimeDocuments(
        repo=pg.repo,
        contexts=pg.documents.contexts,
        sessions=pg.documents.sessions,
        store=pg.store,
        run_id=pg.binding.run_id,
        root=alternate,
    )
    wrong = PathGraphValidation(reader=service.reader, documents=wrong_root)
    after_root = pg.repo.get_case_bundle(pg.binding.case_code)
    for call in (
        lambda: wrong.inspect(key="wrong-root"),
        lambda: wrong.read(key="summary"),
        lambda: wrong.read(key="summary", historical=True),
        lambda: wrong.review(**args),
    ):
        with pytest.raises(ValueError, match="agent_(runtime_graph_lineage|document_scope)"):
            call()
        assert pg.repo.get_case_bundle(pg.binding.case_code) == after_root
    assert len(after_root.artifacts) == len(original.artifacts) + 1
    result = service.review(**args)
    assert result["result"]["status"] == "accept"
    audits = sql(database.admin, "SELECT count(*) FROM audit.event")[0][0]
    assert service.review(**args) == result and f.budget.used == 2
    assert sql(database.admin, "SELECT count(*) FROM audit.event")[0][0] == audits
    saved_bytes = digest(pg.documents.read("path-graph:summary"))
    f.reader.withdrawn = True
    with pytest.raises(ValueError, match="source_withdrawn"):
        service.read(key="summary")
    assert service.read(key="summary", historical=True) == summary
    assert digest(pg.documents.read("path-graph:summary")) == saved_bytes
    formal_storage_matrix(pg, monkeypatch)


def formal_storage_matrix(pg, monkeypatch):
    """Original SQL forbids attaching a formal snapshot to a research run."""
    from psycopg.errors import CheckViolation

    from db.runtime.repository.ports import NotFoundError
    from packages.contracts import GECEGInstance, GraphNodeStatus
    from tests.graph.test_rule_engine_03c import _gate_graph
    from tests.path_graph_validation.test_formal import reader_for

    graph, gate = _gate_graph("OR", (GraphNodeStatus.SUPPORTED, GraphNodeStatus.REFUTED))
    raw = graph.model_dump(mode="json")
    raw["case_code"] = raw["scope"]["case_code"] = pg.binding.case_code
    raw["scope"]["run_id"] = pg.binding.run_id
    graph = GECEGInstance.model_validate(raw)
    before = pg.repo.get_case_bundle(pg.binding.case_code)
    with pytest.raises(CheckViolation, match="clinical_context_required"):
        pg.repo.create_graph_snapshot(graph, "TASK:OID_FORMAL_STORAGE_FIXTURE")
    with pytest.raises(NotFoundError):
        pg.repo.get_graph_snapshot(graph.graph_code, graph.graph_version)
    service = PathGraphValidation(reader=reader_for(graph, (gate,))[0], documents=pg.documents)
    with pytest.raises(ValueError, match="agent_runtime_graph"):
        service.inspect(key="formal-wrong-run")
    assert pg.documents.read("path-graph:formal-wrong-run") is None
    assert pg.repo.get_case_bundle(pg.binding.case_code) == before


@pytest.mark.parametrize("operator", ["AND", "OR"])
def test_formal_synthetic_reports_bind_the_domain_snapshot_owner(monkeypatch, operator):
    """Explicit memory formal-domain writer; never a research Artifact or live PG graph."""
    from packages.contracts import GraphNodeStatus
    from packages.grounding.models import digest
    from packages.path_graph_validation.research import seal_summary
    from packages.rule_engine import snapshot_binding_for
    from tests.agent_provider.support import Documents
    from tests.graph.test_rule_engine_03c import _gate_graph
    from tests.path_graph_validation.test_formal import reader_for

    graph, gate = _gate_graph(operator, (GraphNodeStatus.SUPPORTED, GraphNodeStatus.SUPPORTED))

    class SnapshotDocuments(Documents):
        # The installed domain owner holds the authoritative snapshot independently
        # of both the reader and the caller's report. No Artifact cast or source gate.
        owned = graph

        def validate_snapshot(self, *, case_code, run_id, snapshot):
            if (case_code, run_id) != (self.owned.case_code, self.owned.scope.run_id) or (
                snapshot_binding_for(self.owned) != snapshot
            ):
                raise ValueError("synthetic_snapshot_storage_binding")

    documents = SnapshotDocuments()
    reader = reader_for(graph, (gate,))[0]
    service = PathGraphValidation(reader=reader, documents=documents)
    summary = service.inspect(key="formal")
    assert summary["domain"] == "formal_synthetic"
    assert summary["goals"][0]["derived_status"] == "SUPPORTED"
    assert service.read(key="formal") == service.inspect(key="formal") == summary
    original = digest(documents.saved)
    assert documents.audit.count("path-graph:formal") == 1
    for field, value in (
        ("graph_code", "GRAPH:ABSENT"),
        ("graph_version", 99),
        ("graph_checksum", "sha256:" + "0" * 64),
    ):
        data = reader().model_dump(mode="python", exclude={"checksum", "schema_version"})
        data["graph"] = {**data["graph"], field: value}
        forged = seal_summary(**data)
        wrong = PathGraphValidation(reader=lambda forged=forged: forged, documents=documents)
        with pytest.raises(ValueError, match="synthetic_snapshot_storage_binding"):
            wrong.inspect(key="formal-wrong")
        assert digest(documents.saved) == original
    for change in (dict(case_code="CASE:OTHER"), dict(run_id="RUN:OTHER")):
        with monkeypatch.context() as m:
            other_scope = graph.scope.model_copy(update=change)
            other = graph.model_copy(
                update={"scope": other_scope, "case_code": other_scope.case_code}
            )
            m.setattr(documents, "owned", other)
            with pytest.raises(ValueError, match="synthetic_snapshot_storage_binding"):
                service.inspect(key="formal-other-owner")
            for historical in (False, True):
                with pytest.raises(ValueError, match="synthetic_snapshot_storage_binding"):
                    service.read(key="formal", historical=historical)
        assert digest(documents.saved) == original

    def withdrawn(*args):
        raise ValueError("source_withdrawn")

    reader.validate_sources = withdrawn
    with pytest.raises(ValueError, match="source_withdrawn"):
        service.read(key="formal")
    assert service.read(key="formal", historical=True) == summary
    assert digest(documents.saved) == original
