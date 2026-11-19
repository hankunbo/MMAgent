"""Owned PG software fixture: incomplete research creation is atomic at both entry points."""

import psycopg
import pytest

from apps.ophthalmology_disease_packs import DiseasePackService
from packages.deterministic_pipeline.models import PipelineConfig
from packages.disease_packs import DiseasePackCatalog
from packages.grounding.models import seal
from tests.database.test_runtime_research_arguments_b03 import INCOMPLETE
from tests.deterministic_pipeline.test_demand_resolution import counts, unmapped
from tests.deterministic_pipeline.test_pipeline import invoke
from tests.disease_packs.support import anonymous_case, assembled, code, context
from tests.disease_packs.test_research_pg_b03 import research_context


def test_start_run_parameter_refusals_are_atomic_and_complete_request_succeeds(
    assembled_pipeline, monkeypatch
):
    p = assembled_pipeline
    store = p.service.disease._published._store
    p.data = assembled(p.database, store, research=True)
    pack = p.data["pack"]
    p.case = anonymous_case(pack.definition.target_node_code)
    p.runtime.create_case_bundle(p.case, code("TASK"))
    catalog = DiseasePackCatalog((pack,), (p.data["dataset"],))
    p.service.catalog = catalog
    p.service.disease = DiseasePackService(
        catalog, p.database.reader, store, synthetic_test_mode=True
    )
    c = research_context(p.data)
    p.config = seal(
        PipelineConfig,
        **(
            p.config.model_dump(exclude={"checksum"})
            | {
                "version": "case-pipeline-config/2.0.0",
                "pack_ref": pack.reference(),
                "context": c,
                "mapping_refs": (),
            }
        ),
    )
    p = unmapped(p)
    original_start = p.runtime.start_run
    original_create = p.runtime._create_run
    checked = []

    def checked_start(key, old, graph, run, release, **kwargs):
        before = counts(p)
        assert before[:4] == (0, 0, 0, 0)
        assert p.runtime.read_request(key) == old

        def unchanged():
            assert counts(p) == before  # graph/run/Knowledge/Grounding/Audit
            assert p.runtime.read_request(key) == old  # entire request, not just run_id
            with pytest.raises(
                psycopg.errors.NoDataFound, match="run has no Knowledge release lock"
            ):
                p.runtime.get_knowledge_lock(run.run_id)

        # Preserve the real resolution and valid RUN:OID_ identity. Missing release
        # must not turn into an AttributeError while comparing that resolution.
        for has_release, has_use, has_pack in (
            (False, True, False),
            (False, False, True),
            (False, True, True),
            (True, True, False),
            (True, False, True),
        ):
            arguments = kwargs | {
                "context": c if has_use else context(),
                "research_pack": pack.reference() if has_pack else None,
            }
            with pytest.raises(ValueError, match="^" + INCOMPLETE + "$"):
                original_start(key, old, graph, run, release if has_release else None, **arguments)
            unchanged()
            checked.append((has_release, has_use, has_pack))

        statements = []

        class TracedConnection:
            def __init__(self, connection):
                self.connection = connection

            def execute(self, statement, *args, **kwargs):
                statements.append(statement)
                return self.connection.execute(statement, *args, **kwargs)

        def lose_pack_inside_transaction(connection, pending_run, **arguments):
            # Prove the graph was actually inserted within the existing transaction,
            # then exercise the common guard independently of the outer check.
            assert (
                connection.execute(
                    "SELECT count(*) AS n FROM runtime.graph_snapshot WHERE graph_code=%s",
                    (graph.graph_code,),
                ).fetchone()["n"]
                == 1
            )
            arguments["research_pack"] = None
            return original_create(TracedConnection(connection), pending_run, **arguments)

        with monkeypatch.context() as patch:
            patch.setattr(p.runtime, "_create_run", lose_pack_inside_transaction)
            with pytest.raises(ValueError, match="^" + INCOMPLETE + "$"):
                original_start(key, old, graph, run, release, **kwargs)
        assert statements == []  # common guard rejects before run INSERT or lock SQL
        unchanged()
        # Field completeness is not authorization. Original SQL still owns these
        # exact-use/pack refusals, including research Release through the legacy port.
        bad_use = (
            c.use.model_copy(update={"purpose": "clinical"}),
            c.use.model_copy(
                update={"policy_ref": c.use.policy_ref.model_copy(update={"version": "99.0.0"})}
            ),
            c.use.model_copy(update={"scope_checksum": "sha256:" + "f" * 64}),
        )
        for use in bad_use:
            with pytest.raises(psycopg.errors.CheckViolation, match="KNO_B03_RUN_USE_LOCK"):
                original_start(
                    key,
                    old,
                    graph,
                    run,
                    release,
                    **(kwargs | {"context": c.model_copy(update={"use": use})}),
                )
            unchanged()
        for field, value in (("version", "99.0.0"), ("checksum", "sha256:" + "f" * 64)):
            bad_pack = pack.reference().model_copy(update={field: value})
            with pytest.raises(psycopg.errors.CheckViolation, match="KNO_B03_RUN_USE_LOCK"):
                original_start(
                    key, old, graph, run, release, **(kwargs | {"research_pack": bad_pack})
                )
            unchanged()
        with pytest.raises(psycopg.errors.CheckViolation, match="KNO_B03_USE_REQUIRED"):
            original_start(
                key,
                old,
                graph,
                run,
                release,
                **(kwargs | {"context": context(), "research_pack": None}),
            )
        unchanged()
        # Retry the same request with the complete triple through the real P3 path.
        return original_start(key, old, graph, run, release, **kwargs)

    monkeypatch.setattr(p.runtime, "start_run", checked_start)
    report = invoke(p)
    assert len(checked) == 5
    assert report["stop_reason"] == "demands_unresolved"
    assert report["actions"] == 0 and report["observations"] == []
    assert report["conclusion_status"] == "insufficient"
    assert report["clinical_level"] == "pending_L4"
    lock = p.runtime.get_knowledge_lock(report["run_id"])
    assert lock["use"] == c.use.model_dump(mode="json")
    assert lock["pack_ref"] == pack.reference().model_dump(mode="json")
    before = counts(p)
    assert invoke(p) == report and counts(p) == before
