"""Limited research use reaches P3 with an immutable lock; anonymous isolated fixtures."""

import pytest

from apps.ophthalmology_disease_packs import DiseasePackService
from packages.deterministic_pipeline.models import PipelineConfig
from packages.disease_packs import DiseasePackCatalog
from packages.grounding.models import seal
from tests.deterministic_pipeline.test_demand_resolution import counts, unmapped
from tests.deterministic_pipeline.test_pipeline import invoke
from tests.disease_packs.support import anonymous_case, assembled, code
from tests.disease_packs.test_research_pg_b03 import research_context


@pytest.mark.parametrize("withdraw", [False, True])
def test_research_release_pack_runtime_p3_and_rejection(
    assembled_pipeline, monkeypatch, withdraw, record_testsuite_property
):
    p = assembled_pipeline
    legacy_mapping_refs = p.config.mapping_refs
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
    with pytest.raises(ValueError, match="KNO_B03_NEW_CONFIG_REQUIRED"):
        seal(
            PipelineConfig,
            **(
                p.config.model_dump(exclude={"checksum"})
                | {"version": "case-pipeline-config/1.0.0"}
            ),
        )
    with pytest.raises(ValueError, match="KNO_B03_NEW_CONFIG_REQUIRED"):
        seal(
            PipelineConfig,
            **(
                p.config.model_dump(exclude={"checksum"})
                | {
                    "context": __import__(
                        "tests.disease_packs.support", fromlist=["context"]
                    ).context()
                }
            ),
        )
    with pytest.raises(ValueError, match="KNO_B03_DEMAND_ONLY_CONFIG"):
        seal(
            PipelineConfig,
            **(p.config.model_dump(exclude={"checksum"}) | {"mapping_refs": legacy_mapping_refs}),
        )
    p = unmapped(p)

    def forbidden(*args, **kwargs):
        raise AssertionError("research demand fixture must not execute any tool")

    for name in ("query", "compile", "executor_for"):
        monkeypatch.setattr(p.tools, name, forbidden)
    if withdraw:
        original = p.service._planning

        def revoke(state):
            original(state)
            p.database.publisher.withdraw(
                p.data["release"].reference(), "B03 isolated release withdrawal"
            )

        monkeypatch.setattr(p.service, "_planning", revoke)
    report = invoke(p)
    assert report["stop_reason"] == (
        "knowledge_not_admitted" if withdraw else "demands_unresolved"
    ), report["gaps"]
    assert report["run_id"] and report["actions"] == 0 and report["observations"] == []
    assert (
        report["conclusion_status"] == "insufficient" and report["clinical_level"] == "pending_L4"
    )
    assert not report["real_medical_graph_complete"]
    lock = p.runtime.get_knowledge_lock(report["run_id"])
    assert lock["use"] == c.use.model_dump(mode="json")
    assert lock["pack_ref"] == pack.reference().model_dump(mode="json")
    assert report["knowledge_use"] == lock["use"]
    assert all(n["status"] not in ("SUPPORTED", "REFUTED") for n in report["graph"]["nodes"])
    assert p.runtime.run_status(report["run_id"]) == "COMPLETED_INSUFFICIENT_EVIDENCE"
    before = counts(p)
    assert invoke(p) == report and counts(p) == before
    assert p.service.replay(report["run_id"]).model_dump(mode="json") == report["graph"]
    record_testsuite_property(
        "b03_p3_report_withdraw_" + str(withdraw), __import__("json").dumps(report)
    )
    # The old lock API cannot reuse a research lock as generic permission.
    if not withdraw:
        with pytest.raises(Exception, match="KNO_B03_USE_REQUIRED"):
            p.runtime.lock_knowledge_release(report["run_id"], p.data["release"].reference())
        bad = c.model_copy(update={"use": c.use.model_copy(update={"purpose": "clinical"})})
        from db.knowledge.repository import KnowledgeConflictError

        with pytest.raises(KnowledgeConflictError):
            p.service.disease.admit(
                pack.reference(),
                "OCT",
                bad,
                p.config.retrieval,
                runtime=p.runtime,
                run_id=report["run_id"],
            )


def test_legacy_runtime_lock_cannot_be_upgraded_to_research(assembled_pipeline):
    import psycopg
    from psycopg.types.json import Jsonb

    p = unmapped(assembled_pipeline)
    report = invoke(p)
    original = p.runtime.get_knowledge_lock(report["run_id"])
    assert "use" not in original and "pack_ref" not in original
    data = assembled(p.database, p.service.disease._published._store, research=True)
    c = research_context(data)
    with p.runtime._connect() as conn:
        with pytest.raises(psycopg.errors.CheckViolation, match="KNO_B03_OLD_LOCK_DENIED"):
            conn.execute(
                "SELECT runtime.lock_research_release(%s,%s,%s,%s)",
                (
                    report["run_id"],
                    Jsonb(data["release"].reference().model_dump(mode="json")),
                    Jsonb(c.use.model_dump(mode="json")),
                    Jsonb(data["pack"].reference().model_dump(mode="json")),
                ),
            )
    assert p.runtime.get_knowledge_lock(report["run_id"]) == original
