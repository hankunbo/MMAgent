"""B03 isolated PG software fixtures. 模拟真实内容模式不是医学发布。"""

import pytest

from apps.knowledge_retrieval import PublishedKnowledgeService
from apps.ophthalmology_disease_packs import DiseasePackReplayService, DiseasePackService
from db.knowledge.repository import KnowledgeConflictError, KnowledgeNotFoundError
from packages.disease_packs import DiseasePackCatalog, ResearchDiseasePack, seal
from packages.graph_engine import canonical_checksum
from packages.knowledge_review.research import KnowledgeUseRequest
from packages.object_store import LocalObjectStore
from tests.disease_packs.support import assembled, config, context
from tests.disease_packs.test_runtime_04c import database as database


def research_context(data):
    from packages.retrieval.knowledge.research import ResearchRetrievalContext

    return ResearchRetrievalContext(
        **context().model_dump(),
        use=KnowledgeUseRequest(
            purpose="limited_research_evidence_planning",
            policy_ref=data["policy"].reference().to_versioned_reference(),
            scope_checksum=canonical_checksum(data["pack"].research_scope),
        ),
    )


@pytest.mark.parametrize("simulated_real", [False, True])
def test_research_published_current_admission_and_refusals(database, tmp_path, simulated_real):
    store = LocalObjectStore(tmp_path / "objects")
    data = assembled(database, store, research=True, simulated_real=simulated_real)
    pack = data["pack"]
    catalog = DiseasePackCatalog((pack,), (data["dataset"],))
    service = DiseasePackService(
        catalog, database.reader, store, synthetic_test_mode=not simulated_real
    )
    c = research_context(data)
    a = service.admit_research(pack.reference(), "OCT", c, config())
    assert a.knowledge_gap is None and pack.clinical_level == "pending_L4"
    assert data["claim"].candidate.provenance.configuration.version == "claim-config/3.0.0"
    published = PublishedKnowledgeService(database.reader, store)
    for bad in (
        context(),
        c.model_copy(update={"use": c.use.model_copy(update={"purpose": "clinical"})}),
        c.model_copy(update={"age_years": 18}),
        c.model_copy(
            update={"use": c.use.model_copy(update={"scope_checksum": "sha256:" + "f" * 64})}
        ),
    ):
        with pytest.raises(KnowledgeConflictError):
            published.retrieve_published_knowledge(
                data["release"].reference(), "OCT", bad, config()
            )
        with pytest.raises((KnowledgeConflictError, KnowledgeNotFoundError)):
            database.reader.get_published_evidence(data["requirement"].reference(), context=bad)
    with pytest.raises(KnowledgeConflictError, match="CLINICAL_DENIED"):
        service.admit_clinical(pack.reference(), "OCT", c, config())
    from db.knowledge.models import ApplicabilityContext

    with pytest.raises(KnowledgeNotFoundError):
        database.reader.get_published(
            data["claim"].reference(),
            applicability=ApplicabilityContext(
                population=context().population,
                conditions=context().conditions,
                modalities=context().modalities,
            ),
        )
    # Rehash caller scope/pack cannot change the Release's reviewer-approved scope.
    modified = pack.research_scope.model_copy(
        update={
            "context_checksum": canonical_checksum(
                context().model_dump(mode="json") | {"age_years": 18}
            )
        }
    )
    impostor = seal(
        ResearchDiseasePack,
        **(pack.model_dump(exclude={"checksum"}) | {"research_scope": modified}),
    )
    forged = c.model_copy(
        update={
            "age_years": 18,
            "use": c.use.model_copy(update={"scope_checksum": canonical_checksum(modified)}),
        }
    )
    altered = DiseasePackService(
        DiseasePackCatalog((impostor,), (data["dataset"],)),
        database.reader,
        store,
        synthetic_test_mode=not simulated_real,
    )
    with pytest.raises(KnowledgeConflictError):
        altered.admit_research(impostor.reference(), "OCT", forged, config())
    # A rehashed v1 declaration with self-reported L4 still cannot consume a research Release.
    from packages.disease_packs import DiseasePack

    legacy = seal(
        DiseasePack,
        definition=pack.definition,
        status="Published",
        clinical_level="L4" if simulated_real else "synthetic_only",
        bindings=pack.bindings,
        knowledge_gaps=(),
    )
    legacy_service = DiseasePackService(
        DiseasePackCatalog((legacy,), (data["dataset"],)),
        database.reader,
        store,
        synthetic_test_mode=not simulated_real,
    )
    for legacy_context in (context(), c):
        with pytest.raises(KnowledgeConflictError):
            legacy_service.admit(legacy.reference(), "OCT", legacy_context, config())
    replay = DiseasePackReplayService(catalog, database.editor, store)
    before = replay.replay(pack.reference())
    database.publisher.withdraw(
        data["claim"].source_refs[0], "B03 synthetic source withdrawal fixture"
    )
    with pytest.raises(KnowledgeNotFoundError):
        service.admit_research(pack.reference(), "OCT", c, config())
    after = replay.replay(pack.reference())
    assert before["frozen_history"]["material"] == after["frozen_history"]["material"]
    assert not after["grants_current_admission"]


def test_research_review_source_policy_and_direct_sql_gates(database, tmp_path):
    """Fresh Reviewed/Published gates; approvals always come from the isolated reviewer."""
    from psycopg.types.json import Jsonb

    from db.knowledge import models as m
    from db.knowledge.repository.review import bind_artifact
    from tests.database.test_knowledge_release_04b6 import policy, publish, review
    from tests.disease_packs.support import code

    store = LocalObjectStore(tmp_path / "objects")
    data = assembled(database, store, research=True, simulated_real=True)
    p = data["policy"]
    fields = data["release"].model_dump(exclude={"checksum"})
    # Missing source and missing declared use cannot reach Reviewed even with a fixture opinion.
    for field in ("source_refs", "research_scope"):
        payload = fields | {"code": code("SOURCE")}
        payload.pop(field)
        subject = m.seal(m.Release, **payload)
        database.editor.put(subject)
        r = review(database, subject, p, transition=False)
        with pytest.raises(KnowledgeConflictError):
            database.reviewer.review(subject.reference(), r.reference(), p.reference())
    subject = m.seal(m.Release, **(fields | {"code": code("SOURCE")}))
    database.editor.put(subject)
    # A review for a different exact subject is not approval; ordinary publish cannot bypass B6.
    with pytest.raises(KnowledgeConflictError):
        publish(database, store, subject, data["review"], p)
    # Draft research policy template has no frozen approval authority.
    pending = policy(
        database,
        synthetic=False,
        synthetic_only=False,
        baseline_only=False,
        formal_review_policy_frozen=False,
        research_scope=p.research_scope,
    )
    with pytest.raises(KnowledgeConflictError):
        r = review(database, subject, pending)
        publish(database, store, subject, r, pending)
    # Rebinding an artifact to the old synthetic-only policy is not a route to real content.
    old = policy(database)
    with pytest.raises((ValueError, KnowledgeConflictError)):
        bind_artifact(
            database.editor,
            subject,
            ((data["claim"].reference(), data["review"].reference()),),
            old.reference(),
        )
    with database.reader._connect() as conn:
        with pytest.raises(
            __import__("psycopg").errors.CheckViolation, match="KNO_B03_USE_SCOPE_DENIED"
        ):
            conn.execute(
                "SELECT knowledge.retrieve_published_knowledge(%s,NULL,%s,%s,%s)",
                (
                    Jsonb(data["release"].reference().model_dump(mode="json")),
                    Jsonb(context().model_dump(mode="json")),
                    Jsonb(config().model_dump(mode="json")),
                    "OCT",
                ),
            )


def test_new_migrations_repeat_on_owned_database(database):
    from db import manage

    before = {
        s: manage.recorded_migrations(database.migration, s) for s in ("knowledge", "runtime")
    }
    assert "0014" in before["knowledge"] and "0011" in before["runtime"]
    manage.migrate(database.migration)
    assert before == {s: manage.recorded_migrations(database.migration, s) for s in before}
