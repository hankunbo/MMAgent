"""Complete request regression using real PG and synthetic reviewed Knowledge only."""

import pytest
from pydantic import ValidationError

from apps.knowledge_retrieval import PublishedKnowledgeService
from apps.ophthalmology_disease_packs import DiseasePackReplayService, DiseasePackService
from db.knowledge import models as m
from db.knowledge.repository import KnowledgeNotFoundError, KnowledgePermissionError
from packages.disease_packs import DiseasePack, DiseasePackCatalog, KnowledgeBindings, seal
from packages.object_store import LocalObjectStore
from packages.retrieval.knowledge.scope import ClinicalScope
from tests.disease_packs import support
from tests.disease_packs.test_runtime_04c import database as database
from tests.disease_packs.test_runtime_04c import graph_and_runtime
from tests.disease_packs.test_runtime_04c import no_network as no_network

SCOPE = ClinicalScope(
    age_min=18,
    age_max=65,
    sexes=("female",),
    exams=("OCT",),
    times=("baseline",),
    lateralities=("RIGHT",),
    devices=("synthetic-device",),
    targets=("synthetic-target",),
    disease_packs=("synthetic_ophthalmology_evidence",),
)


def request_context(**changes):
    return support.context().model_copy(
        update=dict(
            age_years=30,
            sexes=SCOPE.sexes,
            exams=SCOPE.exams,
            times=SCOPE.times,
            lateralities=SCOPE.lateralities,
            devices=SCOPE.devices,
            targets=SCOPE.targets,
            disease_packs=SCOPE.disease_packs,
        )
        | changes
    )


def service_for(database, store, data, pack=None):
    catalog = DiseasePackCatalog((pack or data["pack"],), (data["dataset"],))
    return DiseasePackService(catalog, database.reader, store, synthetic_test_mode=True)


@pytest.fixture(scope="module")
def scoped(database, tmp_path_factory):
    store = LocalObjectStore(tmp_path_factory.mktemp("context04c") / "objects")
    data = support.assembled(database, store, scope=SCOPE)
    return data, store, service_for(database, store, data)


@pytest.mark.parametrize("age", [18, 30, 65])
def test_full_scope_positive_and_deterministic_instantiation(database, scoped, age):
    data, store, service = scoped
    ctx = request_context(age_years=age)
    found = PublishedKnowledgeService(database.reader, store).retrieve_published_knowledge(
        data["release"].reference(), "OCT", ctx, support.config()
    )
    assert set(data["pack"].bindings.artifacts()) <= {
        h["artifact"].record.reference() for h in found["hits"]
    }
    for ref in (data["requirement"].reference(), data["requirement"].node_ref):
        snapshot = database.reader.get_published_evidence(ref, context=ctx)
        assert snapshot.record.reference() == ref and snapshot.status == m.KnowledgeStatus.PUBLISHED
    admitted, graph, runtime, run_id = graph_and_runtime(database, data, service, ctx)
    assert graph and admitted.knowledge_gap is None
    assert (
        service.admit(
            data["pack"].reference(), "OCT", ctx, support.config(), runtime=runtime, run_id=run_id
        ).knowledge_gap
        is None
    )


@pytest.mark.parametrize(
    "change",
    [
        {"age_years": 17},
        {"age_years": 66},
        {"age_years": None},
        {"sexes": ("male",)},
        {"sexes": ()},
        *[
            {field: value}
            for field in ("exams", "times", "lateralities", "devices", "targets", "disease_packs")
            for value in ((), ("unmatched",))
        ],
    ],
)
def test_full_scope_unknown_or_mismatched_rejected_at_each_read(database, scoped, change):
    data, store, service = scoped
    ctx = request_context(**change)
    found = PublishedKnowledgeService(database.reader, store).retrieve_published_knowledge(
        data["release"].reference(), "OCT", ctx, support.config()
    )
    assert not found["hits"]
    assert service.admit(data["pack"].reference(), "OCT", ctx, support.config()).knowledge_gap
    for ref in (data["requirement"].reference(), data["requirement"].node_ref):
        with pytest.raises(KnowledgeNotFoundError):
            database.reader.get_published_evidence(ref, context=ctx)


def test_narrow_port_identity_validation_and_old_api_fail_closed(database, scoped):
    data, _, _ = scoped
    ref = data["requirement"].reference()
    old = m.ApplicabilityContext(
        **{k: getattr(request_context(), k) for k in m.ApplicabilityContext.model_fields}
    )
    with pytest.raises(KnowledgeNotFoundError, match="structured clinical"):
        database.reader.get_published(ref, applicability=old)
    with pytest.raises(TypeError):
        database.reader.get_published_evidence(ref, context=old)
    with pytest.raises(TypeError):
        database.reader.get_published(ref, applicability=request_context())
    for repo in (database.editor, database.reviewer, database.publisher):
        with pytest.raises(KnowledgePermissionError):
            repo.get_published_evidence(ref, context=request_context())
    with pytest.raises(ValueError, match="EvidenceRequirement or Node"):
        database.reader.get_published_evidence(data["claim"].reference(), context=request_context())
    for update in (
        {"age_years": "30"},
        {"age_years": True},
        {"version": "retrieval-context/2.0.0"},
        {"hidden_patient_identifier": "synthetic"},
    ):
        with pytest.raises(ValidationError):
            database.reader.get_published_evidence(
                ref, context=request_context().model_copy(update=update)
            )
    with pytest.raises(KnowledgeNotFoundError):
        database.reader.get_published_evidence(
            ref.model_copy(update={"checksum": "sha256:" + "0" * 64}), context=request_context()
        )


def test_old_unextended_pack_and_four_field_reads_compatible(database, tmp_path):
    store = LocalObjectStore(tmp_path / "objects")
    data = support.assembled(database, store)
    old = m.ApplicabilityContext(
        **{k: getattr(support.context(), k) for k in m.ApplicabilityContext.model_fields}
    )
    for ref in (data["requirement"].reference(), data["requirement"].node_ref):
        assert database.reader.get_published(
            ref, applicability=old
        ) == database.reader.get_published_evidence(ref, context=support.context())
    graph_and_runtime(database, data, service_for(database, store, data))


@pytest.mark.parametrize("kind", [m.EvidenceRequirement, m.Node])
@pytest.mark.parametrize("minimum", [25, 45])
def test_independent_object_scope_not_inferred_from_template(database, tmp_path, kind, minimum):
    store = LocalObjectStore(tmp_path / "objects")
    from tests.database.test_knowledge_release_04b6 import policy, publish, review

    p = policy(database)
    claim, _, _, _ = support.make_claim(
        database, store, support.config(), scope=SCOPE.model_copy(update={"age_min": minimum})
    )
    cr = review(database, claim, p)
    publish(database, store, claim, cr, p)
    data = support.assembled(
        database, store, scope=SCOPE, release_policy=p, evidence_claims={kind: (claim, cr)}
    )
    service = service_for(database, store, data)
    ctx = request_context()
    found = PublishedKnowledgeService(database.reader, store).retrieve_published_knowledge(
        data["release"].reference(), "OCT", ctx, support.config()
    )
    returned = {h["artifact"].record.reference() for h in found["hits"]}
    if kind is m.Node and minimum == 45:
        # Owned-node scope already narrows its template in the existing SQL closure.
        assert data["template"].reference() not in returned
    else:
        assert set(data["pack"].bindings.artifacts()) <= returned
    if minimum == 45:
        # A narrower requirement is checked after an eligible template; owned-node
        # scope can already exclude the template. Neither path waives the object check.
        assert service.admit(data["pack"].reference(), "OCT", ctx, support.config()).knowledge_gap
        ref = (
            data["requirement"].reference()
            if kind is m.EvidenceRequirement
            else data["requirement"].node_ref
        )
        with pytest.raises(KnowledgeNotFoundError):
            database.reader.get_published_evidence(ref, context=ctx)
    elif kind is m.EvidenceRequirement:
        # Passing the patient's age still cannot waive the reviewed binding equality.
        with pytest.raises(ValueError, match="share the filtered template scope"):
            service.admit(data["pack"].reference(), "OCT", ctx, support.config())
    else:
        assert (
            service.admit(data["pack"].reference(), "OCT", ctx, support.config()).knowledge_gap
            is None
        )


def resealed_pack(data, **binding_changes):
    bindings = KnowledgeBindings.model_validate(
        data["pack"].bindings.model_dump(mode="python") | binding_changes
    )
    return seal(
        DiseasePack,
        **(
            data["pack"].model_dump(mode="python", exclude={"checksum", "bindings"})
            | {"bindings": bindings}
        ),
    )


def test_cross_release_and_cross_template_exact_references_rejected(database, scoped, tmp_path):
    data, store, _ = scoped
    other = support.assembled(database, store, scope=SCOPE)
    foreign_pack = resealed_pack(
        data, evidence_requirement_refs=(other["requirement"].reference(),)
    )
    with pytest.raises(ValueError, match="exact members"):
        service_for(database, store, data, foreign_pack).admit(
            foreign_pack.reference(), "OCT", request_context(), support.config()
        )
    # A requirement can be Published in a new release yet refer to a foreign template.
    mixed = support.assembled(
        database,
        store,
        scope=SCOPE,
        evidence_overrides={m.EvidenceRequirement: {"node_ref": other["requirement"].node_ref}},
        release_policy=other["policy"],
    )
    with pytest.raises(ValueError, match="another template"):
        service_for(database, store, mixed).admit(
            mixed["pack"].reference(), "OCT", request_context(), support.config()
        )


@pytest.mark.parametrize("kind", [m.EvidenceRequirement, m.Node])
def test_draft_and_withdrawn_exact_objects_do_not_get_template_authorization(
    database, tmp_path, kind
):
    store = LocalObjectStore(tmp_path / "objects")
    data = support.assembled(database, store, scope=SCOPE)
    ctx = request_context()
    ref = (
        data["requirement"].reference()
        if kind is m.EvidenceRequirement
        else data["requirement"].node_ref
    )
    original = database.reader.get_published_evidence(ref, context=ctx).record
    changes = {}
    if kind is m.Node:
        template = data["template"]
        draft_template = m.seal(
            m.Template,
            **(
                template.model_dump(mode="python", exclude={"checksum"})
                | {
                    "version": "1.1.0",
                    "supersedes": template.reference(),
                    "graph_template": template.graph_template.model_copy(
                        update={"template_version": "1.1.0"}
                    ),
                }
            ),
        )
        database.editor.put(draft_template)
        changes["template_ref"] = draft_template.reference()
    draft = m.seal(
        kind,
        **(
            original.model_dump(mode="python", exclude={"checksum"})
            | {"version": "1.1.0", "supersedes": ref}
            | changes
        ),
    )
    database.editor.put(draft)
    with pytest.raises(KnowledgeNotFoundError):
        database.reader.get_published_evidence(draft.reference(), context=ctx)
    catalog = DiseasePackCatalog((data["pack"],), (data["dataset"],))
    service = service_for(database, store, data)
    _, _, runtime, run_id = graph_and_runtime(database, data, service, ctx)
    replay = DiseasePackReplayService(catalog, database.editor, store)
    before = replay.replay(data["pack"].reference())["frozen_history"]
    database.publisher.withdraw(ref, "RULE:SYNTHETIC_WITHDRAWAL")
    with pytest.raises(KnowledgeNotFoundError):
        database.reader.get_published_evidence(ref, context=ctx)
    with pytest.raises(KnowledgeNotFoundError):
        service.admit(
            data["pack"].reference(), "OCT", ctx, support.config(), runtime=runtime, run_id=run_id
        )
    after = replay.replay(data["pack"].reference())["frozen_history"]
    assert before["material"] == after["material"] and before["checksum"] == after["checksum"]
    assert not after["current"]["available"]


@pytest.mark.parametrize("subject", ["source", "new_review"])
def test_full_context_live_dependencies_block_locked_run_without_rewriting_history(
    database, tmp_path, subject
):
    from tests.database.test_knowledge_release_04b6 import review

    store = LocalObjectStore(tmp_path / "objects")
    data = support.assembled(database, store, scope=SCOPE, connected=subject == "new_review")
    service = service_for(database, store, data)
    ctx = request_context()
    _, _, runtime, run_id = graph_and_runtime(database, data, service, ctx)
    catalog = DiseasePackCatalog((data["pack"],), (data["dataset"],))
    replay = DiseasePackReplayService(catalog, database.editor, store)
    before = replay.replay(data["pack"].reference())["frozen_history"]
    if subject == "new_review":
        extra = review(database, data["other"], data["policy"])
        middle = replay.replay(data["pack"].reference())["frozen_history"]
        assert middle["material"] == before["material"]
        assert (
            service.admit(
                data["pack"].reference(),
                "OCT",
                ctx,
                support.config(),
                runtime=runtime,
                run_id=run_id,
            ).knowledge_gap
            is None
        )
        ref = extra.reference()
    else:
        ref = data["claim"].source_refs[0]
    database.publisher.withdraw(ref, "RULE:SYNTHETIC_WITHDRAWAL")
    with pytest.raises(KnowledgeNotFoundError):
        database.reader.get_published_evidence(data["requirement"].reference(), context=ctx)
    with pytest.raises(KnowledgeNotFoundError):
        service.admit(
            data["pack"].reference(), "OCT", ctx, support.config(), runtime=runtime, run_id=run_id
        )
    after = replay.replay(data["pack"].reference())["frozen_history"]
    assert before["material"] == after["material"] and before["checksum"] == after["checksum"]
    assert not after["current"]["available"]


def test_candidate_exam_scope_checked_without_optional_clinical_scope(
    database, tmp_path, monkeypatch
):
    from tests.knowledge_retrieval import support as retrieval_support

    original = retrieval_support.output
    monkeypatch.setattr(
        retrieval_support,
        "output",
        lambda *args, **kwargs: original(*args, **kwargs, exam=("OCT",)),
    )
    store = LocalObjectStore(tmp_path / "objects")
    data = support.assembled(database, store)
    assert data["claim"].candidate.exam == ("OCT",)
    for ref in (data["requirement"].reference(), data["requirement"].node_ref):
        assert (
            database.reader.get_published_evidence(
                ref, context=support.context()
            ).record.reference()
            == ref
        )
    # The reviewed Claim declares OCT even when optional ClinicalScope is absent.
    ctx = support.context().model_copy(update={"exams": ()})
    found = PublishedKnowledgeService(database.reader, store).retrieve_published_knowledge(
        data["release"].reference(), "OCT", ctx, support.config()
    )
    assert not found["hits"]
    for ref in (data["requirement"].reference(), data["requirement"].node_ref):
        with pytest.raises(KnowledgeNotFoundError):
            database.reader.get_published_evidence(ref, context=ctx)


def test_narrow_sql_port_authentication_and_context_guard(database, scoped):
    import psycopg
    from psycopg.types.json import Jsonb

    from tests.database.knowledge_support_04a import connect

    def sql(repo, statement, params):
        with connect(repo._target) as connection:
            connection.execute(
                psycopg.sql.SQL("SET ROLE {}").format(psycopg.sql.Identifier(repo._role))
            )
            return connection.execute(statement, params).fetchone()[0]

    data, _, _ = scoped
    ref = data["requirement"].reference().model_dump(mode="json")
    ctx = request_context().model_dump(mode="json")
    statement = "SELECT knowledge.read_published_evidence(%s,%s)"
    assert (
        sql(database.reader, statement, (Jsonb(ref), Jsonb(ctx)))["record"]["checksum"]
        == ref["checksum"]
    )
    for repo in (database.editor, database.reviewer, database.publisher):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            sql(repo, statement, (Jsonb(ref), Jsonb(ctx)))
    for bad_ref, bad_ctx in (
        (ref, ctx | {"version": "retrieval-context/2.0.0"}),
        (data["claim"].reference().model_dump(mode="json"), ctx),
    ):
        with pytest.raises(psycopg.errors.CheckViolation):
            sql(database.reader, statement, (Jsonb(bad_ref), Jsonb(bad_ctx)))
    for field in (
        "age_years",
        "sexes",
        "devices",
        "exams",
        "times",
        "lateralities",
        "targets",
        "disease_packs",
    ):
        with pytest.raises(psycopg.errors.NoDataFound):
            sql(
                database.reader,
                statement,
                (Jsonb(ref), Jsonb({k: v for k, v in ctx.items() if k != field})),
            )
