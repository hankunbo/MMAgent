from dataclasses import replace

import pytest

from apps.ophthalmology_disease_packs import (
    DiseasePackReplayService,
    DiseasePackService,
    build_rule_package,
)
from db.knowledge.repository import KnowledgeConflictError, KnowledgeNotFoundError
from packages.contracts import (
    CapabilityAvailability,
    DataAvailability,
    EdgeType,
    ImmutablePolicyReference,
    VersionedReference,
)
from packages.disease_packs import DiseasePack, DiseasePackCatalog, PackDefinition, seal
from packages.disease_packs.gaps import missing_capability
from packages.graph_engine import (
    Applicability,
    InstantiationContext,
    NodeResolution,
    apply_graph_delta,
    canonical_checksum,
    plan_graph_delta,
    replay_accepted_deltas,
)
from packages.object_store import LocalObjectStore
from packages.rule_engine import (
    PropagationBudget,
    PropagationContext,
    bind_capability_gap,
    compute_rule_package_checksum,
    propagate,
    snapshot_binding_for,
    validate_rule_package,
)
from tests.database.knowledge_support_04a import free_port, prepare_database
from tests.disease_packs.support import anonymous_case, assembled, code, config, context


def test_complete_context_legal_admission(database, tmp_path, monkeypatch):
    from apps.knowledge_retrieval import PublishedKnowledgeService
    from packages.retrieval.knowledge.scope import ClinicalScope
    from tests.disease_packs import support

    original = support.make_claim

    def scoped_claim(*args, **kwargs):
        kwargs["scope"] = ClinicalScope(age_min=18, age_max=65, sexes=("female",))
        return original(*args, **kwargs)

    monkeypatch.setattr(support, "make_claim", scoped_claim)
    store = LocalObjectStore(tmp_path / "objects")
    data = support.assembled(database, store)
    request = context().model_copy(update={"age_years": 30, "sexes": ("female",)})
    found = PublishedKnowledgeService(database.reader, store).retrieve_published_knowledge(
        data["release"].reference(), "OCT", request, config()
    )
    assert set(data["pack"].bindings.artifacts()) <= {
        h["artifact"].record.reference() for h in found["hits"]
    }
    catalog = DiseasePackCatalog((data["pack"],), (data["dataset"],))
    service = DiseasePackService(catalog, database.reader, store, synthetic_test_mode=True)
    admission = service.admit(data["pack"].reference(), "OCT", request, config())
    assert admission.knowledge_gap is None
    for ref in (data["requirement"].reference(), data["requirement"].node_ref):
        assert (
            database.reader.get_published_evidence(ref, context=request).record.reference() == ref
        )
    graph_and_runtime(database, data, service, request)


@pytest.fixture(scope="module")
def database(database_lease):
    with database_lease.cluster(free_port()):
        yield prepare_database()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    connect = socket.socket.connect

    def guarded(sock, address):
        if isinstance(address, tuple):
            assert address[0] in ("127.0.0.1", "::1", "localhost")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture
def setup(database, tmp_path, request):
    store = LocalObjectStore(tmp_path / "objects")
    connected = getattr(request.node, "callspec", None)
    connected = connected is not None and connected.params.get("subject") == "new_review"
    data = assembled(database, store, connected=connected)
    catalog = DiseasePackCatalog((data["pack"],), (data["dataset"],))
    service = DiseasePackService(catalog, database.reader, store, synthetic_test_mode=True)
    return data, store, catalog, service


def graph_and_runtime(database, data, service, retrieval_context=None):
    from db import manage
    from db.runtime.repository import PostgresRuntimeRepository, RunRecord

    retrieval_context = retrieval_context or context()
    case = anonymous_case(data["pack"].definition.target_node_code)
    run_id = code("RUN")
    slot = data["pack"].definition.evidence_slots[0]
    ctx = InstantiationContext(
        graph_code=code("GRAPH"),
        graph_version=1,
        template_checksum=canonical_checksum(data["template"].graph_template),
        created_at=data["created_at"],
        generator_version="04C.synthetic",
        source_codes=(data["claim"].code,),
        resolutions=(
            NodeResolution(
                slot.node_code,
                Applicability.APPLICABLE,
                DataAvailability.UNAVAILABLE,
                CapabilityAvailability.UNAVAILABLE_CAPABILITY,
            ),
        ),
        rule_versions=(data["rule"].reference().to_versioned_reference(),),
        run_id=None,
    )
    admission, instantiated = service.instantiate(
        data["pack"].reference(), case, ctx, "OCT", retrieval_context, config()
    )
    assert admission.knowledge_gap is None
    assert (
        instantiated
        == service.instantiate(
            data["pack"].reference(), case, ctx, "OCT", retrieval_context, config()
        )[1]
    )
    graph = instantiated.instance
    target = manage.test_target("runtime")
    manage.bootstrap_runtime_principal(database.admin, target.user)
    runtime = PostgresRuntimeRepository(target)
    runtime.create_case_bundle(case, code("TASK"))
    runtime.create_graph_snapshot(graph, code("TASK"))
    ref = data["pack"].reference()
    runtime.create_run(
        RunRecord(
            run_id=run_id,
            case_code=case.case_code,
            graph_code=graph.graph_code,
            graph_version=1,
            tool_catalog_ref=VersionedReference(
                code="TOOL:SYNTHETIC", version="1.0.0", checksum="sha256:" + "a" * 64
            ),
            policy_refs=(
                ImmutablePolicyReference(
                    policy_code=ref.code, version=ref.version, checksum=ref.checksum
                ),
            ),
            template_ref=graph.template_ref,
            rule_refs=(data["rule"].reference().to_versioned_reference(),),
        )
    )
    runtime.lock_knowledge_release(run_id, data["release"].reference())
    _, instantiated = service.instantiate(
        data["pack"].reference(),
        case,
        replace(ctx, graph_version=2, run_id=run_id),
        "OCT",
        retrieval_context,
        config(),
    )
    graph = instantiated.instance
    runtime.create_graph_snapshot(graph, code("TASK"))
    return admission, graph, runtime, run_id


def test_published_pack_missing_model_completes_loop_and_replays(database, setup):
    data, store, catalog, service = setup
    admission, graph, runtime, run_id = graph_and_runtime(database, data, service)
    package = build_rule_package(admission, graph)
    slot = data["pack"].definition.evidence_slots[0]
    gap, document = missing_capability(data["pack"], graph, slot)
    assert gap.clinical_effect.value == "NONE"
    assert document.validation_plan.frozen_datasets == ()
    assert document.attempted_candidates == () and document.replay_when_available
    pc = PropagationContext(
        package.package_version,
        package.package_checksum,
        snapshot_binding_for(graph),
        capability_gaps=(bind_capability_gap(graph, gap),),
        budget=PropagationBudget(4, 0, 4, True),
    )
    admitted, result = service.propagate_for_run(
        data["pack"].reference(), runtime, run_id, graph, package, pc, "OCT", context(), config()
    )
    assert admitted.knowledge_gap is None
    assert result.trace.blocked_evidence
    assert not result.evaluation.refutation.contributing_nodes
    # Apply candidate via 03B and finish the loop with a freshly snapshot-bound gap.
    events = []
    original = graph
    for step in range(4):
        if result.candidate_delta is None:
            break
        applied = apply_graph_delta(graph, plan_graph_delta(graph, result.candidate_delta))
        events.append(applied.accepted_event)
        graph = applied.instance
        gap, _ = missing_capability(data["pack"], graph, slot)
        pc = replace(
            pc,
            snapshot_binding=snapshot_binding_for(graph),
            capability_gaps=(bind_capability_gap(graph, gap),),
            budget=PropagationBudget(4, step + 1, 4, True),
        )
        result = propagate(graph, package, pc)
    assert result.stop_suggestion is not None and result.candidate_delta is None
    assert replay_accepted_deltas(original, tuple(events)) == graph
    assert service.validate_conclusion(data["pack"].reference(), "insufficient") == "insufficient"
    with pytest.raises(ValueError):
        service.validate_conclusion(data["pack"].reference(), "definitive_poag_diagnosis")
    history = DiseasePackReplayService(catalog, database.editor, store).replay(
        data["pack"].reference()
    )
    assert history["grants_current_admission"] is False
    assert history["frozen_history"]["checksum"] == admission.manifest_checksum


def test_reviewed_definition_tampering_and_draft_rules_rejected(database, setup):
    data, store, catalog, service = setup
    admission, graph, runtime, run_id = graph_and_runtime(database, data, service)
    package = build_rule_package(admission, graph)
    pc = PropagationContext(
        package.package_version, package.package_checksum, snapshot_binding_for(graph)
    )
    draft = replace(package, release_status="DRAFT")
    draft = replace(draft, package_checksum=compute_rule_package_checksum(draft))
    with pytest.raises(ValueError, match="reviewed"):
        service.propagate_for_run(
            data["pack"].reference(), runtime, run_id, graph, draft, pc, "OCT", context(), config()
        )
    unsafe = propagate(graph, draft, replace(pc, expected_package_checksum=draft.package_checksum))
    assert unsafe.knowledge_gaps and unsafe.stop_suggestion and unsafe.candidate_delta is None
    definition = seal(
        PackDefinition,
        **(
            data["pack"].definition.model_dump(mode="python", exclude={"checksum"})
            | {"target_proposition": "Unreviewed replacement evidence goal"}
        ),
    )
    changed = seal(
        DiseasePack,
        **(
            data["pack"].model_dump(mode="python", exclude={"checksum"})
            | {"definition": definition}
        ),
    )
    changed_catalog = DiseasePackCatalog((changed,), (data["dataset"],))
    changed_service = DiseasePackService(
        changed_catalog, database.reader, store, synthetic_test_mode=True
    )
    with pytest.raises(ValueError, match="exact pack definition"):
        changed_service.admit(changed.reference(), "OCT", context(), config())
    with pytest.raises(ValueError, match="synthetic"):
        DiseasePackService(catalog, database.reader, store).admit(
            data["pack"].reference(), "OCT", context(), config()
        )


@pytest.mark.parametrize("subject", ["review", "source", "new_review"])
def test_live_withdrawal_blocks_old_run_but_history_remains(database, setup, subject):
    data, store, catalog, service = setup
    admission, graph, runtime, run_id = graph_and_runtime(database, data, service)
    replay = DiseasePackReplayService(catalog, database.editor, store)
    before = replay.replay(data["pack"].reference())["frozen_history"]
    if subject == "new_review":
        from tests.database.test_knowledge_release_04b6 import review

        extra = review(database, data["other"], data["policy"])
        middle = replay.replay(data["pack"].reference())["frozen_history"]
        assert middle["material"] == before["material"]
        assert middle["checksum"] == before["checksum"]
        assert (
            service.admit(
                data["pack"].reference(), "OCT", context(), config(), runtime=runtime, run_id=run_id
            ).knowledge_gap
            is None
        )
        ref = extra.reference()
    else:
        ref = data["review"].reference() if subject == "review" else data["claim"].source_refs[0]
    database.publisher.withdraw(ref, "RULE:SYNTHETIC_WITHDRAWAL")
    with pytest.raises((KnowledgeNotFoundError, KnowledgeConflictError)):
        service.admit(
            data["pack"].reference(), "OCT", context(), config(), runtime=runtime, run_id=run_id
        )
    after = replay.replay(data["pack"].reference())["frozen_history"]
    assert before["material"] == after["material"] and before["checksum"] == after["checksum"]
    assert not after["current"]["available"] and after["current"]["unavailable_reasons"]


def test_draft_real_pack_closes_with_knowledge_gap(database, tmp_path):
    from tests.disease_packs.test_declarations_04c import CONFIGS, real_packs

    catalog = DiseasePackCatalog.from_directory(CONFIGS)
    service = DiseasePackService(catalog, database.reader, LocalObjectStore(tmp_path / "objects"))
    for pack in real_packs():
        result = service.admit(pack.reference(), "unused", context(), config())
        assert result.knowledge_gap and result.template is None
        with pytest.raises(ValueError, match="KnowledgeGap"):
            build_rule_package(result, None)


@pytest.mark.parametrize("edge_type", [EdgeType.AND_CHILD, EdgeType.OR_CHILD, EdgeType.FALSIFIES])
def test_only_reviewed_gate_or_falsification_enters_stage03(database, tmp_path, edge_type):
    store = LocalObjectStore(tmp_path / "objects")
    data = assembled(database, store, edge_type=edge_type)
    catalog = DiseasePackCatalog((data["pack"],), (data["dataset"],))
    service = DiseasePackService(catalog, database.reader, store, synthetic_test_mode=True)
    admission, graph, runtime, run_id = graph_and_runtime(database, data, service)
    package = build_rule_package(admission, graph)
    validate_rule_package(
        graph,
        package,
        expected_version=package.package_version,
        expected_checksum=package.package_checksum,
    )
    if edge_type in (EdgeType.AND_CHILD, EdgeType.OR_CHILD):
        assert package.gates[0].operator == edge_type.value.split("_")[0]
    else:
        assert package.gates == ()
    pc = PropagationContext(
        package.package_version, package.package_checksum, snapshot_binding_for(graph)
    )
    _, result = service.propagate_for_run(
        data["pack"].reference(), runtime, run_id, graph, package, pc, "OCT", context(), config()
    )
    assert not result.evaluation.refutation.contributing_nodes  # unavailable never falsifies


def test_new_version_cannot_replace_old_run_or_historical_pack(database, setup):
    old, store, old_catalog, old_service = setup
    _, graph, runtime, run_id = graph_and_runtime(database, old, old_service)
    history = DiseasePackReplayService(old_catalog, database.editor, store).replay(
        old["pack"].reference()
    )
    new = assembled(database, store, version="1.1.0")
    catalog = DiseasePackCatalog((old["pack"], new["pack"]), (old["dataset"], new["dataset"]))
    service = DiseasePackService(catalog, database.reader, store, synthetic_test_mode=True)
    with pytest.raises(ValueError, match="not pinned"):
        service.admit(
            new["pack"].reference(), "OCT", context(), config(), runtime=runtime, run_id=run_id
        )
    assert (
        service.admit(
            old["pack"].reference(), "OCT", context(), config(), runtime=runtime, run_id=run_id
        ).knowledge_gap
        is None
    )
    replay = DiseasePackReplayService(catalog, database.editor, store).replay(
        old["pack"].reference()
    )
    assert replay["pack"] == history["pack"]
    assert replay["frozen_history"]["material"] == history["frozen_history"]["material"]
    assert replay["frozen_history"]["checksum"] == history["frozen_history"]["checksum"]
