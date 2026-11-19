"""Real isolated PG and harmless offline tool assembly; standard tests have no real model."""

from types import SimpleNamespace

import pytest

from apps.ophthalmology_disease_packs import DiseasePackService
from db import manage
from db.audit.repository import PostgresAuditReader, PostgresAuditWriter
from db.runtime.repository.case_pipeline import CasePipelineRepository
from db.runtime.repository.grounding import RuntimeGroundingLocks
from db.runtime.repository.grounding_query import GroundingQueryService
from db.runtime.repository.tool_execution import RuntimeToolExecutions
from packages.contracts import CaseBundle, VersionedReference
from packages.deterministic_pipeline.models import PipelineConfig, PipelinePolicy
from packages.deterministic_pipeline.service import DeterministicCasePipeline
from packages.deterministic_pipeline.toolchain import FormalToolchain
from packages.deterministic_pipeline.validation import DeterministicValidator
from packages.disease_packs import DiseasePackCatalog
from packages.grounding.models import digest, seal, validate
from packages.object_store import LocalObjectStore
from packages.tool_admission import AdmissionSelection
from packages.tool_execution import ToolExecutionService
from tests.disease_packs.support import anonymous_case, code, config, context
from tests.grounding.query_support import NOW, metadata, request_for, updated
from tests.grounding.test_database import database as database
from tests.grounding.test_query_runtime import scoped_mapping
from tests.grounding.test_runtime import integrated as integrated
from tests.tool_admission.support import Owners, register_deployments, register_inputs, reseal
from tests.tool_execution.support import FIXTURE, prepared, runtime_files


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    import socket

    connect = socket.socket.connect

    def guarded(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError("07B external network forbidden")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture(scope="module")
def closure(tmp_path_factory):
    return runtime_files(tmp_path_factory.mktemp("07b-runtime-closure"))


@pytest.fixture(scope="module")
def mapped(integrated):
    return scoped_mapping(integrated)


@pytest.fixture
def assembled_pipeline(database, mapped, tmp_path, closure, request):
    variant = getattr(request, "param", "success")
    if variant == "shared":
        mapped = request.getfixturevalue("shared_mapped")
    data, mapping, writer, catalogs = mapped
    mapping_refs = tuple(getattr(writer, "fixture_mapping_refs", (mapping.reference(),)))
    rt = manage.test_target("runtime")
    manage.bootstrap_audit_principals(
        database.admin,
        manage.test_target("audit_writer").user,
        manage.test_target("audit_reader").user,
    )
    runtime = CasePipelineRepository(rt)
    audit = PostgresAuditWriter(manage.test_target("audit_writer"))
    audit_reader = PostgresAuditReader(manage.test_target("audit_reader"))
    objects = LocalObjectStore(tmp_path / "objects")
    owners = Owners()
    case = anonymous_case(data["pack"].definition.target_node_code)
    scope = dict(
        case_code=case.case_code,
        patient_hash=case.patient_hash,
        encounter_code=case.encounter_code,
        laterality=case.laterality,
    )
    q = request_for(
        mapping,
        writer.policy,
        scope=scope,
        target_node_code=data["pack"].definition.evidence_slots[0].node_code,
        context=context(),
    )
    q = updated(
        q,
        artifacts=[
            metadata(q, p.constraint, code("ASSET").split("_")[-1])
            for p in mapping.candidates[0].inputs
            if p.required
        ],
    )
    q = register_inputs(q, owners, objects)
    # Explicit synthetic contract/type binding, not real SLO adaptation.
    for k, r in list(owners.records.items()):
        owners.records[k] = r.model_copy(
            update={
                "artifact": r.artifact.model_copy(
                    update={"artifact_type_code": "ARTIFACT:SLO_IMAGE"}
                ),
                "type_binding": reseal(r.type_binding, contract_type_code="ARTIFACT:SLO_IMAGE"),
            }
        )
    case = CaseBundle.model_validate_json(
        case.model_copy(
            update={"artifacts": tuple(r.artifact for r in owners.records.values())}
        ).model_dump_json()
    )
    runtime.create_case_bundle(case, code("TASK"))
    locks = RuntimeGroundingLocks(rt)
    grounding = GroundingQueryService(locks, writer)
    catalog = DiseasePackCatalog((data["pack"],), (data["dataset"],))
    store = writer.knowledge.published_service._store
    disease = DiseasePackService(catalog, database.reader, store, synthetic_test_mode=True)
    validator = DeterministicValidator(database.reader, objects, writer.knowledge.published_service)

    class LazyTools:
        reference = VersionedReference(
            code="POLICY:SYNTHETIC_07B_TOOLS",
            version="1.0.0",
            checksum=digest(dict(fixture=FIXTURE, mode=variant)),
        )
        delegate = None
        ready = None
        mode = variant if variant in {"nonzero", "badlabels", "shape"} else "success"

        def query(self, i, graph, lock, artifacts, requirement, c, started):
            nonlocal q
            target = next(
                r.node_ref.code for r in data["requirements"] if r.code == requirement.code
            )
            output = mapping.candidates[0].output
            extra = []
            from packages.grounding.query_models import ArtifactMetadata

            for a in artifacts:
                if (
                    a.asset_code not in owners.records
                    and a.artifact_type_code == data["requirement"].artifact_type_code
                ):
                    extra.append(
                        validate(
                            ArtifactMetadata,
                            dict(
                                metadata(q, output, "DUMMY"),
                                asset_code=a.asset_code,
                                checksum=a.checksum,
                                run_id=a.scope.run_id,
                                qc_status=a.qc_status,
                            ),
                        )
                    )
            current = updated(
                q,
                lock=lock,
                requirement_ref=requirement,
                config=c.grounding,
                target_node_code=target,
                artifacts=(*q.artifacts, *extra),
            )
            if self.delegate is None:
                result = grounding.query(current)
                invocations = register_deployments(
                    current, catalogs.read_snapshot(mapping.catalog_ref), owners, tmp_path
                )
                base = SimpleNamespace(
                    q=current,
                    grounding=grounding,
                    catalogs=catalogs,
                    owners=owners,
                    objects=objects,
                    result=result,
                    service=SimpleNamespace(clock=lambda: NOW),
                    selection=AdmissionSelection(
                        candidate_refs=tuple(x.identity for x in result.candidates),
                        invocations=invocations,
                    ),
                )
                p = prepared(
                    tmp_path / "tool",
                    base=base,
                    repository=RuntimeToolExecutions(rt),
                    files=closure,
                    mode=self.mode,
                    output_artifact_type=data["requirement"].artifact_type_code,
                    mode_by_implementation={
                        result.candidates[0].candidate.implementation_ref.code: "nonzero"
                    }
                    if variant == "fallback"
                    else None,
                )
                self.ready = p

                def executor_for(plan):
                    policy = p.policy.model_copy(
                        update={"allowed_plan_checksums": (plan.checksum,)}
                    )
                    return ToolExecutionService(
                        compiler=p.compiler,
                        catalogs=catalogs,
                        objects=objects,
                        qc=p.qc,
                        repository=RuntimeToolExecutions(rt),
                        policy=policy,
                        workspace=tmp_path / "attempts",
                    )

                def builder(i, graph, lock, artifacts, requirement, c, started):
                    return current

                selection = p.plan.to_bundle_json()
                import json

                from packages.tool_admission import InvocationInput

                invocations = tuple(
                    InvocationInput.model_validate_json(json.dumps(x))
                    for x in json.loads(selection)["context"]["selection"]["invocations"]
                )
                self.delegate = FormalToolchain(
                    grounding=grounding,
                    query_builder=builder,
                    invocations=invocations,
                    admission=p.compiler.admission,
                    compiler=p.compiler,
                    executor_for=executor_for,
                    reference=self.reference,
                )
            self.delegate.query_builder = lambda *args: current
            return self.delegate.query(i, graph, lock, artifacts, requirement, c, started)

        def compile(self, *args):
            return self.delegate.compile(*args)

        def executor_for(self, plan):
            return self.delegate.executor_for(plan)

        action_key = staticmethod(FormalToolchain.action_key)

    tools = LazyTools()
    policy = seal(
        PipelinePolicy, max_actions=4, max_graph_rounds=12, max_seconds=3600, max_cost=100.0
    )
    cfg = seal(
        PipelineConfig,
        pack_ref=data["pack"].reference(),
        catalog_ref=mapping.catalog_ref,
        mapping_refs=mapping_refs,
        context=context(),
        query="OCT",
        retrieval=config(),
        grounding=q.config,
        policy=policy,
        service_refs=(tools.reference, validator.reference),
    )
    service = DeterministicCasePipeline(
        visible_reader=None,
        runtime=runtime,
        audit=audit,
        audit_reader=audit_reader,
        objects=objects,
        disease=disease,
        catalog=catalog,
        locks=locks,
        mappings=writer,
        tools=tools,
        validator=validator,
        configs=(cfg,),
        synthetic_test_mode=True,
    )
    return SimpleNamespace(
        service=service,
        case=case,
        config=cfg,
        tools=tools,
        data=data,
        runtime=runtime,
        database=database,
        objects=objects,
        audit=audit,
        audit_reader=audit_reader,
    )


@pytest.fixture(scope="module")
def shared_mapped(database, tmp_path_factory):
    """Two reviewed synthetic AND slots share the same product; actual Knowledge publishes them."""
    from unittest.mock import patch

    from packages.contracts import EdgeType
    from tests.disease_packs import support
    from tests.grounding import test_runtime
    from tests.grounding.test_runtime import integrated

    original = support.assembled

    def two(*args, **kwargs):
        return original(*args, **kwargs, slot_count=2, edge_type=EdgeType.AND_CHILD)

    base_fixture = test_runtime.fixture

    def unique_mapping():
        mapping, *rest = base_fixture()
        return reseal(mapping, code=code("BINDING")), *rest

    with (
        patch.object(support, "assembled", two),
        patch.object(test_runtime, "fixture", unique_mapping),
    ):
        values = integrated.__wrapped__(database, tmp_path_factory)
    data, mapping, writer, catalogs = scoped_mapping(values)
    refs = [mapping.reference()]
    for requirement in data["requirements"][1:]:
        other = reseal(
            mapping,
            code=code("BINDING"),
            requirement_ref=requirement.reference().model_dump(mode="json"),
        )
        writer.register_draft(other, context(), request_key=code("TASK"))
        writer.machine_review(other.reference(), context(), request_key=code("TASK"))
        refs.append(other.reference())
    writer.fixture_mapping_refs = tuple(refs)
    return data, mapping, writer, catalogs
