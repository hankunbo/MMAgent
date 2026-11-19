"""Grounding version locks on real Knowledge releases, graph snapshots and existing Runs."""

import json

import psycopg
import pytest

from apps.knowledge_retrieval import PublishedKnowledgeService
from db import manage
from db.grounding.knowledge import PublicKnowledgeReader
from db.grounding.repository import GroundingRepository
from db.runtime.repository import PostgresRuntimeRepository, RunRecord
from db.runtime.repository.grounding import RuntimeGroundingLocks
from packages.contracts import (
    CapabilityAvailability,
    DataAvailability,
    ImmutablePolicyReference,
    VersionedReference,
)
from packages.graph_engine import (
    Applicability,
    InstantiationContext,
    NodeResolution,
    canonical_checksum,
    instantiate_template,
)
from packages.grounding import CatalogSnapshot, GroundingError, Mapping, State, validate
from packages.grounding.synthetic import SyntheticCatalogReader, fixture
from packages.object_store import LocalObjectStore
from tests.database.knowledge_support_04a import connect, sql
from tests.disease_packs import support
from tests.grounding.test_database import database as database
from tests.grounding.test_database import key
from tests.grounding.test_models import reseal


@pytest.fixture(scope="module")
def integrated(database, tmp_path_factory):
    store = LocalObjectStore(tmp_path_factory.mktemp("grounding-knowledge") / "objects")
    data = support.assembled(database, store)
    mapping, policy, catalog, _, raw = fixture()
    # Bind the synthetic manifest shape to the actual synthetic Published requirement type.
    # This remains a derived fixture, never a real medgraph adapter/validation claim.
    raw[0]["ports"][-1]["artifact_code"] = data["requirement"].artifact_type_code
    from packages.grounding import digest

    old_artifact = catalog.artifacts[0]
    new_artifact = {
        **old_artifact.model_dump(mode="json"),
        "code": data["requirement"].artifact_type_code,
    }
    old_tool = catalog.tools[0].tool_ref.model_dump(mode="json")
    new_tool = {**old_tool, "checksum": digest(raw[0])}

    def replace_tree(value):
        if isinstance(value, dict):
            if value == old_artifact.model_dump(mode="json"):
                return new_artifact
            if value == old_tool:
                return new_tool
            return {k: replace_tree(v) for k, v in value.items()}
        if isinstance(value, list):
            return [replace_tree(v) for v in value]
        return value

    catalog = reseal(
        CatalogSnapshot,
        {**replace_tree(catalog.model_dump(mode="json")), "manifest_checksum": digest(raw)},
    )
    fields = {
        name + "_ref": data[name].reference().model_dump(mode="json")
        for name in ("requirement", "release", "template")
    }
    fields.update(
        claim_refs=[data["claim"].reference().model_dump(mode="json")],
        source_refs=[r.model_dump(mode="json") for r in data["requirement"].source_refs],
        applicability_refs=[
            r.model_dump(mode="json") for r in data["requirement"].applicability_refs
        ],
    )
    mapping = reseal(
        Mapping,
        {
            **replace_tree(mapping.model_dump(mode="json")),
            **fields,
            "catalog_ref": catalog.reference().model_dump(mode="json"),
        },
    )
    catalogs = SyntheticCatalogReader([catalog])
    knowledge = PublicKnowledgeReader(
        database.reader,
        database.editor,
        published_service=PublishedKnowledgeService(database.reader, store),
        query="OCT",
        config=support.config(),
    )
    writer = GroundingRepository(
        manage.test_target("grounding_writer"),
        capability="writer",
        policy=policy,
        knowledge=knowledge,
        catalogs=catalogs,
    )
    writer.register_draft(mapping, support.context(), request_key=key())
    writer.machine_review(mapping.reference(), support.context(), request_key=key())
    return data, mapping, writer, catalogs


def create_run(database, data, mapping, writer):
    case = support.anonymous_case(data["pack"].definition.target_node_code)
    slot = data["pack"].definition.evidence_slots[0]
    ctx = InstantiationContext(
        graph_code=support.code("GRAPH"),
        graph_version=1,
        template_checksum=canonical_checksum(data["template"].graph_template),
        created_at=data["created_at"],
        generator_version="05A.synthetic",
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
    graph = instantiate_template(data["template"].graph_template, case, ctx).instance
    target = manage.test_target("runtime")
    runtime = PostgresRuntimeRepository(target)
    runtime.create_case_bundle(case, support.code("TASK"))
    runtime.create_graph_snapshot(graph, support.code("TASK"))
    run_id = support.code("RUN")
    policy = writer.policy
    runtime.create_run(
        RunRecord(
            run_id=run_id,
            case_code=case.case_code,
            graph_code=graph.graph_code,
            graph_version=graph.graph_version,
            tool_catalog_ref=VersionedReference(**mapping.catalog_ref.model_dump(exclude={"kind"})),
            policy_refs=(
                ImmutablePolicyReference(
                    policy_code=policy.code, version=policy.version, checksum=policy.checksum
                ),
            ),
            template_ref=graph.template_ref,
            rule_refs=(data["rule"].reference().to_versioned_reference(),),
        ),
        knowledge_release=data["release"].reference(),
    )
    graph_checksum = sql(
        database.admin,
        "SELECT snapshot_checksum FROM runtime.graph_snapshot "
        "WHERE graph_code=%s AND graph_version=1",
        (graph.graph_code,),
    )[0][0]
    return dict(
        run_id=run_id,
        graph_code=graph.graph_code,
        graph_version=graph.graph_version,
        graph_checksum=graph_checksum,
        references=(mapping.reference(),),
        mappings=writer,
        context=support.context(),
    )


def test_old_run_lock_survives_catalog_and_mapping_publication(database, integrated, tmp_path):
    data, mapping, writer, catalogs = integrated
    args = create_run(database, data, mapping, writer)
    locks = RuntimeGroundingLocks(manage.test_target("runtime"))
    old = locks.bind(**args)
    assert locks.bind(**args) == old == locks.read(args["run_id"])
    assert old.template_ref.checksum != old.graph_template_ref.checksum
    assert locks.query(args["run_id"], writer, support.context())[0].record == mapping
    catalog = catalogs.read_snapshot(mapping.catalog_ref)
    # An actual additional synthetic implementation, not just a renamed catalog version.
    from copy import deepcopy

    from packages.grounding import digest

    manifests = fixture()[4]
    manifests[0]["ports"][-1]["artifact_code"] = data["requirement"].artifact_type_code
    extra_manifest = deepcopy(manifests[0])
    extra_manifest["code"] = "synthetic_ratio_2"
    manifests.append(extra_manifest)
    old_tool = catalog.tools[0].tool_ref.model_dump(mode="json")
    extra_tool_ref = {
        **old_tool,
        "code": extra_manifest["code"],
        "checksum": digest(extra_manifest),
    }

    def new_tool_scope(value):
        if isinstance(value, dict):
            if value == old_tool:
                return extra_tool_ref
            return {k: new_tool_scope(v) for k, v in value.items()}
        if isinstance(value, list):
            return [new_tool_scope(v) for v in value]
        return value

    extra_tool = new_tool_scope(catalog.tools[0].model_dump(mode="json"))
    extra_tool["implementation_ref"] = {
        **extra_tool["implementation_ref"],
        "code": "synthetic_impl_2",
        "checksum": digest({"synthetic_implementation": extra_manifest}),
    }
    extra_candidate = new_tool_scope(mapping.candidates[0].model_dump(mode="json"))
    extra_candidate["code"] = "CAPABILITY:SYNTHETIC:PATH2"
    extra_candidate["implementation_ref"] = extra_tool["implementation_ref"]
    newer_catalog = reseal(
        CatalogSnapshot,
        {
            **catalog.model_dump(mode="json"),
            "version": "1.1.0",
            "manifest_checksum": digest(manifests),
            "tools": [*(t.model_dump(mode="json") for t in catalog.tools), extra_tool],
        },
    )
    catalogs.add(newer_catalog)
    newer = reseal(
        Mapping,
        {
            **mapping.model_dump(mode="json"),
            "version": "1.1.0",
            "catalog_ref": newer_catalog.reference().model_dump(mode="json"),
            "candidates": [
                *(c.model_dump(mode="json") for c in mapping.candidates),
                extra_candidate,
            ],
        },
    )
    writer.new_version(mapping.reference(), newer, support.context(), request_key=key())
    writer.machine_review(newer.reference(), support.context(), request_key=key())
    assert locks.read(args["run_id"]) == old
    assert locks.query(args["run_id"], writer, support.context())[0].record == mapping
    with pytest.raises(psycopg.errors.UniqueViolation):
        locks.bind(**{**args, "references": (newer.reference(),)})
    second = create_run(database, data, newer, writer)
    assert locks.bind(**second).mapping_refs == (newer.reference(),)
    assert len(locks.query(second["run_id"], writer, support.context())[0].record.candidates) == 3
    assert len(locks.query(args["run_id"], writer, support.context())[0].record.candidates) == 2
    catalogs.disable(mapping.candidates[0].implementation_ref)
    with pytest.raises(GroundingError, match="tool_safety_revoked"):
        locks.query(args["run_id"], writer, support.context())
    assert (
        locks.read(args["run_id"]) == old
        and writer.replay(mapping.reference()).snapshot.record == mapping
    )
    # Artifact for review: strict model includes no patient observations or clinical text.
    (tmp_path / "grounding-lock.json").write_text(old.model_dump_json(indent=2))
    catalogs._disabled.clear()  # Synthetic fixture reset only; no real catalog mutation.


def test_lock_permissions_atomic_audit_failure_and_scope(database, integrated):
    data, mapping, writer, _ = integrated
    args = create_run(database, data, mapping, writer)
    locks = RuntimeGroundingLocks(manage.test_target("runtime"))
    before = sql(database.admin, "SELECT count(*) FROM audit.event")[0][0]
    sql(
        database.admin,
        "CREATE FUNCTION public.fail_lock_test() RETURNS trigger LANGUAGE plpgsql "
        "AS $$ BEGIN RAISE EXCEPTION 'test'; END $$",
    )
    sql(
        database.admin,
        "CREATE TRIGGER fail_lock BEFORE INSERT ON audit.event "
        "FOR EACH ROW EXECUTE FUNCTION public.fail_lock_test()",
    )
    try:
        with pytest.raises(psycopg.Error):
            locks.bind(**args)
        assert sql(
            database.admin,
            "SELECT count(*) FROM runtime.run_grounding_lock WHERE run_id=%s",
            (args["run_id"],),
        ) == [(0,)]
        assert sql(database.admin, "SELECT count(*) FROM audit.event")[0][0] == before
    finally:
        sql(database.admin, "DROP TRIGGER fail_lock ON audit.event")
        sql(database.admin, "DROP FUNCTION public.fail_lock_test()")
    with pytest.raises(psycopg.errors.CheckViolation):
        locks.bind(**{**args, "graph_checksum": "sha256:" + "0" * 64})
    lock = locks.bind(**args)
    with connect(manage.test_target("runtime")) as conn:
        conn.execute("SET ROLE mmagent_runtime_app")
        with pytest.raises(psycopg.errors.InsufficientPrivilege), conn.transaction():
            conn.execute(
                "INSERT INTO runtime.run_grounding_lock VALUES(%s,%s)",
                (args["run_id"], json.dumps(lock.model_dump(mode="json"))),
            )
        with pytest.raises(psycopg.errors.InsufficientPrivilege), conn.transaction():
            conn.execute("UPDATE runtime.run_grounding_lock SET payload=payload")
    with pytest.raises(psycopg.errors.CheckViolation):
        sql(database.admin, "UPDATE runtime.run_grounding_lock SET payload=payload")
    bad = lock.model_copy(update={"mapping_refs": ()})
    with pytest.raises(ValueError):
        validate(type(lock), bad)
    writer.invalidate(mapping.reference(), state=State.DISABLED, request_key=key())
    with pytest.raises(GroundingError):
        locks.query(args["run_id"], writer, support.context())
    assert locks.read(args["run_id"]) == lock
