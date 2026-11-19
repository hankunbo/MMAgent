"""A locator must not replace original current metadata and byte validation."""

import json
from types import SimpleNamespace as NS

import pytest

from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.object_store import checksum
from packages.research_runtime.agent_records import RuntimeDocuments


def setup():
    docs = RuntimeDocuments.__new__(RuntimeDocuments)
    docs.run_id = "RUN:CURRENT"
    scope = NS(case_code="CASE:CURRENT")
    graph = NS(code="ASSET:GRAPH", checksum=checksum(b"graph"))
    raw = {}
    artifacts = {}
    bundles = {}
    inventory = []
    scans = []
    reads = []
    root = NS(
        asset_code="ASSET:ROOT",
        scope=scope,
        checksum=checksum(b"root"),
        uri="test:root",
        artifact_type_code="ARTIFACT:GRAPH",
    )
    docs.root = root
    artifacts[root.asset_code] = root
    raw[root.uri] = b"root"
    inventory.append(root)

    def get_bundle(case):
        scans.append(case)
        return NS(artifacts=tuple(a for a in inventory if a.scope.case_code == case))

    repo = NS(
        read_case_artifact=lambda case, code: artifacts.get(code),
        read_observation_bundle=lambda key: bundles.get(key),
        get_case_bundle=get_bundle,
    )
    docs.repo = repo
    docs.contexts = NS(read_research_run=lambda rid: NS(request=NS(scope=scope)))

    def read(uri, bound):
        reads.append(uri)
        value = raw[uri]
        assert len(value) <= bound
        return value

    docs.store = NS(read_bounded=read)

    def add(name, value, **changes):
        env = dict(
            run_id=docs.run_id,
            root_checksum=root.checksum,
            key_checksum=digest(name),
            index=0,
            value=value,
        )
        env.update(changes)
        data = json.dumps(env).encode()
        sha = checksum(data)
        a = NS(
            asset_code=identity("ASSET", docs.run_id + sha),
            scope=scope,
            checksum=sha,
            uri="test:" + str(len(raw)),
            artifact_type_code="ARTIFACT:AGENT_RECORD",
        )
        raw[a.uri] = data
        artifacts[a.asset_code] = a
        inventory.append(a)
        bundles[docs.key(name)] = (a,)
        return a

    key = "test-acceptance"
    receipt = add(
        "accepted:" + key,
        dict(
            schema_version="research-acceptance/2.0.0",
            status="accepted",
            parent_graph=dict(code=graph.code, checksum=graph.checksum),
        ),
    )
    phase = add("dynamic-phase-result", dict(run_id=docs.run_id, accepted=True, acceptance_key=key))
    return NS(
        docs=docs,
        graph=graph,
        root=root,
        receipt=receipt,
        phase=phase,
        raw=raw,
        artifacts=artifacts,
        bundles=bundles,
        inventory=inventory,
        scans=scans,
        reads=reads,
        add=add,
        key=key,
    )


def test_exact_current_lookup_avoids_case_inventory_but_never_caches():
    f = setup()
    assert f.docs.consumed(f.graph) and not f.scans
    before = len(f.reads)
    assert f.docs.consumed(f.graph) and len(f.reads) > before and not f.scans
    f.raw[f.receipt.uri] += b" "
    with pytest.raises(ValueError, match="agent_object_checksum"):
        f.docs.consumed(f.graph)


def test_phase_alone_is_not_acceptance():
    f = setup()
    f.bundles.pop(f.docs.key("accepted:" + f.key))
    f.inventory.remove(f.receipt)
    assert not f.docs.consumed(f.graph) and f.scans


@pytest.mark.parametrize(
    "fault", ["missing_phase", "unaccepted_phase", "foreign_phase", "missing_key", "unbound_key"]
)
def test_invalid_locator_preserves_original_legacy_fallback(fault):
    f = setup()
    if fault == "missing_phase":
        f.bundles.pop(f.docs.key("dynamic-phase-result"))
    else:
        value = dict(run_id=f.docs.run_id, accepted=True, acceptance_key=f.key)
        if fault == "unaccepted_phase":
            value["accepted"] = False
        if fault == "foreign_phase":
            value["run_id"] = "RUN:FOREIGN"
        if fault == "missing_key":
            value.pop("acceptance_key")
        if fault == "unbound_key":
            value["acceptance_key"] = "other"
        f.add("dynamic-phase-result", value)
    assert f.docs.consumed(f.graph) and f.scans


@pytest.mark.parametrize(
    "fault",
    [
        "parent_code",
        "parent_checksum",
        "receipt_run",
        "receipt_root",
        "not_accepted",
        "wrong_type",
        "wrong_case",
        "metadata_removed",
    ],
)
def test_wrong_receipt_cannot_be_authorized_by_locator(fault):
    f = setup()
    f.inventory.remove(f.receipt)
    value = dict(
        schema_version="research-acceptance/2.0.0",
        status="accepted",
        parent_graph=dict(code=f.graph.code, checksum=f.graph.checksum),
    )
    extra = {}
    if fault == "parent_code":
        value["parent_graph"]["code"] = "ASSET:OTHER"
    if fault == "parent_checksum":
        value["parent_graph"]["checksum"] = checksum(b"other")
    if fault == "receipt_run":
        extra["run_id"] = "RUN:OTHER"
    if fault == "receipt_root":
        extra["root_checksum"] = checksum(b"other")
    if fault == "not_accepted":
        value["status"] = "submitted_not_accepted"
    a = f.add("accepted:" + f.key, value, **extra)
    if fault == "wrong_type":
        a.artifact_type_code = "ARTIFACT:GRAPH"
    if fault == "wrong_case":
        a.scope = NS(case_code="CASE:OTHER")
    if fault == "metadata_removed":
        f.artifacts.pop(a.asset_code)
        f.inventory.remove(a)
    assert not f.docs.consumed(f.graph)


def test_legacy_noncanonical_key_envelope_still_uses_inventory():
    f = setup()
    f.inventory.remove(f.receipt)
    a = f.add(
        "accepted:" + f.key,
        dict(
            schema_version="research-acceptance/2.0.0",
            status="accepted",
            parent_graph=dict(code=f.graph.code, checksum=f.graph.checksum),
        ),
        key_checksum=digest("legacy-key"),
    )
    assert f.docs.consumed(f.graph) and f.scans and a.uri in f.reads
