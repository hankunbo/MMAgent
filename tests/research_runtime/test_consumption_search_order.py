"""Ordering metadata never replaces original byte, scope or receipt checks."""

import json
from types import SimpleNamespace

import pytest

from db.runtime.repository.case_pipeline import identity
from packages.object_store import checksum
from packages.research_runtime.agent_records import RuntimeDocuments


def fixture():
    docs = RuntimeDocuments.__new__(RuntimeDocuments)
    docs.run_id = "RUN:CURRENT"
    docs.root = SimpleNamespace(checksum=checksum(b"root"))
    graph = SimpleNamespace(code="ASSET:GRAPH", checksum=checksum(b"graph"))
    artifacts, blobs, reads, metadata = [], {}, [], []

    def add(run, *, accepted=True, canonical=True, root=None, graph_sha=None):
        value = dict(
            schema_version="research-acceptance/2.0.0",
            status="accepted" if accepted else "submitted_not_accepted",
            parent_graph=dict(code=graph.code, checksum=graph_sha or graph.checksum),
        )
        raw = json.dumps(
            dict(run_id=run, root_checksum=root or docs.root.checksum, value=value)
        ).encode()
        sha = checksum(raw)
        uri = "test:" + str(len(artifacts))
        artifact = SimpleNamespace(
            asset_code=identity("ASSET", run + sha) if canonical else "ASSET:LEGACY",
            checksum=sha,
            uri=uri,
            artifact_type_code="ARTIFACT:AGENT_RECORD",
        )
        blobs[uri] = raw
        artifacts.append(artifact)
        return artifact

    def read(uri, bound):
        reads.append(uri)
        assert len(blobs[uri]) <= bound
        return blobs[uri]

    def check():
        metadata.append(True)
        return SimpleNamespace(artifacts=tuple(artifacts))

    docs._check = check
    docs.store = SimpleNamespace(read_bounded=read)
    return SimpleNamespace(
        docs=docs,
        graph=graph,
        add=add,
        artifacts=artifacts,
        blobs=blobs,
        reads=reads,
        metadata=metadata,
    )


def test_current_receipt_is_checked_first_without_caching():
    f = fixture()
    for n in range(500):
        f.add("RUN:OTHER_" + str(n))
    chosen = f.add(f.docs.run_id)
    assert f.docs.consumed(f.graph)
    assert f.reads == [chosen.uri]
    assert f.docs.consumed(f.graph)
    assert f.reads == [chosen.uri, chosen.uri] and len(f.metadata) == 2
    f.blobs[chosen.uri] += b" "
    with pytest.raises(ValueError, match="agent_object_checksum"):
        f.docs.consumed(f.graph)


def test_noncanonical_receipt_is_not_excluded_and_negative_reads_every_record():
    f = fixture()
    f.add("RUN:OTHER")
    f.add(f.docs.run_id, accepted=False)
    assert not f.docs.consumed(f.graph) and len(f.reads) == 2
    legacy = f.add(f.docs.run_id, canonical=False)
    f.reads.clear()
    assert f.docs.consumed(f.graph) and legacy.uri in f.reads
    assert len(f.reads) == 3


@pytest.mark.parametrize("fault", ["root", "graph", "envelope_run"])
def test_identity_hint_does_not_authorize_wrong_binding(fault):
    f = fixture()
    a = f.add(
        f.docs.run_id,
        root=checksum(b"wrong") if fault == "root" else None,
        graph_sha=checksum(b"wrong") if fault == "graph" else None,
    )
    if fault == "envelope_run":
        raw = json.loads(f.blobs[a.uri])
        raw["run_id"] = "RUN:FOREIGN"
        f.blobs[a.uri] = json.dumps(raw).encode()
        a.checksum = checksum(f.blobs[a.uri])
        a.asset_code = identity("ASSET", f.docs.run_id + a.checksum)
    assert not f.docs.consumed(f.graph) and f.reads == [a.uri]
