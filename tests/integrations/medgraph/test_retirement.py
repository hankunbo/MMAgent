"""Project withdrawal blocks new use without rewriting native history."""

import pytest

from packages.integrations.medgraph import (
    AdapterError,
    MedgraphCatalogReader,
    PostgresSource,
    SnapshotStore,
)
from packages.integrations.medgraph.retirement import RETIRED_IMPLEMENTATIONS, retired
from tests.integrations.medgraph.test_projection import config, envelope, native


@pytest.mark.parametrize("identity", sorted(RETIRED_IMPLEMENTATIONS))
def test_retired_native_manifest_rejected_before_connection(identity):
    def resolver():
        pytest.fail("withdrawn implementation must not resolve credentials")

    source = PostgresSource(config(), dsn_resolver=resolver)
    with pytest.raises(AdapterError, match="implementation_withdrawn_by_project"):
        source.exact_manifest(*identity)


def test_unrelated_models_not_withdrawn():
    assert not retired("medical_sam_optic_disc_segmenter", "1.0.0", "local_medagent_pro")
    assert not retired("fairclip_slo_glaucoma_classifier", "1.0.0", "local_fairclip_race_v1")
    assert not retired("samed_segmenter", "1.0.0", "local_samed")


@pytest.mark.parametrize("identity", sorted(RETIRED_IMPLEMENTATIONS))
def test_old_snapshot_readable_but_manifest_and_safety_reject(tmp_path, identity):
    data = native()
    tool_code, version, key = identity
    impl = data["implementations"][0]
    v = next(v for v in data["versions"] if v["id"] == impl["tool_version_id"])
    t = next(t for t in data["tools"] if t["id"] == v["tool_id"])
    old_code, old_version, old_key = t["code"], v["version"], impl["implementation_key"]
    for m in data["manifests"]:
        if m["tool_code"] == old_code:
            m["tool_code"] = tool_code
            if (m["version"], m["implementation_key"]) == (old_version, old_key):
                m.update(version=version, implementation_key=key)
    t["code"], v["version"], impl["implementation_key"] = identity
    old = envelope(data)
    store = SnapshotStore(tmp_path)
    ref = store.save(old)

    class Source:
        def capture(self):
            return old.native

    reader = MedgraphCatalogReader(Source(), store, old.config, old.policy)
    binding = next(b for b in old.bindings if b.implementation_id == impl["id"])
    assert reader.read_snapshot(ref) == old.catalog
    assert reader.read_native(ref) == old.native
    with pytest.raises(AdapterError, match="implementation_withdrawn_by_project"):
        reader.read_manifest(ref, binding.tool_ref, binding.implementation_ref)
    assert binding.implementation_ref in reader.current_safety(ref).disabled_implementations
    assert store.read(ref) == old
