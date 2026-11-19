"""Metamorphic and malformed-boundary tests using an actual PG-exported native fixture."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from packages.grounding.models import canonical, digest, seal
from packages.integrations.medgraph import (
    AdapterConfig,
    AdapterError,
    MedgraphCatalogReader,
    ProjectionPolicy,
    SnapshotStore,
)
from packages.integrations.medgraph.models import DeploymentManifest, NativeSnapshot, parse
from packages.integrations.medgraph.projection import project
from packages.integrations.medgraph.resources import (
    LocalResourcePolicy,
    local_path,
    observe_resource,
)


def native():
    return json.loads(Path(__file__).with_name("native.synthetic.json").read_text())


def config(**changes):
    return AdapterConfig(source_code="FIXTURE", dsn_env="MEDGRAPH_06A_TEST_DSN", **changes)


def envelope(data=None, cfg=None):
    return project(
        data if data is not None else native(), cfg or config(), ProjectionPolicy(), synthetic=True
    )


@pytest.mark.parametrize(
    "state,expected",
    [
        ("surprise", "quarantined"),
        ("failed", "failed"),
        ("deprecated", "deprecated"),
        ("validated", "registered"),
    ],
)
def test_unknown_status_is_never_admission(state, expected):
    data = native()
    data["implementations"][0]["validation_status"] = state
    for m in data["manifests"]:
        if m["tool_code"] == "fixture_tool" and m["version"] == "1.0.0":
            m["validation_status"] = state
    result = envelope(data)
    binding = next(b for b in result.bindings if b.implementation_id == 1)
    tool = next(
        t for t in result.catalog.tools if t.implementation_ref == binding.implementation_ref
    )
    assert tool.catalog_status.value == expected
    assert not tool.enabled
    assert tool.validation_status.value != "auto_validated_research"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["manifests"][0].update(enabled=True),
        lambda d: d["manifests"][0].pop("invocation"),
        lambda d: d["manifests"][0].update(version="9.9.9"),
        lambda d: d["implementations"][0].update(tool_version_id=999),
        lambda d: d["ports"][0].update(artifact_type_id=999),
        lambda d: d["ports"].append(copy.deepcopy(d["ports"][0])),
        lambda d: d["artifacts"][0].update(unrecognized=True),
        lambda d: d["edges"][0].update(target_step_id=999),
    ],
)
def test_missing_unknown_and_cross_scope_fields(mutation):
    data = native()
    mutation(data)
    with pytest.raises(ValueError):
        envelope(data)


def test_unknown_constraint_preserved_but_never_invented_as_known():
    data = native()
    data["ports"][0]["constraints"]["future_constraint"] = {"x": 1}
    result = envelope(data)
    assert "future_constraint" in result.native.ports[0].constraints
    assert all(t.inputs[0].constraint.modality.state == "unknown" for t in result.catalog.tools)
    assert all(not t.enabled for t in result.catalog.tools)


def test_deterministic_order_and_configuration_version():
    data = native()
    expected = envelope(data)
    for rows in data.values():
        rows.reverse()
    for manifest in data["manifests"]:
        manifest["resources"].reverse()
    assert envelope(data) == expected
    assert envelope(cfg=config(version="1.0.1")).catalog.reference() != expected.catalog.reference()
    data["manifests"][0]["invocation"]["argv"].reverse()
    with pytest.raises(AdapterError, match="drift"):
        envelope(data)


def test_snapshot_roundtrip_tampering_exact_manifest(tmp_path):
    old = envelope()
    store = SnapshotStore(tmp_path)
    reference = store.save(old)
    assert store.save(old) == reference
    assert store.read(reference) == old

    class Source:
        def capture(self):
            return old.native

    reader = MedgraphCatalogReader(Source(), store, old.config, old.policy)
    t, other = old.catalog.tools[:2]
    with pytest.raises(AdapterError):
        reader.read_manifest(reference, t.tool_ref, other.implementation_ref)
    manifest = reader.read_manifest(reference, t.tool_ref, t.implementation_ref)
    with pytest.raises(ValueError):
        seal(DeploymentManifest, **{**manifest.model_dump(mode="json"), "tool_ref": other.tool_ref})
    bad = json.loads(canonical(old))
    bad["catalog"]["tools"][0]["enabled"] = True
    bad["catalog"].pop("checksum")
    bad["catalog"]["checksum"] = digest(bad["catalog"])
    bad.pop("checksum")
    bad["checksum"] = digest(bad)
    with pytest.raises(AdapterError, match="derived_projection"):
        store.save(bad)
    path = tmp_path / (reference.checksum[7:] + ".json")
    path.chmod(0o644)
    path.write_text("{}")
    with pytest.raises(AdapterError):
        store.read(reference)


@pytest.mark.parametrize(
    "payload",
    [
        {"password": "do-not-leak"},
        {"nested": {"api_key": "do-not-leak"}},
        {"source": "https://username:do-not-leak@example.test/source"},
        {"argv": ["--token=do-not-leak"]},
    ],
)
def test_secret_rejection_never_echoes_values(payload):
    data = native()
    data["tools"][0]["metadata"] = payload
    with pytest.raises(AdapterError) as error:
        envelope(data)
    assert "do-not-leak" not in str(error.value)


def test_construct_copy_is_revalidated():
    data = parse(NativeSnapshot, native())
    copied = data.model_copy(update={"arbitrary": True})
    with pytest.raises(AdapterError):
        envelope(copied)
    data.tools[0].metadata["password"] = "secret"
    with pytest.raises(AdapterError):
        envelope(data)


def test_public_import_does_not_load_models_or_start_processes():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import subprocess
subprocess.Popen = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("process started"))
import packages.integrations.medgraph
for name in ("torch", "cv2", "Glaucoma", "numpy", "mcp", "Algorithm_Tool", "Seg_Model"):
    assert name not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def local_resource_directory(tmp_path):
    return tmp_path


def test_local_resource_directory_belongs_to_current_session(
    local_resource_directory, tmp_path_factory
):
    base = tmp_path_factory.getbasetemp()
    session_root = base.parent
    project_local = Path(__file__).resolve().parents[3] / ".local"
    assert session_root.parent == project_local
    assert session_root.name.startswith("pytest-session-")
    assert not session_root.is_symlink()
    assert not local_resource_directory.is_symlink()
    assert local_resource_directory.resolve().is_relative_to(base.resolve())
    owner = json.loads((session_root / "owner.json").read_text())
    assert owner["pid"] == os.getpid()
    assert owner["nonce"]
    print(
        json.dumps(
            {"resource_directory": str(local_resource_directory), "session_root": str(session_root)}
        )
    )


def test_local_uri_literal_percent_and_tree_digests(local_resource_directory):
    tmp_path = local_resource_directory
    policy = LocalResourcePolicy(allowed_roots=(str(tmp_path),))
    literal = tmp_path / "weights%20name"
    literal.write_bytes(b"weights")
    assert "%2520" in literal.as_uri()
    assert local_path(literal.as_uri(), policy) == literal
    first = observe_resource(literal.as_uri(), policy)
    literal.write_bytes(b"new weights")
    assert observe_resource(literal.as_uri(), policy).checksum != first.checksum
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "b").write_bytes(b"b")
    (tree / "a").write_bytes(b"a")
    first = observe_resource(tree.as_uri(), policy)
    assert first == observe_resource(tree.as_uri(), policy)
    (tree / "a").rename(tree / "c")
    assert first.checksum != observe_resource(tree.as_uri(), policy).checksum
    link = tmp_path / "source_link"
    link.symlink_to(tree)
    assert observe_resource(link.as_uri(), policy).checksum
    (tree / "cycle").symlink_to(tree)
    with pytest.raises(AdapterError, match="cycle"):
        observe_resource(tree.as_uri(), policy)


@pytest.mark.parametrize(
    "suffix", ["/../escape", "/%2e%2e/escape", "/x%00bad", "/x?token=bad", "/x#bad", "/x%QQ"]
)
def test_local_paths_reject_traversal(tmp_path, suffix):
    policy = LocalResourcePolicy(allowed_roots=(str(tmp_path),))
    with pytest.raises(ValueError):
        local_path(tmp_path.as_uri() + suffix, policy)


def test_symlink_escape_resource_bounds_and_no_creation(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    link = root / "escape"
    link.symlink_to(outside)
    policy = LocalResourcePolicy(allowed_roots=(str(root),))
    with pytest.raises(AdapterError, match="outside"):
        observe_resource(link.as_uri(), policy)
    good = root / "good"
    good.write_bytes(b"12345")
    with pytest.raises(AdapterError, match="byte_limit"):
        observe_resource(good.as_uri(), policy.model_copy(update={"max_bytes": 1}))
    before = set(root.iterdir())
    with pytest.raises(FileNotFoundError):
        local_path((root / "new" / "out").as_uri(), policy)
    assert set(root.iterdir()) == before


def test_resource_single_file_resolves_once_before_read(tmp_path, monkeypatch):
    from pathlib import Path

    item = tmp_path / "current"
    item.write_bytes(b"current bytes")
    policy = LocalResourcePolicy(allowed_roots=(str(tmp_path),))
    original = Path.resolve
    calls = []

    def resolved(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolved)
    first = observe_resource(item.as_uri(), policy)
    assert calls == [tmp_path, item, item]
    # No cache spans observations: a later symlink escape still rejects.
    item.unlink()
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    try:
        item.symlink_to(outside, target_is_directory=True)
        with pytest.raises(AdapterError, match="outside_roots"):
            observe_resource(item.as_uri(), policy)
        assert first.byte_size == len(b"current bytes")
    finally:
        item.unlink()
        outside.rmdir()


def test_source_backed_policy_is_only_restriction_not_gate(tmp_path):
    from packages.integrations.medgraph.configuration import load_projection_policy

    source = tmp_path / "research.md"
    source.write_text("Synthetic tool use is restricted to research_only.")
    import hashlib

    raw = {
        "declarations": [
            {
                "tool_code": "fixture_tool",
                "source_path": "research.md",
                "source_checksum": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
                "excerpt": "restricted to research_only",
                "intended_use": "research_only",
            }
        ]
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    policy = load_projection_policy(path, tmp_path)
    result = project(native(), config(), policy, synthetic=True)
    assert all(not t.enabled for t in result.catalog.tools)
    source.write_text("changed")
    with pytest.raises(AdapterError, match="drift"):
        load_projection_policy(path, tmp_path)


def test_real_catalog_projects_into_existing_grounding_filter():
    from packages.grounding.compatibility import tool_compatibility
    from packages.grounding.models import Candidate, Policy
    from packages.grounding.ports import ToolSafety
    from packages.grounding.synthetic import fixture

    result = envelope()
    tool = result.catalog.tools[0]
    policy = fixture()[1]
    policy = seal(
        Policy, **{**policy.model_dump(mode="json"), "accepted_validation": ["smoke_tested"]}
    )
    candidate = Candidate(
        code="CANDIDATE:MEDGRAPH:TEST",
        capability_ref=tool.capabilities[0],
        tool_ref=tool.tool_ref,
        implementation_ref=tool.implementation_ref,
        workflow_ref=None,
        output_port=tool.output_ports[0],
        output=tool.outputs[0],
        inputs=tool.inputs,
        validation_required=tool.validation_status,
        intended_use=tool.intended_use,
        execution_policy=policy.execution_policy,
        estimate_ref=None,
    )
    safety = ToolSafety(
        catalog_ref=result.catalog.reference(), disabled_implementations=(), revision=0
    )
    assert "tool_not_active" in tool_compatibility(candidate, result.catalog, policy, safety)


def test_manifest_resource_bytes_have_separate_exact_binding(tmp_path):
    from packages.integrations.medgraph.resources import observe_manifest_resources

    data = native()
    for r in data["resources"]:
        path = tmp_path / str(r["id"])
        path.write_bytes(b"source-or-weights")
        r["uri"] = path.as_uri()
    for m in data["manifests"]:
        v = next(
            v
            for v in data["versions"]
            if v["version"] == m["version"]
            and v["tool_id"] == next(t["id"] for t in data["tools"] if t["code"] == m["tool_code"])
        )
        i = next(i for i in data["implementations"] if i["tool_version_id"] == v["id"])
        for r in m["resources"]:
            r["uri"] = next(
                x["uri"]
                for x in data["resources"]
                if x["implementation_id"] == i["id"] and x["resource_key"] == r["resource_key"]
            )
    result = envelope(data)
    store = SnapshotStore(tmp_path)
    store.save(result)
    reader = MedgraphCatalogReader(None, store, result.config, result.policy)
    tool = result.catalog.tools[0]
    manifest = reader.read_manifest(
        result.catalog.reference(), tool.tool_ref, tool.implementation_ref
    )
    policy = LocalResourcePolicy(allowed_roots=(str(tmp_path),))
    before = observe_manifest_resources(manifest, policy)
    path = local_path(manifest.native_manifest.resources[0].uri, policy)
    path.write_bytes(b"changed-code-or-weights")
    after = observe_manifest_resources(manifest, policy)
    assert before.checksum != after.checksum
    assert before.manifest_checksum == after.manifest_checksum == manifest.checksum
    assert not after.execution_authorized


def test_resource_tree_changed_during_read_is_rejected(local_resource_directory, monkeypatch):
    import os
    from pathlib import Path

    tree = local_resource_directory / "changing"
    tree.mkdir()
    (tree / "original").write_bytes(b"original")
    before = tree.stat()
    original = Path.iterdir

    def changed(path):
        yield from original(path)
        if path == tree:
            (tree / "late-addition").write_bytes(b"changed while observing")
            # Force an observable metadata change even on coarse timestamp filesystems.
            os.utime(tree, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))

    monkeypatch.setattr(Path, "iterdir", changed)
    policy = LocalResourcePolicy(allowed_roots=(str(local_resource_directory),))
    with pytest.raises(AdapterError, match="resource_changed_during_read"):
        observe_resource(tree.as_uri(), policy)
