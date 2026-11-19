"""Reviewed overlays use native fixture provenance, never real GPU evidence."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.grounding.models import canonical, digest, seal
from packages.integrations.medgraph import (
    AdapterConfig,
    MedgraphCatalogReader,
    ProjectionPolicy,
    SnapshotStore,
)
from packages.integrations.medgraph.models import NativeSnapshot
from packages.integrations.medgraph.projection import expected_manifest
from packages.integrations.medgraph.research import (
    ResearchReview,
    ReviewedCatalog,
    ReviewedCatalogSet,
)
from packages.integrations.medgraph.resources import LocalResourcePolicy, observe_manifest_resources
from packages.tool_execution.trusted_models import ReviewedFile


def pin(p):
    return ReviewedFile(
        path=str(p), checksum=digest_bytes(p.read_bytes()), byte_size=p.stat().st_size
    )


def digest_bytes(b):
    import hashlib

    return "sha256:" + hashlib.sha256(b).hexdigest()


@pytest.fixture
def reviewed(tmp_path):
    raw = json.loads(Path(__file__).with_name("native.synthetic.json").read_text())
    code = tmp_path / "runner.py"
    code.write_text('print("fixture only")\n')
    for r in raw["resources"]:
        r.update(uri=code.as_uri(), sha256=pin(code).checksum[7:], byte_size=code.stat().st_size)
    native = NativeSnapshot.model_validate_json(json.dumps(raw))
    native = native.model_copy(
        update={
            "manifests": tuple(
                expected_manifest(native, i) for i in native.implementations if i.enabled
            )
        }
    )

    class Source:
        value = native

        def capture(self):
            return self.value

    source = Source()
    store = tmp_path / "store"
    store.mkdir()
    reader = MedgraphCatalogReader(
        source,
        SnapshotStore(store),
        AdapterConfig(source_code="FIXTURE", dsn_env="TEST"),
        ProjectionPolicy(),
    )
    catalog = reader.capture_snapshot()
    tool = next(t for t in catalog.tools if t.validation_status.value == "smoke_tested")
    manifest = reader.read_manifest(catalog.reference(), tool.tool_ref, tool.implementation_ref)
    policy = LocalResourcePolicy(allowed_roots=(str(tmp_path),))
    evidence = tmp_path / "native-resources.json"
    evidence.write_bytes(canonical(observe_manifest_resources(manifest, policy)))
    note = tmp_path / "review.md"
    note.write_text("Fixture review: exact file, input and output, no clinical claim.\n")
    review = seal(
        ResearchReview,
        source_catalog_ref=catalog.reference(),
        source_implementation_ref=tool.implementation_ref,
        source_manifest_checksum=manifest.checksum,
        config_checksum=manifest.config_checksum,
        resolutions=[
            dict(
                gap=g,
                basis="Reviewed fixture only; real GPU and performance not claimed.",
                evidence=pin(note),
                disposition="formal_gate_deferred_shadow_only"
                if g == "validation_gates_missing"
                else "verified_for_shadow",
            )
            for g in manifest.binding.gaps
        ],
        input_constraints=[i.constraint for i in tool.inputs],
        output_constraints=tool.outputs,
        deployment_declaration=manifest.native_manifest,
        parameter_schema={},
        preprocessing_policy={},
        postprocessing_policy={},
        interpreter=pin(code),
        native_resource_evidence=pin(evidence),
    )
    record = tmp_path / "installed-review.json"
    record.write_bytes(canonical(review))
    return SimpleNamespace(
        reader=reader,
        catalog=catalog,
        source=source,
        manifest=manifest,
        policy=policy,
        review=review,
        record=record,
        code=code,
    )


def test_exact_installed_review_preserves_legacy_and_native(reviewed):
    f = reviewed
    old = canonical(f.reader.read_snapshot(f.catalog.reference()))
    c = ReviewedCatalog(f.reader, pin(f.record), f.policy)
    assert c.tool.enabled and c.tool.validation_status.value == "smoke_tested"
    assert c.manifest.native_manifest == f.manifest.native_manifest
    assert c.manifest.original_manifest_checksum == f.manifest.checksum
    assert canonical(f.reader.read_snapshot(f.catalog.reference())) == old
    assert not any(t.enabled for t in f.catalog.tools)
    assert c.catalog.reference() != f.catalog.reference()
    assert not c.current_safety(c.catalog.reference()).disabled_implementations


@pytest.mark.parametrize(
    "change", ["missing", "old_version", "forged", "incomplete", "source_version", "resource"]
)
def test_uninstalled_or_drifted_review_rejected(reviewed, change):
    f = reviewed
    pinned = pin(f.record)
    data = f.review.model_dump(mode="json")
    data.pop("checksum")
    if change == "missing":
        f.record.unlink()
    elif change == "resource":
        f.code.write_text("changed bytes")
    elif change == "forged":
        data["approved"] = True
        f.record.write_text(json.dumps(data))
    elif change == "old_version":
        data["version"] = "medgraph-research-review/0.0.0"
        f.record.write_text(json.dumps(data))
    else:
        if change == "incomplete":
            data["resolutions"] = data["resolutions"][1:]
        else:
            data["source_manifest_checksum"] = digest("other version")
        f.record.write_bytes(canonical(seal(ResearchReview, **data)))
        pinned = pin(f.record)  # Even a newly pinned incomplete record cannot waive checks.
    with pytest.raises((ValueError, OSError)):
        ReviewedCatalog(f.reader, pinned, f.policy)


def test_current_revocation_and_failed_source_are_not_overridden(reviewed):
    f = reviewed
    c = ReviewedCatalog(f.reader, pin(f.record), f.policy)
    target = f.manifest.binding.implementation_id
    changed = tuple(
        i.model_copy(update={"validation_status": "failed"}) if i.id == target else i
        for i in f.source.value.implementations
    )
    f.source.value = f.source.value.model_copy(update={"implementations": changed})
    assert c.current_safety(c.catalog.reference()).disabled_implementations == (
        c.tool.implementation_ref,
    )
    with pytest.raises(ValueError):
        c.read_snapshot(f.catalog.reference())


def test_installed_set_is_new_lock_and_preserves_current_safety(reviewed):
    f = reviewed
    component = ReviewedCatalog(f.reader, pin(f.record), f.policy)
    bundle = ReviewedCatalogSet((component,))
    assert bundle.catalog.reference() != component.catalog.reference()
    manifest = bundle.read_manifest(
        bundle.catalog.reference(), component.tool.tool_ref, component.tool.implementation_ref
    )
    assert manifest.native_manifest == f.manifest.native_manifest
    assert manifest.catalog_ref == bundle.catalog.reference()
    with pytest.raises(ValueError):
        bundle.read_snapshot(f.catalog.reference())
    with pytest.raises(ValueError):
        ReviewedCatalogSet((component, component))
    f.source.value = f.source.value.model_copy(
        update={
            "implementations": tuple(
                i.model_copy(update={"enabled": False})
                if i.id == f.manifest.binding.implementation_id
                else i
                for i in f.source.value.implementations
            )
        }
    )
    assert bundle.current_safety(bundle.catalog.reference()).disabled_implementations == (
        component.tool.implementation_ref,
    )
