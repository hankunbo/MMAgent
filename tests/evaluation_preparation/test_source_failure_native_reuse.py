"""Same-window Source failure may reuse, never execute, its exact native fact."""

import copy
import json
from pathlib import Path

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.discovery_continuation import (
    definition_successor_policy,
    reuse_failed_native,
)
from tests.evaluation_preparation.test_root_definition_backlog import root_parent


def setup(tmp_path, monkeypatch, *, scoped=False):
    c, ref, spec, _, _, _, _ = root_parent(tmp_path, confirmed_schema=True)
    base = c.base
    base.output = tmp_path / "new"
    base.output.mkdir()
    base.manifest = type("Manifest", (), {"checksum": spec["input_checksum"]})()
    requirement = dict(id="check:one", component="image:one")
    record = dict(
        schema_version="native-fundus-acceptance/1.1.0"
        if scoped
        else "native-fundus-acceptance/1.0.0",
        plan=dict(target=requirement["id"], component=requirement["component"]),
        bundle_key="original-runtime-bundle",
    )
    directory = Path(ref["parent"]) / "native-fundus"
    if scoped:
        directory /= digest((requirement["id"], requirement["component"]))[7:39]
    directory.mkdir(parents=True)
    path = directory / "accepted.json"
    path.write_text(json.dumps(record))
    reads = []

    def current_owner(current, value):
        assert current is base and value == record
        reads.append(value)

    monkeypatch.setattr("packages.multimodal_research.native_fundus.recheck", current_owner)
    return base, ref, requirement, record, path, reads


@pytest.mark.parametrize("scoped", [False, True])
def test_same_fact_repeated_and_cold_reuse_no_gpu_or_deadline_reset(tmp_path, monkeypatch, scoped):
    base, ref, requirement, record, path, reads = setup(tmp_path, monkeypatch, scoped=scoped)
    original = path.read_bytes()
    runner = reuse_failed_native(base, ref)

    def no_dispatch():
        pytest.fail("native dispatch must not happen")

    assert runner(base, requirement, before_dispatch=no_dispatch, component_scoped=scoped) == record
    files = {p.name: p.read_bytes() for p in base.output.iterdir()}
    assert len(files) == 1
    assert runner(base, requirement, before_dispatch=no_dispatch, component_scoped=scoped) == record
    cold = reuse_failed_native(base, ref, read_only=True)
    assert cold(base, requirement, before_dispatch=no_dispatch, component_scoped=scoped) == record
    assert len(reads) == 3 and path.read_bytes() == original
    assert files == {p.name: p.read_bytes() for p in base.output.iterdir()}
    receipt = json.loads(next(iter(files.values())))
    assert receipt["original_failure"] == ref and receipt["new_model_runs"] == 0


@pytest.mark.parametrize("change", ["parent", "target", "component", "unknown", "current_owner"])
def test_changed_binding_or_unknown_denied_without_receipt(tmp_path, monkeypatch, change):
    base, ref, requirement, _, _, _ = setup(tmp_path, monkeypatch)
    if change == "parent":
        ref["entry"] = digest("wrong")
    elif change in {"target", "component"}:
        requirement["id" if change == "target" else "component"] = "wrong"
    elif change == "unknown":
        ref["mode"] = "root_technical_backlog_no_source_replay"
    else:

        def revoked(*args):
            raise ValueError("native_fundus_current_binding")

        monkeypatch.setattr("packages.multimodal_research.native_fundus.recheck", revoked)
    with pytest.raises(ValueError, match="native"):
        reuse_failed_native(base, ref)(base, requirement, before_dispatch=lambda: None)
    assert not list(base.output.iterdir())


def test_new_profile_only_exact_same_policy_allowed(tmp_path):
    _, _, spec, _, _, _, _ = root_parent(tmp_path, confirmed_schema=True)
    path = "configs/research/multimodal_structural_coverage_v2.json"
    profile = json.loads(Path(path).read_bytes())
    spec["active_acquisition"] = dict(path=path, checksum=digest(profile))
    definition_successor_policy(spec, profile, defer=True, root_confirmed=True)
    changed = copy.deepcopy(profile)
    changed["id"] += "-changed"
    with pytest.raises(ValueError, match="definition_successor_plan_delta"):
        definition_successor_policy(spec, changed, defer=True, root_confirmed=True)


@pytest.mark.parametrize("merged", [False, True])
def test_native_consumer_recomputes_new_reference_bindings_and_cold_reuse(
    tmp_path, monkeypatch, merged
):
    from tests.evaluation_preparation.test_source_provider_transition import selected

    c, ref, _, _ = selected(tmp_path)
    if merged:
        ref["resolved_decisions_policy"] = "retain-resolved-definition-merges/1.0.0"
    c.base.output = tmp_path / "native-consumer"
    c.base.output.mkdir()
    requirement = dict(id="check:one", component="image:one")
    native = dict(
        schema_version="native-fundus-acceptance/1.1.0",
        plan=dict(target=requirement["id"], component=requirement["component"]),
        bundle_key="original-bundle",
    )
    folder = (
        Path(ref["parent"])
        / "native-fundus"
        / digest((requirement["id"], requirement["component"]))[7:39]
    )
    folder.mkdir(parents=True)
    path = folder / "accepted.json"
    path.write_text(json.dumps(native))
    before = path.read_bytes()
    reads = []
    monkeypatch.setattr(
        "packages.multimodal_research.native_fundus.recheck",
        lambda base, record: reads.append(record),
    )
    options = dict(following_provider=c.config, following_review_contract="rating-presence/1.1.0")
    with pytest.raises(ValueError, match="native_provider_binding_required"):
        reuse_failed_native(c.base, ref)
    for cold in (False, True, False):
        runner = reuse_failed_native(c.base, ref, read_only=cold, **options)
        assert (
            runner(
                c.base,
                requirement,
                component_scoped=True,
                before_dispatch=lambda: pytest.fail("new tool execution"),
            )
            == native
        )
    assert len(reads) == 3 and path.read_bytes() == before
    bad = copy.deepcopy(ref)
    bad["provider_transition"]["source_and_vision_roles_unchanged"] = False
    with pytest.raises(ValueError, match="native_parent_changed"):
        reuse_failed_native(c.base, bad, **options)
    monkeypatch.setattr(
        "packages.multimodal_research.native_fundus.recheck",
        lambda *a: (_ for _ in ()).throw(ValueError("current_tool_revoked")),
    )
    with pytest.raises(ValueError, match="current_tool_revoked"):
        reuse_failed_native(c.base, ref, **options)(
            c.base,
            requirement,
            component_scoped=True,
            before_dispatch=lambda: pytest.fail("new tool"),
        )
