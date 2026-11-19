"""Request scoping must reduce irrelevant work, never current consumption gates."""

import json
from types import SimpleNamespace

import pytest

from apps.vectorengine_case_materials import ref
from packages.grounding.models import digest
from packages.multimodal_research import acquisition_scope as scope
from packages.multimodal_research.campaign import InputEvidence, MultimodalCampaign


@pytest.fixture
def base():
    calls = []
    root = ref("resource", "ASSET:INPUT", digest("input"))
    tool = ref("resource", "ASSET:TOOL", digest("tool"))
    value = SimpleNamespace(
        acquisition_scope_policy=scope.POLICY,
        ledger=[dict(id="visual", acquisition_method="vision", component="image")],
        resources=(root, tool),
        calls=calls,
        check=lambda: calls.append("full"),
        check_definition_materials=lambda: calls.append("input_sources_definition_projection"),
        component_observations=SimpleNamespace(
            target="geometry", resource=lambda: tool, check=lambda: calls.append("tool")
        ),
    )
    return value


def test_unrelated_tool_excluded_from_request_and_checks(base):
    assert scope.resources(base, "visual") == base.resources[:1]
    scope.check(base, "visual")
    assert base.calls == ["input_sources_definition_projection"]
    assert len(base.resources) == 2  # The original case still owns its tool fact.


@pytest.mark.parametrize("mode", ["dependency", "quality", "adoption", "component", "method"])
def test_real_dependency_restores_current_tool_owner(base, mode):
    row = base.ledger[0]
    if mode == "dependency":
        row["dependency_paths"] = ((base.resources[1].code,),)
    elif mode == "quality":
        row["quality_source"] = base.resources[1]
    elif mode == "adoption":
        row["definition_receipt_key"] = "accepted-definition"
    elif mode == "component":
        base.component_observations.target = "visual"
    else:
        row["acquisition_method"] = "unrecognized-method"
    assert scope.resources(base, "visual") == base.resources
    scope.check(base, "visual")
    assert base.calls[-1] == "tool"


@pytest.mark.parametrize("mode", ["review", "legacy"])
def test_review_and_old_configuration_keep_full_owner(base, mode):
    if mode == "legacy":
        del base.acquisition_scope_policy
    target = None if mode == "review" else "visual"
    assert scope.resources(base, target) == base.resources
    scope.check(base, target)
    assert base.calls == ["full"]


@pytest.mark.parametrize("paths", [(), ((),), (("ASSET:UNKNOWN",),)])
def test_missing_dependency_is_not_silently_dropped(base, paths):
    base.ledger[0]["dependency_paths"] = paths
    with pytest.raises(ValueError, match="multimodal_acquisition_scope_"):
        scope.check(base, "visual")


def test_input_source_projection_changes_still_refuse(base):
    def changed():
        raise ValueError("source_changed")

    base.check_definition_materials = changed
    with pytest.raises(ValueError, match="source_changed"):
        scope.check(base, "visual")
    assert not base.calls


def test_actual_image_loader_uses_matched_target_scope(base):
    image = ref("image", "ASSET:VIEW:" + digest("visual")[7:23], digest("image"))
    base.view = lambda target: (SimpleNamespace(reference=image), b"image", {})
    assert InputEvidence.image(base, image) == b"image"
    assert base.calls == ["input_sources_definition_projection"]


def test_nested_execution_context_restored_after_failure(base, monkeypatch, tmp_path):
    from packages.evaluation_preparation.dynamic_campaign import Campaign

    campaign = object.__new__(MultimodalCampaign)
    campaign.base = base
    campaign.sessions = None
    campaign.config = object()
    campaign.output = tmp_path
    campaign.phase_path = lambda target: tmp_path / str(target)
    campaign.guard_window = lambda: None
    campaign.guard_control = lambda **kw: base.calls.append("control")

    def execute(self, target=None):
        self.guard(dispatch=True)
        if target:
            self.execute_phase(None)
            assert self._acquisition_target == target
            raise ValueError("transport_failure")
        return dict(target=target, accepted=True)

    monkeypatch.setattr(Campaign, "execute_phase", execute)
    with pytest.raises(ValueError, match="transport_failure"):
        campaign.execute_phase("visual")
    assert campaign._acquisition_target is None
    assert base.calls == ["input_sources_definition_projection", "control", "full", "control"]
    campaign.guard()
    assert base.calls[-2:] == ["full", "control"]


def test_public_new_profile_changes_only_explicit_scope_and_identity(tmp_path):
    from packages.multimodal_research.active_requirements import load

    old = load("configs/research/multimodal_active_requirements_v8.json")
    new = load("configs/research/multimodal_active_requirements_v9.json")
    assert new.pop("acquisition_scope_policy") == scope.POLICY
    assert new.pop("id") != old.pop("id")
    assert new == old
    bad = dict(new, acquisition_scope_policy="unchecked")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="active_acquisition_scope_policy"):
        load(path)
