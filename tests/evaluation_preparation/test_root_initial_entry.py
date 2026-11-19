"""New cases opt into the explicit root question from their first frozen entry."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.grounding.models import digest
from packages.multimodal_research import entry, implementation
from packages.multimodal_research.campaign import MultimodalCampaign, configuration
from packages.multimodal_research.single_proposition import root_direction_spec
from tests.dataset_adapters.test_multimodal import gamma, gdp, grape

PROFILE = "relay-root-question-v1"


def test_report_reconstruction_preserves_explicit_task_question_and_old_default():
    from packages.multimodal_research.report_recovery import restore_execution_context

    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    question = root_direction_spec(task)
    spec = dict(
        schema_version="multimodal-live-entry/1.25.0",
        execution_profile=PROFILE,
        root_direction_review=question,
    )
    protocol = dict(spec, schema_version="multimodal-live-campaign/1.15.0")
    base = SimpleNamespace(task=task)
    restore_execution_context(base, spec, protocol)
    assert base.root_direction_review == question
    for tamper in (
        {"root_direction_review": {**question, "title": "unbound"}},
        {"schema_version": "multimodal-live-campaign/1.14.0"},
        {"execution_profile": "relay-snapshot-context-v1"},
    ):
        with pytest.raises(ValueError, match="saved_report_(root_question|execution_profile)"):
            restore_execution_context(SimpleNamespace(task=task), spec, {**protocol, **tamper})
    old_spec = dict(
        schema_version="multimodal-live-entry/1.19.0", execution_profile="relay-snapshot-context-v1"
    )
    old_base = SimpleNamespace(task=task)
    restore_execution_context(old_base, old_spec, dict(old_spec))
    assert not hasattr(old_base, "root_direction_review")
    with pytest.raises(ValueError, match="saved_report_root_question_version"):
        restore_execution_context(
            old_base, {**old_spec, "root_direction_review": question}, old_spec
        )


def test_root_pending_recovery_keeps_provider_deadline_and_zero_wire_boundaries(tmp_path):
    from packages.multimodal_research.campaign_renewal import (
        unstarted_context_proof,
        validate_context_window,
    )
    from tests.evaluation_preparation.test_full_context_recovery import unstarted_fixture

    parent, result, plan, values = unstarted_fixture(tmp_path)
    parent.base.execution_profile = PROFILE
    parent.protocol["schema_version"] = "multimodal-live-campaign/1.15.0"
    proof = unstarted_context_proof(parent, result, plan["tip"])
    assert proof["run_id"] == plan["tip"]["run_id"]
    window = dict(
        whole_case_execution_profile=PROFILE,
        original_window_deadline=parent.protocol["deadline"],
        deadline=parent.protocol["deadline"],
        started_at=parent.sessions.read(plan["tip"]["run_id"]).binding.started_at.isoformat(),
        pending_review={"selected_ids": ["a"]},
        new_vision_requests=0,
        new_public_search_allowed=False,
        provider=digest(parent.config),
    )
    validate_context_window(window, parent)
    with pytest.raises(ValueError, match="original_window_binding"):
        validate_context_window({**window, "provider": digest("other")}, parent)
    values["journal:0"] = ({"phase": "reserved"},)
    with pytest.raises(ValueError, match="not_zero_wire_deadline"):
        unstarted_context_proof(parent, result, plan["tip"])


def test_source_vision_route_limits_and_old_profiles_stay_exact():
    options = dict(directional=True, resilient_vision=True, exact_primary_status=True)
    old = configuration(**options, execution_profile="relay-snapshot-context-v1")
    new = configuration(**options, execution_profile=PROFILE)
    a, b = old.model_dump(mode="json"), new.model_dump(mode="json")
    ar = next(r for r in a["roles"] if r["role"] == "review")
    br = next(r for r in b["roles"] if r["role"] == "review")
    assert ar["prompt_version"] == "review-direction-2.4.0"
    assert br["prompt_version"] == "review-direction-2.7.0"
    br.update(prompt_version=ar["prompt_version"], prompt_checksum=ar["prompt_checksum"])
    assert a == b


@pytest.mark.parametrize("dataset", ["harvard_gdp", "gamma", "grape"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_original_adapter_prepare_and_local_preflight_root_binding(
    tmp_path, monkeypatch, dataset, batch_size
):
    # The adapter-only fixture deliberately contains JPEG headers without pixel
    # data. This entry test executes the real display worker, so provide an
    # actually decodable synthetic image instead of mocking that worker.
    encoded = subprocess.check_output(
        [
            "/opt/conda/bin/python",
            "-c",
            "import io,sys; from PIL import Image; b=io.BytesIO(); "
            "Image.new('RGB',(3,2),(60,70,80)).save(b,format='JPEG'); "
            "sys.stdout.buffer.write(b.getvalue())",
        ]
    )
    monkeypatch.setattr("tests.dataset_adapters.test_multimodal.jpeg", lambda: encoded)
    options = {
        "harvard_gdp": (
            gdp,
            "data_0001.npz",
            None,
            "harvard-gdp-v1",
            "multimodal_harvard_visual_priority_v2.json",
        ),
        "gamma": (gamma, "0001", "training", None, "multimodal_active_requirements_v9.json"),
        "grape": (
            lambda p: grape(p, with_images=True),
            "1:OD",
            None,
            "grape-v1",
            "multimodal_grape_active_requirements_v3.json",
        ),
    }
    fixture, sample, selection, semantics, profile = options[dataset]
    root, out = fixture(tmp_path / "source"), tmp_path / "entry"
    entry.prepare_entry(
        dataset=dataset,
        root=root,
        sample=sample,
        selection=selection,
        cutoff_visit=1 if dataset == "grape" else None,
        task="advanced_glaucoma",
        output=out,
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition="configs/research/" + profile,
        resilient_vision=True,
        view_profile="detail-v1",
        input_semantics=semantics,
        execution_profile=PROFILE,
        review_batch_size=batch_size,
    )
    # Local route readiness isn't a tool qualification test. Keep real dataset/
    # task/config checks but do not inspect an installed patient's native model.
    monkeypatch.setattr(
        "packages.multimodal_research.entry_preflight.check_local_route",
        lambda *args: "test_local_route_only",
    )
    monkeypatch.setattr(entry, "owners", lambda: pytest.fail("preflight reached database owners"))
    assert entry.assemble_entry(out, preflight_only=True)["status"] == "local_preflight_passed"
    path = out / "entry.json"
    spec = json.loads(path.read_bytes())
    assert spec["schema_version"] == (
        "multimodal-live-entry/1.26.0" if batch_size != 1 else "multimodal-live-entry/1.25.0"
    )
    assert spec["provider"]["review_batching"]["batch_size"] == batch_size
    assert spec["root_direction_review"]["title"] == "晚期青光眼"
    assert spec["seconds"] == 5400 and spec["max_requests"] == 40
    assert (
        next(r for r in spec["provider"]["roles"] if r["role"] == "review")["prompt_version"]
        == "review-direction-2.7.0"
    )
    assert "SECRET_DIAGNOSIS" not in json.dumps(spec)
    for changed in (
        dict(spec, root_direction_review={**spec["root_direction_review"], "title": "other"}),
        dict(spec, previous="foreign"),
        dict(spec, schema_version="multimodal-live-entry/1.19.0"),
    ):
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="root_question|contextual_execution"):
            entry.assemble_entry(out, preflight_only=True)
    path.write_text(json.dumps(spec))


def test_batched_cold_recovery_matches_explicit_configuration():
    from packages.multimodal_research.report_recovery import restore_execution_context

    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    spec = dict(
        schema_version="multimodal-live-entry/1.26.0",
        execution_profile=PROFILE,
        review_batch_size=3,
        root_direction_review=root_direction_spec(task),
    )
    plan = dict(spec, schema_version="multimodal-live-campaign/1.16.0")
    base = SimpleNamespace(task=task)
    restore_execution_context(base, spec, plan)
    assert base.review_batch_size == 3
    for changed in (
        dict(plan, review_batch_size=1),
        dict(plan, review_batch_size=True),
        dict(plan, schema_version="multimodal-live-campaign/1.15.0"),
    ):
        with pytest.raises(ValueError, match="saved_report_"):
            restore_execution_context(SimpleNamespace(task=task), spec, changed)


def test_original_campaign_freezes_question_with_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(implementation, "freeze", lambda: {})
    c = object.__new__(MultimodalCampaign)
    c.output, c.campaign_id = tmp_path, "root-initial-test"
    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    sha = SimpleNamespace(checksum=digest("synthetic"))
    c.base = SimpleNamespace(
        manifest=sha,
        visible_value=sha,
        task=task,
        view_checksum=digest("views"),
        sources=SimpleNamespace(initial=[]),
        execution_profile=PROFILE,
        root_direction_review=root_direction_spec(task),
    )
    c.config = configuration(
        directional=True,
        resilient_vision=True,
        exact_primary_status=True,
        execution_profile=PROFILE,
    )
    c.targets = ()
    c.previous_attempts = lambda: []
    plan = c.freeze()
    assert plan["schema_version"] == "multimodal-live-campaign/1.15.0"
    assert plan["root_direction_review"] == root_direction_spec(task)
    assert c.freeze() == plan
    c.base.root_direction_review["desired_direction"] = "support"
    with pytest.raises(ValueError, match="root_question_campaign_binding"):
        c.freeze()
