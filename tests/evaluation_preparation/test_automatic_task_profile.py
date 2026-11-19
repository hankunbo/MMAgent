import json
from copy import deepcopy
from pathlib import Path

import pytest

from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.grounding.models import digest
from tests.dataset_adapters.support import fair

HERE = Path(__file__).parent
REGISTRY = Path("configs/research/multimodal_tasks_v2.json")
TEMPLATE = Path("configs/research/multimodal_catalog_acquisition_v8.json")


@pytest.fixture
def candidate():
    from packages.multimodal_research import automatic_profile, entry

    return automatic_profile, entry


@pytest.mark.parametrize(
    "task_id,registry",
    [
        ("glaucoma_detection", REGISTRY),
        ("early_glaucoma", Path("configs/research/multimodal_propositions_v2.json")),
        ("advanced_glaucoma", Path("configs/research/multimodal_propositions_v2.json")),
    ],
)
def test_original_question_controls_public_context_not_dataset(candidate, task_id, registry):
    helper, _ = candidate
    before = TEMPLATE.read_bytes()
    raw = json.loads(before)
    task, _ = load_intent(task_id, registry)
    value, proof = helper.derive(TEMPLATE, task)
    assert value["definition_expansion"]["public_target"] == task.target_code
    assert value["requirement_catalog"] == raw["requirement_catalog"]
    assert value["measurement_fulfillment"] == raw["measurement_fulfillment"]
    assert value["scope_selection_policy"] == raw["scope_selection_policy"]
    for key in [
        "max_candidates",
        "max_rounds",
        "max_operations",
        "reserve_review_requests",
        "reserve_close_seconds",
    ]:
        assert value["definition_expansion"][key] == raw["definition_expansion"][key]
    if task_id != "advanced_glaucoma":
        body = json.dumps(value["definition_expansion"]["passage_plans"])
        assert "advanced glaucoma" not in body and "severe" not in body
    assert "dataset" not in proof and not proof["execution_authorized"]
    assert TEMPLATE.read_bytes() == before


@pytest.mark.parametrize("explicit_registry", [True, False])
def test_fair_original_prepare_cold_read_and_tamper_boundaries(
    candidate, tmp_path, monkeypatch, explicit_registry
):
    helper, entry = candidate
    root = tmp_path / "data"
    fair(root)
    monkeypatch.setattr(entry, "owners", lambda: pytest.fail("preflight accessed DB"))
    out = tmp_path / "run"
    entry.prepare_entry(
        dataset="fairvlmed",
        root=root,
        sample="data_00000.npz",
        selection="training",
        task="glaucoma_detection",
        output=out,
        **({"registry": REGISTRY} if explicit_registry else {}),
        automatic_requirements=True,
        reuse_procedures=True,
        resilient_vision=True,
        view_profile="detail-v1",
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
    )
    spec = json.loads((out / "entry.json").read_bytes())
    assert spec["automatic_requirement_selection"]["target"] == "GOAL:GLAUCOMA_DETECTION"
    assert "SLO" in spec["intent"]["allowed_modalities"]
    profile = json.loads(Path(spec["active_acquisition"]["path"]).read_bytes())
    assert profile["definition_expansion"]["public_target"] == "GOAL:GLAUCOMA_DETECTION"
    assert entry.assemble_entry(out, preflight_only=True)["status"] == "local_preflight_passed"
    assert entry.assemble_entry(out, preflight_only=True)["api_requests"] == 0
    for fault in ("missing", "task", "query"):
        changed = deepcopy(spec)
        if fault == "missing":
            changed.pop("automatic_requirement_selection")
        elif fault == "task":
            changed["intent"]["target_code"] = "GOAL:ADVANCED_GLAUCOMA"
        else:
            bad = deepcopy(profile)
            bad["definition_expansion"]["public_target"] = "GOAL:ADVANCED_GLAUCOMA"
            Path(spec["active_acquisition"]["path"]).write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="automatic_requirement_"):
            helper.check(changed)
        Path(spec["active_acquisition"]["path"]).write_text(json.dumps(profile))
    assert digest(json.loads((out / "entry.json").read_bytes())) == digest(spec)
