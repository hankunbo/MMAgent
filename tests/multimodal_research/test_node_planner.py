import json
from types import SimpleNamespace

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.capabilities import from_preparation
from packages.dataset_adapters.multimodal.preparation import persist
from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.multimodal_research.requirement_catalog import read as read_catalog
from packages.multimodal_research.node_planner import run
from tests.dataset_adapters.test_multimodal import gamma


def test_reuse_plan_is_initial_tree_only(tmp_path):
    root = gamma(tmp_path / "gamma")
    prepared = prepare("gamma", root, "0001")
    input_path = persist(prepared, tmp_path / "prepared")
    result = run(
        input_path=input_path,
        task_id="advanced_glaucoma",
        output=tmp_path / "plan",
        registry="configs/research/multimodal_propositions_v1.json",
        catalog_profile="configs/research/multimodal_catalog_acquisition_v2.json",
    )
    plan = json.loads((tmp_path / "plan" / "plan.json").read_bytes())
    assert result["execution_authorized"] is False
    assert plan["api_requests"] == 0
    assert plan["model_runs"] == 0
    assert plan["database_writes"] == 0
    assert plan["qualification"].startswith("initial_plan_only")
    assert any(n["type"] == "proposition" for n in plan["tree"]["nodes"])
    assert any(n["type"] == "examination" for n in plan["tree"]["nodes"])
    assert all("score" not in n for n in plan["tree"]["nodes"])


def test_reuse_plan_keeps_missing_input_as_gap(tmp_path):
    root = gamma(tmp_path / "gamma", photo=False)
    prepared = prepare("gamma", root, "0001")
    input_path = persist(prepared, tmp_path / "prepared")
    run(
        input_path=input_path,
        task_id="advanced_glaucoma",
        output=tmp_path / "plan",
        registry="configs/research/multimodal_propositions_v1.json",
        catalog_profile="configs/research/multimodal_catalog_acquisition_v2.json",
    )
    plan = json.loads((tmp_path / "plan" / "plan.json").read_bytes())
    assert any(g["state"] == "missing_input" for g in plan["gaps"])
    assert plan["execution_authorized"] is False


def test_matching_modality_is_selected_before_qualitative_review(tmp_path):
    root = gamma(tmp_path / "gamma")
    prepared = prepare("gamma", root, "0001")
    input_path = persist(prepared, tmp_path / "prepared")
    run(
        input_path=input_path,
        task_id="advanced_glaucoma",
        output=tmp_path / "plan",
        registry="configs/research/multimodal_propositions_v1.json",
        catalog_profile="configs/research/multimodal_catalog_acquisition_v2.json",
    )
    plan = json.loads((tmp_path / "plan" / "plan.json").read_bytes())
    visual = next(n for n in plan["tree"]["nodes"] if n.get("check_id") == "cup_geometry")
    assert visual["status"] == "selected"
    assert "qualitative_scope_pending" in visual["pending_conditions"]
    # A single-eye fixture still leaves the explicitly bilateral check
    # conditional; ordinary visual checks are not blocked by absent scope QC.
    assert plan["counts"]["conditional"] == 1
    assert not any(g["check"] == "cup_geometry" for g in plan["gaps"])
    assert any(g["check"] == "cup_geometry" for g in plan["pending_conditions"])


def test_memory_hint_is_procedure_only(tmp_path):
    root = gamma(tmp_path / "gamma")
    prepared = prepare("gamma", root, "0001")
    input_path = persist(prepared, tmp_path / "prepared")
    memory = tmp_path / "memory.json"
    memory.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "key": "memory:cup-001",
                        "kind": "tool_experience",
                        "case_code": "old-case",
                        "detail": {
                            "target": "cup_geometry",
                            "modality": "color_fundus",
                            "steps": [
                                {"operation": "read_input", "tool": "old-tool"},
                                {"operation": "review", "tool": "old-tool"},
                            ],
                            "old_patient_value": 0.91,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    run(
        input_path=input_path,
        task_id="advanced_glaucoma",
        output=tmp_path / "plan",
        registry="configs/research/multimodal_propositions_v1.json",
        catalog_profile="configs/research/multimodal_catalog_acquisition_v2.json",
        memory_index=memory,
    )
    plan = json.loads((tmp_path / "plan" / "plan.json").read_bytes())
    cup = next(n for n in plan["tree"]["nodes"] if n.get("check_id") == "cup_geometry")
    assert cup["method_source"] == "memory_procedure_hint"
    assert any(n["type"] == "method" for n in plan["tree"]["nodes"])
    assert "old_patient_value" not in json.dumps(plan, ensure_ascii=False)


def test_selection_uses_capabilities_not_dataset_name(tmp_path):
    root = gamma(tmp_path / "gamma")
    prepared = prepare("gamma", root, "0001")
    capabilities = from_preparation(persist(prepared, tmp_path / "prepared"))
    task, _ = load_intent(
        "advanced_glaucoma", "configs/research/multimodal_propositions_v1.json"
    )
    profile = json.loads(
        open("configs/research/multimodal_catalog_acquisition_v2.json", encoding="utf-8").read()
    )
    catalog = read_catalog(profile["requirement_catalog"])
    first = __import__("packages.multimodal_research.node_planner", fromlist=["build_plan"])
    build = first.build_plan
    common = dict(capabilities=capabilities, task=task, catalog=catalog)
    a = build(
        manifest=SimpleNamespace(checksum=prepared.manifest.checksum, dataset="gamma"),
        **common,
    )
    b = build(
        manifest=SimpleNamespace(checksum=prepared.manifest.checksum, dataset="other_dataset"),
        **common,
    )
    assert a["tree"] == b["tree"]
    without_fundus = capabilities.model_copy(
        update={"components": tuple(c for c in capabilities.components if c.modality != "color_fundus")}
    )
    c = build(
        manifest=SimpleNamespace(checksum=prepared.manifest.checksum, dataset="gamma"),
        capabilities=without_fundus,
        task=task,
        catalog=catalog,
    )
    status_a = next(n for n in a["tree"]["nodes"] if n.get("check_id") == "cup_geometry")["status"]
    status_c = next(n for n in c["tree"]["nodes"] if n.get("check_id") == "cup_geometry")["status"]
    assert status_a != status_c and status_c == "missing_input"
