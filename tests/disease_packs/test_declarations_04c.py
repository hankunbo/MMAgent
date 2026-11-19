import json
from pathlib import Path

import pytest

from packages.disease_packs import DiseasePack, DiseasePackCatalog, seal, validate_conclusion
from tests.disease_packs.support import ROOT, declarations

CONFIGS = ROOT / "configs/disease_packs"


def real_packs():
    return tuple(
        DiseasePack.model_validate_json(p.read_bytes()) for p in sorted(CONFIGS.glob("*.json"))
    )


def test_machine_declarations_and_schema_snapshots():
    from packages.disease_packs import DatasetReference, PackDefinition

    catalog = DiseasePackCatalog.from_directory(CONFIGS)
    for pack in real_packs():
        assert catalog.pack(pack.reference()) == pack
        assert pack.status == "Draft" and pack.clinical_level == "pending_L4"
        assert pack.bindings is None and pack.knowledge_gaps
        for ref in pack.definition.dataset_refs:
            ds = catalog.dataset(ref)
            assert ds.label_purpose == "evaluation_only_not_publication"
            if ds.owner == "06E":
                assert ds.visibility.planner_visible == ()
    for model in (DiseasePack, DatasetReference, PackDefinition):
        assert (
            json.loads(
                (
                    ROOT / "packages/disease_packs/schemas" / f"{model.__name__}.schema.json"
                ).read_text()
            )
            == model.model_json_schema()
        )


@pytest.mark.parametrize(
    "field",
    [
        "glaucoma",
        "gpt4_summary",
        "note",
        "note_extra",
        "age",
        "gender",
        "md",
        "tds",
        "diagnosis_or_assessment_text",
        "expert_reference_graph",
        "use",
        "filename",
        "unknown",
        "evidence_note_redacted.extra",
    ],
)
def test_hidden_and_unknown_fields_never_planner_visible(field):
    catalog = DiseasePackCatalog.from_directory(CONFIGS)
    pack = next(p for p in real_packs() if p.definition.pack_code.startswith("fairvlmed"))
    with pytest.raises(ValueError, match="Planner field"):
        catalog.validate_planner_fields(pack.definition.dataset_refs[0], (field,))


def test_redaction_is_an_owned_precondition_not_an_04c_adapter():
    catalog = DiseasePackCatalog.from_directory(CONFIGS)
    pack = next(p for p in real_packs() if p.definition.pack_code.startswith("fairvlmed"))
    ref = pack.definition.dataset_refs[0]
    with pytest.raises(ValueError, match="07A"):
        catalog.validate_planner_fields(ref, ("evidence_note_redacted",))
    assert catalog.validate_planner_fields(
        ref, ("evidence_note_redacted",), reviewed_derived_fields=("evidence_note_redacted",)
    ) == ("evidence_note_redacted",)


@pytest.mark.parametrize(
    "conclusion",
    [
        "definitive_poag_diagnosis",
        "treatment_order",
        "infer_oct_from_slo",
        "infer_visual_field_from_slo",
        "infer_angle_from_slo",
        "any_new_diagnosis",
        {"diagnosis": "POAG"},
    ],
)
def test_forbidden_conclusions(conclusion):
    for pack in real_packs():
        with pytest.raises(ValueError):
            validate_conclusion(pack, conclusion)
        assert validate_conclusion(pack, "insufficient") == "insufficient"


def test_false_cross_dataset_patient_relation():
    catalog = DiseasePackCatalog.from_directory(CONFIGS)
    packs = real_packs()
    refs = tuple({r.code: r for p in packs for r in p.definition.dataset_refs}.values())
    for left in refs:
        for right in refs:
            if left != right or catalog.dataset(left).owner == "06E":
                with pytest.raises(ValueError, match="patient relationship"):
                    catalog.validate_patient_relation(left, right)


def test_checksum_bypass_and_fake_publication_rejected():
    pack = real_packs()[0]
    bad = pack.model_copy(update={"status": "Published"})
    with pytest.raises(ValueError):
        DiseasePack.model_validate(bad)
    data = pack.model_dump(mode="json")
    data["definition"]["target_proposition"] = "POAG diagnosis"
    with pytest.raises(ValueError):
        DiseasePack.model_validate_json(json.dumps(data))
    with pytest.raises(ValueError):
        seal(
            DiseasePack,
            **(pack.model_dump(mode="python", exclude={"checksum"}) | {"status": "Published"}),
        )


def test_exact_versions_no_latest_and_no_mixed_real_synthetic():
    pack = real_packs()[0]
    catalog = DiseasePackCatalog.from_directory(CONFIGS)
    with pytest.raises(KeyError):
        catalog.pack(pack.reference().model_copy(update={"version": "9.0.0"}))
    with pytest.raises(ValueError):
        catalog.pack(pack.reference().model_copy(update={"checksum": "sha256:" + "a" * 64}))
    ds, definition = declarations()
    assert ds.local_root is None and definition.synthetic
    with pytest.raises(ValueError, match="duplicate"):
        DiseasePackCatalog(
            (pack, pack), tuple(catalog.dataset(r) for r in pack.definition.dataset_refs)
        )


def test_no_dataset_or_model_access(monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert "/download/" not in str(path) and "/hf_cache/" not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    DiseasePackCatalog.from_directory(CONFIGS)
    import ast

    for folder in ("packages/disease_packs", "apps"):
        files = (
            (ROOT / folder).glob("*.py")
            if folder.startswith("packages")
            else [ROOT / folder / "ophthalmology_disease_packs.py"]
        )
        for path in files:
            tree = ast.parse(path.read_text())
            imported = {
                n.module.split(".")[0]
                for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module
            }
            imported |= {
                a.name.split(".")[0]
                for n in ast.walk(tree)
                if isinstance(n, ast.Import)
                for a in n.names
            }
            assert not imported & {"numpy", "torch", "PIL", "cv2", "openai", "pandas", "requests"}


@pytest.mark.parametrize(
    "channel", ["report_uri", "extension", "label_observation", "nested_observation", "goal"]
)
def test_prepared_case_contract_rejects_label_channels(channel):
    from packages.contracts import CaseBundle
    from packages.disease_packs.catalog import validate_prepared_case_contract
    from tests.disease_packs.support import anonymous_case

    pack = real_packs()[0]
    case = anonymous_case(pack.definition.target_node_code)
    assert validate_prepared_case_contract(pack, case) == case
    data = case.model_dump(mode="json")
    if channel == "report_uri":
        data["report_uri"] = "object://synthetic/raw-note"
    elif channel == "extension":
        data["extensions"] = {"glaucoma": 1}
    elif channel == "goal":
        data["goals"][0]["goal_code"] = "GOAL:POAG_DIAGNOSIS"
    else:
        data["observations"] = [
            dict(
                concept_code="CONCEPT:GLAUCOMA_LABEL"
                if channel == "label_observation"
                else pack.definition.evidence_slots[0].observation_code,
                observation="POSITIVE",
                value=1 if channel == "label_observation" else {"glaucoma": 1},
                unit=None,
                source_code="SOURCE:SYNTHETIC",
                observed_at=None,
            )
        ]
    with pytest.raises(ValueError):
        validate_prepared_case_contract(pack, CaseBundle.model_validate_json(json.dumps(data)))


def test_seal_does_not_discard_unknown_inputs():
    pack = real_packs()[0]
    fields = pack.model_dump(mode="python", exclude={"checksum"})
    with pytest.raises(ValueError, match="declared fields"):
        seal(DiseasePack, **fields, unregistered_clinical_threshold=0.8)


def test_mask_requirement_keeps_pixel_coordinates_and_no_fake_benchmark():
    from packages.contracts import GECEGInstance
    from packages.disease_packs.gaps import missing_capability

    pack = next(p for p in real_packs() if p.definition.pack_code.startswith("fairvlmed"))
    slot = next(
        s
        for s in pack.definition.evidence_slots
        if s.output_artifact_code == "ARTIFACT:DISC_CUP_MASK"
    )
    data = json.loads(
        (ROOT / "packages/contracts/examples/v1/GECEGInstance.valid.json").read_text()
    )
    old = next(n["node_code"] for n in data["nodes"] if n["node_type"] == "EVIDENCE")
    for node in data["nodes"]:
        if node["node_code"] == old:
            node["node_code"] = slot.node_code
    for edge in data["edges"]:
        for key in ("source_node_code", "target_node_code"):
            if edge[key] == old:
                edge[key] = slot.node_code
    graph = GECEGInstance.model_validate_json(json.dumps(data))
    gap, requirement = missing_capability(pack, graph, slot)
    assert gap.clinical_effect.value == "NONE"
    assert requirement.required_input_contract[0].coordinate_space.value == "PIXEL_2D"
    assert requirement.required_output_contract[0].coordinate_space.value == "PIXEL_2D"
    assert requirement.validation_plan.frozen_datasets == ()
    assert requirement.validation_plan.threshold_policy is None
