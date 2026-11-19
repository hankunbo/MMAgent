"""Independent dataset development preparation, not an execution or diagnosis port."""

import json
from pathlib import Path

from packages.dataset_adapters.papila import PapilaClinicalPreparation
from packages.dataset_adapters.prepared import capture
from packages.grounding.models import digest
from packages.object_store import checksum

from .service import adapter_for


def save(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        path.chmod(0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def fair_case(adapter, sample_id):
    """Use the original owner/capture; never project source notes or label-bearing rows."""
    snap = capture(adapter.prepare(sample_id, split="training"))
    image, shape, eye, modality = snap.images[0]
    fields = snap.fields["npz"]
    visible = dict(
        schema_version="fairvlmed-development-input/1.0.0",
        sample_id=sample_id,
        dataset="FairVLMed",
        modality=modality,
        laterality=eye,
        dimensions=shape,
        image_checksum=checksum(image),
        demographics={
            k: fields[k]
            for k in ("age", "gender", "race", "ethnicity", "language", "maritalstatus")
        },
        demographics_encoding="dataset_native_requires_original_owner_mapping",
        clinical_observations=[],
        patient_group=None,
        patient_independence_proven=False,
        runtime_admission=False,
        final_test_unlocked=False,
    )
    restricted = dict(
        purpose="restricted_evaluation_only",
        sample_id=sample_id,
        source_files=[f.model_dump(mode="json") for f in snap.files],
        prepared_fingerprint=snap.fingerprint(),
        labels=dict(npz=fields["glaucoma"], csv=snap.fields["csv"]["glaucoma"]),
        note_withheld=True,
        original_note_copied=False,
    )
    return visible, restricted, image


def select_fair(adapter, *, excluded, count=2):
    if type(count) is not int or not 1 <= count <= 10:
        raise ValueError("development_count_invalid")
    chosen = [n for n in adapter.candidates(count=50, split="training") if n not in excluded]
    if len(chosen) < count:
        raise ValueError("development_candidates_insufficient")
    return chosen[:count]


def prepare(fair_root, papila_root, output):
    """2 SLO images + 2 PAPILA patients, filename-only selection, no label stratification."""
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    visible_dir, restricted_dir = output / "visible", output / "restricted"
    visible_dir.mkdir(mode=0o700)
    restricted_dir.mkdir(mode=0o700)
    # This is a narrow known-exposure list, not a claim of complete history auditing.
    exclusions = {
        "FairVLMed": {"data_00001.npz": "13A probe", "data_06231.npz": "historical first case"},
        "PAPILA": {"RET002": "13A probe patient"},
    }
    adapter = adapter_for("fairvlmed", fair_root)
    selected = select_fair(adapter, excluded=exclusions["FairVLMed"])
    entries = []
    for name in selected:
        visible, hidden, image = fair_case(adapter, name)
        stem = "fairvlmed_" + Path(name).stem
        image_path = visible_dir / (stem + ".png")
        with image_path.open("xb") as stream:
            image_path.chmod(0o600)
            stream.write(image)
        visible["image_file"] = image_path.name
        save(visible_dir / (stem + ".json"), visible)
        save(restricted_dir / (stem + ".json"), hidden)
        entries.append(
            dict(
                dataset="FairVLMed",
                sample_id=name,
                visible=stem + ".json",
                image_checksum=checksum(image),
                profiles=["image_demographics"],
            )
        )
    papila = PapilaClinicalPreparation(papila_root)
    inventory = papila.inventory()
    patients = sorted({s[:6] for s in inventory["sample_ids"]} - set(exclusions["PAPILA"]))[:2]
    if len(patients) != 2:
        raise ValueError("papila_development_patients_insufficient")
    for patient in patients:
        for eye in ("OD", "OS"):
            sample_id = patient + eye
            for profile in ("image_demographics", "image_demographics_clinical"):
                projected = papila.prepare(sample_id, profile=profile)
                save(
                    visible_dir / f"papila_{sample_id}_{profile}.json",
                    projected.model_dump(mode="json"),
                )
            save(
                restricted_dir / f"papila_{sample_id}.json", papila.restricted_diagnosis(sample_id)
            )
            entries.append(
                dict(
                    dataset="PAPILA",
                    sample_id=sample_id,
                    patient_group="PAPILA:" + patient,
                    image_checksum=projected.image.checksum,
                    profiles=["image_demographics", "image_demographics_clinical"],
                )
            )
    save(output / "papila-inventory.json", inventory)
    protocol = dict(
        schema_version="new-case-development-preparation/1.0.0",
        status="inputs_prepared",
        selection="filename_order_after_known_exclusions",
        selection_uses_labels=False,
        exclusions=exclusions,
        exclusion_scope="known_first_case_and_13a_probes_not_full_history",
        entries=entries,
        real_requests=0,
        model_executions=0,
        runtime_writes=0,
        clinical_decisions=0,
        final_test_unlocked=False,
        input_profiles_not_independent_cases=True,
        patient_independence=dict(
            PAPILA="local_patient_ids_group_both_eyes", FairVLMed="unresolved"
        ),
        next_gate="new_case_runtime_assembly_and_rgb_tool_applicability",
        pending=[
            "per_case_fresh_observations_and_graph",
            "bounded_runtime_entry_for_new_inputs",
            "formal_evaluation_and_review_protocol",
            "upstream_archive_byte_equivalence",
        ],
    )
    files = {
        str(p.relative_to(output)): checksum(p.read_bytes())
        for p in output.rglob("*")
        if p.is_file()
    }
    protocol["input_files"] = files
    protocol["input_set_checksum"] = digest(files)
    save(output / "protocol.json", protocol)
    return dict(
        status="inputs_prepared_not_executed",
        fairvlmed_images=len(selected),
        papila_patients=len(patients),
        papila_eyes=2 * len(patients),
        api_requests=0,
        runtime_writes=0,
        final_test_unlocked=False,
    )
