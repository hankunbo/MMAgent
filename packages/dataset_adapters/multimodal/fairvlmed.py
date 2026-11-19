"""Bridge the original selected FairVLMed reader into shared component storage.

No second NPZ parser or hidden clinical-note projection. The original owner
validates training membership and framing; only its lossless SLO and age-only
projection become visible components. This is not a method execution grant.
"""

from packages.dataset_adapters.prepared import capture
from packages.dataset_adapters.public_context import POLICY, age_observation
from packages.dataset_adapters.runtime_input import factory

from .adapters import Builder
from .models import Component, SourceFile

ADAPTER = "multimodal-adapters/1.3.0"


def prepare(root, sample):
    owner = factory("fairvlmed", root)
    selected = owner.prepare(sample, split="training")
    snapshot = capture(selected, owner.catalog)
    b = Builder(root)
    b.files = {
        f.relative_path: SourceFile(
            path=f.relative_path, checksum=f.checksum, byte_size=f.byte_size
        )
        for f in snapshot.files
    }
    path = "Dataset/Training/" + sample
    raw, shape, eye, modality = snapshot.images[0]
    locator = path + "#slo_fundus"
    pid = b.payload("slo_fundus", raw, path, locator, "image/png", "selected_observation")
    b.components.append(
        Component(
            id="slo_fundus",
            kind="image",
            concept="SLO",
            modality=modality,
            status="obtained",
            shape=shape,
            dtype="uint8_grayscale_png",
            payload_ids=(pid,),
            source_paths=(path,),
            source_locator=locator,
            laterality=eye,
            axis_semantics="source_index_only",
            semantic_limits=(
                "lossless_uint8_grayscale_projection_no_color_information",
                "visible_anatomy_requires_current_input_scope_observation",
                "no_physical_scale_or_eye_pairing_supplied",
            ),
        )
    )
    age = age_observation(snapshot)
    b.components.append(
        Component(
            id="age",
            kind="scalar",
            concept="age",
            modality="structured_measurement",
            status=age["status"],
            value=age["value"],
            reason=age["reason"],
            unit=age["unit"],
            unit_basis=POLICY + ":npz.age",
            source_paths=(path,),
            source_locator=path + "#age",
            axis_semantics="not_applicable",
        )
    )
    keys = {m["key"] for m in snapshot.fields["npz_metadata"]}
    hidden = set(snapshot.fields["csv"]) | keys
    b.quarantine = hidden - {
        "filename",
        "use",
        "slo_fundus",
        "age",
        "glaucoma",
        "note",
        "gpt4_summary",
    }
    return b.finish(
        dataset="fairvlmed",
        adapter_version=ADAPTER,
        sample=sample,
        selection="training",
        task="glaucoma_detection",
        release_basis="SOURCE:DATASET:FAIRVLMED:" + snapshot.dataset.checksum,
        excluded_fields=("glaucoma", "note", "gpt4_summary", "csv.age"),
        limitations=(
            "original_selected_input_owner_training_membership_checked",
            "SLO_is_not_color_fundus_or_raw_OCT",
            "original_age_only_policy_no_patient_identity_inferred",
            "hidden_labels_and_notes_not_visible",
            "research_input_preparation_not_clinical_acceptance",
        ),
    )
