"""Explicit selected-input versions; old declarations and stored bytes are unchanged."""

from typing import Literal

from pydantic import Field, model_validator

from packages.disease_packs.models import SealedDeclaration

from .models import FileBinding, ObjectReference, Value, VisibleManifest


class RuntimeInputProfile(Value):
    version: Literal["fairvlmed-selected/1.0.0"] = "fairvlmed-selected/1.0.0"
    sample: str = Field(pattern=r"^[A-Za-z0-9_-]+\.npz$")
    source_files: tuple[FileBinding, ...]
    missing_optional_fields: tuple[str, ...]
    quarantined_field_names: tuple[str, ...]
    image_transform: Literal["lossless_uint8_grayscale_png"] = "lossless_uint8_grayscale_png"
    unknown_field_policy: Literal["quarantine_not_visible"] = "quarantine_not_visible"


class RuntimeVisibleManifest(VisibleManifest):
    schema_version: Literal["visible-input/2.0.0"] = "visible-input/2.0.0"
    adapter_version: Literal["dataset-adapters/2.0.0"] = "dataset-adapters/2.0.0"
    input_profile: RuntimeInputProfile

    @model_validator(mode="after")
    def visible_boundary(self):
        if self.dataset_ref.code != "SOURCE:DATASET:FAIRVLMED":
            raise ValueError("runtime_dataset_not_registered")
        # Same original image/identity/leakage boundary. Only absent notes are new.
        VisibleManifest.visible_boundary(self.model_copy(update={"text_review": "pending"}))
        return self


class RuntimeRegistration(SealedDeclaration):
    schema_version: Literal["dataset-registration/3.0.0"] = "dataset-registration/3.0.0"
    preparation_version: Literal["prepared-input/3.0.0"] = "prepared-input/3.0.0"
    fingerprint_version: Literal["registration-fingerprint/3.0.0"] = (
        "registration-fingerprint/3.0.0"
    )
    visible: RuntimeVisibleManifest
    restricted_manifest: ObjectReference


def read_visible(raw):
    import json

    from .multimodal.storage import MultimodalVisible

    model = {
        "visible-input/1.0.0": VisibleManifest,
        "visible-input/2.0.0": RuntimeVisibleManifest,
        "visible-input/3.0.0": MultimodalVisible,
    }.get(json.loads(raw).get("schema_version"))
    if model is None:
        raise ValueError("visible_input_version_unsupported")
    return model.model_validate_json(raw)
