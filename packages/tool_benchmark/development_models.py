"""Explicit development reanalysis contract; never unlocks the frozen 06E v1 test gate."""

from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Value, check_seal, unique

from .models import FileReference, SampleId


class DevelopmentSample(Value):
    sample_id: SampleId
    data: FileReference
    prediction: FileReference
    input_png: FileReference
    result: FileReference
    patient_id: None = None
    laterality: Literal["unknown"] = "unknown"
    prior_exposure: Literal["development"] = "development"


class DevelopmentProtocol(Value):
    schema_version: Literal["fairseg-development/1.0.0"] = "fairseg-development/1.0.0"
    checksum: Checksum
    purpose: Literal["development_reanalysis"] = "development_reanalysis"
    dataset: Literal["FairSeg"] = "FairSeg"
    model: Literal["segman_slo_disc_cup_segmenter"] = "segman_slo_disc_cup_segmenter"
    mode: Literal["reuse_saved_predictions"] = "reuse_saved_predictions"
    official_split: Literal["Test"] = "Test"
    metric_version: Literal["native-mask-geometry/1.0.0"] = "native-mask-geometry/1.0.0"
    prepared_at: str
    dataset_root: str
    prediction_root: str
    sources: dict[str, Checksum]
    implementation: dict[str, Checksum]
    historical_manifest: FileReference
    historical_predictions: FileReference
    label_semantics: Literal["raw_0_background_minus1_rim_minus2_cup"] = (
        "raw_0_background_minus1_rim_minus2_cup"
    )
    samples: tuple[DevelopmentSample, ...] = Field(min_length=1, max_length=8)
    strata_fields: tuple[Literal["gender", "race"], ...] = ("gender", "race")
    min_public_stratum: Literal[3] = 3
    confidence_interval: Literal["not_estimated_independence_unverified"] = (
        "not_estimated_independence_unverified"
    )
    formal_evaluation_locked: Literal[True] = True
    max_seconds: Literal[120] = 120
    new_model_calls: Literal[0] = 0
    tool_qualification_updated: Literal[False] = False

    @model_validator(mode="after")
    def coherent(self):
        unique(self.samples, lambda s: s.sample_id)
        unique(self.samples, lambda s: s.data.checksum)
        check_seal(self)
        return self
