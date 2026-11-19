"""06E G2 references, never patient contracts or execution permission."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Value, check_seal, unique

Text = Annotated[str, Field(min_length=1, max_length=512)]
SampleId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class BenchmarkError(ValueError):
    """A fixed technical reason, not a diagnostic finding."""


class FileReference(Value):
    relative_path: Text
    uri: Text
    checksum: Checksum
    byte_size: int = Field(ge=1, le=16777216)


class TruthReference(Value):
    file: FileReference
    target: Literal["optic_disc", "optic_cup"]
    expert: Literal["exp1", "exp2", "challenge"]
    encoding: Literal["xy_contour", "bmp_gray_labels", "png_gray_labels"]


class BenchmarkSample(Value):
    sample_id: SampleId
    dataset: Literal["PAPILA", "REFUGE2"]
    dataset_version: Text
    group_id: Text
    group_basis: Literal["papila_filename_patient", "unlinked_image_only"]
    split: Literal["development", "reserved"]
    official_split: Literal["unspecified", "train", "val", "test"]
    prior_exposure: Literal["development", "unknown", "unseen_attested"]
    exposure_evidence: FileReference | None = None
    image: FileReference
    dimensions: tuple[int, int]
    laterality: Literal["right", "left", "unknown"]
    truths: tuple[TruthReference, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def coherent(self):
        if min(self.dimensions) < 1 or self.dimensions[0] * self.dimensions[1] > 16777216:
            raise BenchmarkError("image_pixel_limit")
        unique(self.truths, lambda t: (t.target, t.expert))
        if any(
            t.file.uri == self.image.uri or t.file.checksum == self.image.checksum
            for t in self.truths
        ):
            raise BenchmarkError("truth_in_model_input")
        if self.split == "reserved" and self.prior_exposure == "development":
            raise BenchmarkError("development_cannot_be_reserved")
        if self.prior_exposure == "unseen_attested" and self.exposure_evidence is None:
            raise BenchmarkError("unseen_requires_evidence")
        if self.dataset == "PAPILA":
            if self.group_basis != "papila_filename_patient" or self.laterality == "unknown":
                raise BenchmarkError("papila_identity_missing")
            if {(t.target, t.expert, t.encoding) for t in self.truths} != {
                (target, expert, "xy_contour")
                for target in ("optic_disc", "optic_cup")
                for expert in ("exp1", "exp2")
            }:
                raise BenchmarkError("papila_expert_truth_incomplete")
        return self


class LabelDefinition(Value):
    version: Text
    source: FileReference
    source_excerpt: Text
    background: int = Field(ge=0, le=255)
    disc_rim: int = Field(ge=0, le=255)
    cup: int = Field(ge=0, le=255)
    disc_includes_cup: Literal[True] = True

    @model_validator(mode="after")
    def distinct(self):
        if len({self.background, self.disc_rim, self.cup}) != 3:
            raise BenchmarkError("label_values_must_be_distinct")
        return self


class BenchmarkManifest(Value):
    schema_version: Literal["tool-benchmark/1.0.0"] = "tool-benchmark/1.0.0"
    checksum: Checksum
    intended_use: Literal["research_only"] = "research_only"
    scope: Literal["tool_benchmark"] = "tool_benchmark"
    samples: tuple[BenchmarkSample, ...] = Field(min_length=1, max_length=32)
    conversion: Literal["xy_integer_even_odd/1.0.0"] = "xy_integer_even_odd/1.0.0"
    aggregation: Literal["separate_experts_macro_mean/1.0.0"] = "separate_experts_macro_mean/1.0.0"
    label_definition: LabelDefinition | None = None
    formal_evaluation_locked: Literal[True] = True

    @model_validator(mode="after")
    def coherent(self):
        unique(self.samples, lambda s: (s.dataset, s.sample_id))
        groups, images = {}, {}
        for sample in self.samples:
            key = (sample.dataset, sample.group_id)
            if key in groups and groups[key] != sample.split:
                raise BenchmarkError("patient_group_split_leakage")
            groups[key] = sample.split
            if sample.image.checksum in images:
                raise BenchmarkError("duplicate_image_content")
            images[sample.image.checksum] = sample.split
            if sample.dataset == "REFUGE2" and self.label_definition is None:
                raise BenchmarkError("refuge_label_definition_unverified")
        check_seal(self)
        return self


class ModelInput(Value):
    """Only the image crosses the model boundary; no truth/diagnosis/prompt labels."""

    schema_version: Literal["benchmark-image-input/1.0.0"] = "benchmark-image-input/1.0.0"
    benchmark_checksum: Checksum
    sample_id: SampleId
    image: FileReference
    dimensions: tuple[int, int]
    laterality: Literal["right", "left", "unknown"]
    purpose: Literal["shadow"] = "shadow"
    decisive_evidence: Literal[False] = False


class EvaluationPolicy(Value):
    version: Literal["benchmark-policy/1.0.0"] = "benchmark-policy/1.0.0"
    baseline_only: bool = True
    min_dice: float | None = Field(default=None, ge=0, le=1)
    max_failure_rate: float | None = Field(default=None, ge=0, le=1)
    min_samples: int | None = Field(default=None, ge=1, le=100000)
    statistics: Literal["patient_macro_mean/1.0.0"] = "patient_macro_mean/1.0.0"
    confidence_interval: Literal["not_implemented"] = "not_implemented"
    required_gates: tuple[Literal[0, 1, 2, 3, 4, 5, 6], ...] = (0, 1, 2, 3, 4, 5, 6)
    max_samples: int = Field(default=4, ge=1, le=32)
    max_input_bytes: int = Field(default=4194304, ge=1, le=4194304)
    max_output_bytes: int = Field(default=4194304, ge=1, le=4194304)
    max_seconds_per_sample: int = Field(default=300, ge=1, le=3600)

    @model_validator(mode="after")
    def all_gates(self):
        if self.required_gates != (0, 1, 2, 3, 4, 5, 6):
            raise BenchmarkError("mandatory_gates_cannot_be_removed")
        return self


class EvaluationFreeze(Value):
    """Required offline bindings before any future formal evaluator can be considered.

    This artifact is a content lock, never permission to execute or publish a catalog.
    """

    schema_version: Literal["benchmark-evaluation-freeze/1.0.0"] = (
        "benchmark-evaluation-freeze/1.0.0"
    )
    checksum: Checksum
    manifest_checksum: Checksum
    policy_checksum: Checksum
    metrics_code_checksum: Checksum
    statistics_code_checksum: Checksum
    conversion_code_checksum: Checksum
    catalog_checksum: Checksum
    implementation_checksum: Checksum
    model_source_checksum: Checksum
    weights_checksum: Checksum
    preprocessing_checksum: Checksum
    postprocessing_checksum: Checksum
    runtime_closure_checksum: Checksum
    evaluation_samples: tuple[SampleId, ...] = Field(min_length=1, max_length=32)
    execution_authorized: Literal[False] = False

    @model_validator(mode="after")
    def integrity(self):
        unique(self.evaluation_samples)
        check_seal(self)
        return self
