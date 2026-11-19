"""Versioned development candidate selection, with no invented clinical truth."""

from typing import Literal

from packages.disease_packs.models import SealedDeclaration, seal

from .models import FileBinding


class GoldenCandidate(SealedDeclaration):
    schema_version: Literal["golden-candidate/1.0.0"] = "golden-candidate/1.0.0"
    candidate_id: str
    level: Literal["G3", "G4"]
    split: Literal["training"] = "training"
    source_files: tuple[FileBinding, ...]
    input_fingerprint: str
    selection_version: Literal["training-ordered-dev/1.0.0"] = "training-ordered-dev/1.0.0"
    selection_bias: Literal["ordered_engineering_baseline_not_representative"] = (
        "ordered_engineering_baseline_not_representative"
    )
    reference_status: Literal["pending"] = "pending"
    expected_template_version: None = None
    allowed_actions: tuple[Literal["inspect_visible_inputs", "report_gap", "safe_stop"], ...] = (
        "inspect_visible_inputs",
        "report_gap",
        "safe_stop",
    )
    tool_dag: tuple[()] = ()
    automatic_assertions: tuple[
        Literal["unique_source_alignment", "checksum", "contract_readback"], ...
    ] = ("unique_source_alignment", "checksum", "contract_readback")
    expert_expected_evidence: None = None
    forbidden_conclusions: tuple[Literal["definitive_poag_diagnosis", "treatment_order"], ...] = (
        "definitive_poag_diagnosis",
        "treatment_order",
    )
    required_reviews: tuple[
        Literal[
            "visibility",
            "node_edge_reference",
            "image_quality",
            "stop_rules",
            "coverage_and_grouping",
        ],
        ...,
    ] = (
        "visibility",
        "node_edge_reference",
        "image_quality",
        "stop_rules",
        "coverage_and_grouping",
    )


class BoundGoldenCandidate(GoldenCandidate):
    schema_version: Literal["golden-candidate/2.0.0"] = "golden-candidate/2.0.0"
    fingerprint_version: Literal["registration-fingerprint/2.0.0"] = (
        "registration-fingerprint/2.0.0"
    )


def prepare_candidates(adapter, *, count=20):
    if not 20 <= count <= 50:
        raise ValueError("development_candidates_require_20_to_50")
    names = adapter.candidates(count=count)
    if len(names) != count:
        raise ValueError("insufficient_training_candidates")
    result = []
    for name in names:
        sample = adapter.prepare(name)
        result.append(
            seal(
                BoundGoldenCandidate,
                candidate_id="CANDIDATE:" + sample.fingerprint()[7:39].upper(),
                level="G3" if sample.dataset.name == "FairVLMed" else "G4",
                source_files=sample.files,
                input_fingerprint=sample.fingerprint(),
            )
        )
    return tuple(result)
