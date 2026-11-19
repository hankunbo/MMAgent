"""V4 source-specific decisions, separate from acquiring missing patient inputs."""

from typing import Literal

from pydantic import Field, model_validator

from packages.evidence_details.models import Strict
from packages.evidence_scoring.participation_models import ParticipationScores


class LiteratureReference(Strict):
    kind: Literal["source_version"]
    code: str
    version: str
    checksum: str


class Reading(Strict):
    id: str
    title: str
    url: str
    reference: LiteratureReference | None
    material_checksum: str
    reading_scope: Literal[
        "abstract", "selected_pdf_pages_24_27", "full_text_methods_excerpt_not_full_results"
    ]
    receipt: str


class Adjudication(Strict):
    assessment_id: str
    state: Literal[
        "direction_reviewed",
        "method_mismatch",
        "reference_missing",
        "context_only",
        "ungradable",
        "input_missing",
        "duplicate_or_intermediate",
        "not_reviewed",
    ]
    comparison: str = Field(min_length=1, max_length=2500)
    reference_interpretation: str = Field(min_length=1, max_length=2500)
    unresolved: str = Field(min_length=1, max_length=2500)
    next_action: str = Field(min_length=1, max_length=2500)
    source_ids: list[str]
    assessment_checksum: str
    source_binding_checksum: str
    review_completed: bool


class ImageReview(Strict):
    assessment_id: str
    parent_assessment_id: str
    observer: Literal["assistant_direct_image_inspection_not_deployed_vlm"]
    input_checksum: str
    # Half-open bounding box in original image pixels, not anatomical orientation.
    region_xyxy: list[int] = Field(min_length=4, max_length=4)
    image_size_wh: list[int] = Field(min_length=2, max_length=2)
    observed: str
    unresolved: str
    source_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def bounds(self):
        x1, y1, x2, y2 = self.region_xyxy
        w, h = self.image_size_wh
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError("interpretation_image_region")
        return self


class InterpretationScores(ParticipationScores):
    schema_version: Literal["case-research-scoring/4.0.0"]
    policy_version: Literal["argument-interpretation/4.0.0"]
    readings: list[Reading] = Field(max_length=32)
    adjudications: list[Adjudication] = Field(max_length=512)
    image_reviews: list[ImageReview] = Field(max_length=16)

    @model_validator(mode="after")
    def adjudication_inventory(self):
        rows = {r.id: r for r in self.reviews}
        ids = [a.assessment_id for a in self.adjudications]
        sources = [s.id for s in self.readings]
        if len(ids) != len(set(ids)) or set(ids) != set(rows):
            raise ValueError("interpretation_inventory")
        if len(sources) != len(set(sources)):
            raise ValueError("interpretation_source_duplicate")
        for a in self.adjudications:
            r = rows[a.assessment_id]
            if (
                a.assessment_checksum != r.assessment_checksum
                or a.source_binding_checksum != r.source_binding_checksum
                or not set(a.source_ids) <= set(sources)
                or a.review_completed != (a.state != "not_reviewed")
                or (a.state == "direction_reviewed") != (r.judgment.value is not None)
            ):
                raise ValueError("interpretation_binding_or_direction")
        visual_ids = [r.assessment_id for r in self.image_reviews]
        if len(visual_ids) != len(set(visual_ids)):
            raise ValueError("interpretation_visual_duplicate")
        for v in self.image_reviews:
            if (
                v.assessment_id not in rows
                or v.parent_assessment_id not in rows
                or v.input_checksum != self.input_checksum
                or not set(v.source_ids) <= set(sources)
            ):
                raise ValueError("interpretation_visual_binding")
        return self
