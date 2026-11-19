"""One opt-in, lossless redundant-field interpretation; no guesses or repairs to scores."""

import copy
from typing import Literal

from pydantic import Field

from packages.grounding.models import Checksum, Value, digest, validate
from packages.object_store import checksum

POLICY = "review-missing-method/1.0.0"


class ReviewResponseAlignment(Value):
    schema_version: Literal["review-response-alignment/1.0.0"] = "review-response-alignment/1.0.0"
    policy: Literal["review-missing-method/1.0.0"] = POLICY
    raw_checksum: Checksum
    candidate_checksum: Checksum
    redundant_observation_method_indices: tuple[int, ...] = Field(max_length=5)
    unchanged_scores_citations_and_status: Literal[True] = True


class NullableReviewAlignment(ReviewResponseAlignment):
    schema_version: Literal["review-response-alignment/1.1.0"] = "review-response-alignment/1.1.0"
    policy: Literal["review-optional-null/1.1.0"] = "review-optional-null/1.1.0"
    optional_null_paths: tuple[tuple[int, str, str], ...] = Field(max_length=64)


def decode(raw_text, config):
    from packages.evidence_details.review_batches import proposal_type
    from packages.llm_provider.config import (
        NullableReviewProviderConfig,
        ReviewShapeProviderConfig,
        read_provider_config,
    )
    from packages.llm_provider.provider import strict_json

    config = read_provider_config(config)
    ReviewBatchProposal = proposal_type(config)
    raw = strict_json(raw_text)
    if not isinstance(config, ReviewShapeProviderConfig):
        return validate(ReviewBatchProposal, raw), None
    obj = copy.deepcopy(raw)
    indices = []
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        for n, item in enumerate(obj["items"]):
            if not isinstance(item, dict) or not isinstance(item.get("observation"), dict):
                continue  # The unchanged strict schema will reject malformed items.
            rating = item["observation"]
            if "method" not in rating:
                continue
            judgment = item.get("judgment", {})
            if not (
                rating["method"] == "not_assessed"
                and rating.get("value", "absent") is None
                and isinstance(rating.get("missing_reason"), str)
                and rating["missing_reason"].strip()
                and item.get("observation_available") is False
                and item.get("status") in {"input_missing", "observation_ungradable"}
                and isinstance(judgment, dict)
                and judgment.get("method") == "not_assessed"
                and all(judgment.get(k, "absent") is None for k in ("value", "lower", "upper"))
            ):
                raise ValueError("review_redundant_method_conflict")
            del rating["method"]
            indices.append(n)
    nulls = []
    if isinstance(config, NullableReviewProviderConfig) and isinstance(obj, dict):
        for n, item in enumerate(obj.get("items", [])):
            if not isinstance(item, dict):
                continue
            for section in (
                "observation",
                "relation",
                "applicability",
                "judgment",
                "direction_assessment",
            ):
                row = item.get(section)
                if not isinstance(row, dict):
                    continue
                names = (
                    ("neutral_reason", "unassessable_reason", "resolvable_gap", "next_action")
                    if section == "direction_assessment"
                    else ("missing_reason",)
                )
                for name in names:
                    if name not in row:
                        row[name] = None
                        nulls.append((n, section, name))
    candidate = validate(ReviewBatchProposal, obj)
    # Do not allow additional implicit conversions or dropped fields in this adapter.
    if obj != candidate.model_dump(mode="json"):
        raise ValueError("review_response_non_equivalent_shape")
    cls = (
        NullableReviewAlignment
        if isinstance(config, NullableReviewProviderConfig)
        else ReviewResponseAlignment
    )
    proof = cls(
        raw_checksum=checksum(raw_text.encode() if isinstance(raw_text, str) else raw_text),
        candidate_checksum=digest(candidate),
        redundant_observation_method_indices=tuple(indices),
        **(dict(optional_null_paths=tuple(nulls)) if cls is NullableReviewAlignment else {}),
    )
    return candidate, proof
