"""06-owned explicit research output; normalized scores are not disease probabilities."""

import json
from typing import Literal

from pydantic import model_validator

from packages.grounding.models import Checksum, Value

from .core import CLASSES, CONFIG_CHECKSUM, WEIGHT_SHA256, scores_from_cosines


class FairCLIPOutput(Value):
    schema_version: Literal["fairclip-output/1.0.0"] = "fairclip-output/1.0.0"
    input_checksum: Checksum
    model_checksum: Checksum
    config_checksum: Checksum
    raw_cosines: tuple[float, float]
    normalized_model_scores: tuple[float, float]
    predicted_index: Literal[0, 1]
    predicted_class: Literal["non-glaucoma", "glaucoma"]
    calibration: Literal["unverified"] = "unverified"
    research_only: Literal[True] = True
    decisive_evidence: Literal[False] = False
    demographics_used: tuple[()] = ()

    @model_validator(mode="after")
    def coherent(self):
        scores, predicted = scores_from_cosines(self.raw_cosines)
        if self.model_checksum != "sha256:" + WEIGHT_SHA256:
            raise ValueError("fairclip_model_drift")
        if self.config_checksum != CONFIG_CHECKSUM:
            raise ValueError("fairclip_config_drift")
        if any(
            abs(a - b) > 1e-6 for a, b in zip(scores, self.normalized_model_scores, strict=True)
        ):
            raise ValueError("fairclip_softmax_mismatch")
        if any(not 0 <= v <= 1 for v in self.normalized_model_scores):
            raise ValueError("fairclip_score_range")
        if self.predicted_index != predicted or self.predicted_class != CLASSES[predicted]:
            raise ValueError("fairclip_class_order_or_tie")
        return self


def read_output(raw):
    if len(raw) > 65536:
        raise ValueError("fairclip_output_size")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("fairclip_duplicate_field")
            result[key] = value
        return result

    json.loads(raw, object_pairs_hook=unique_pairs)
    return FairCLIPOutput.model_validate_json(raw)
