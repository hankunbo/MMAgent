"""06-owned bounded scalar wire output; metadata is never an evidence validation."""

import math
from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Value


class ScalarOutput(Value):
    schema_version: Literal["scalar-score-output/1.0.0"] = "scalar-score-output/1.0.0"
    metric: Literal["quality", "measurement", "logit", "segmentation_confidence", "uncertainty"]
    value: float
    unit: str = Field(min_length=1, max_length=64)
    input_checksum: Checksum

    @model_validator(mode="after")
    def valid(self):
        if not math.isfinite(self.value):
            raise ValueError("scalar_nonfinite")
        if self.metric == "logit" and self.unit != "logit":
            raise ValueError("scalar_logit_unit")
        if self.metric == "segmentation_confidence" and (
            self.unit != "1" or not 0 <= self.value <= 1
        ):
            raise ValueError("scalar_confidence_range")
        return self


class GeometryOutput(Value):
    schema_version: Literal["disc-geometry/1.0.0"]
    input_checksum: Checksum
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    area_px: int = Field(gt=0)
    bbox_xyxy: tuple[int, int, int, int]
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
