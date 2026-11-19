"""Versioned clinical scope; no patient identifiers or persisted query context."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Term = Annotated[str, StringConstraints(min_length=1, max_length=128)]
Terms = Annotated[tuple[Term, ...], Field(max_length=32)]


class ClinicalScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    version: Literal["clinical-scope/1.0.0"] = "clinical-scope/1.0.0"
    targets: Terms = ()
    disease_packs: Terms = ()
    sexes: Terms = ()
    exams: Terms = ()
    times: Terms = ()
    lateralities: Terms = ()
    devices: Terms = ()
    age_min: int | None = Field(default=None, ge=0, le=150)
    age_max: int | None = Field(default=None, ge=0, le=150)

    @model_validator(mode="after")
    def bounds(self):
        if self.age_min is not None and self.age_max is not None and self.age_min > self.age_max:
            raise ValueError("age bounds inverted")
        return self
