"""Data annotations are not clinical acceptance or a capability grant."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from packages.knowledge_claims.ports import LLMConfig
from packages.llm_provider.curl_client import CurlClient
from packages.llm_provider.network_route import NetworkRoute

Text = Annotated[str, Field(min_length=1, max_length=700)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Annotation(Record):
    category: Literal["type_modality", "content_regions", "visibility", "semantics"]
    label: Text
    value: Text
    basis: Literal["content_observation", "inference"]
    view_id: str = Field(pattern=r"^V[0-9]+$")
    location: Text
    evidence: Text
    certainty: Literal["tentative", "moderate", "high"]
    limitation: Text
    attribute: (
        Literal["modality", "laterality", "unit", "orientation", "field_meaning", "time_basis"]
        | None
    ) = None


class VisualResult(Record):
    schema_version: Literal["qualitative-visual-result/1.0.0"]
    annotations: list[Annotation] = Field(min_length=1, max_length=24)
    unconfirmed: list[Text] = Field(max_length=16)
    disease_assessment: Literal["not_performed"]


class Anatomy(Record):
    name: Text
    identification: Literal["identified", "candidate"]
    view_id: str = Field(pattern=r"^V[0-9]+$")
    location: Text
    evidence: Text
    limitation: Text


class VisualLaterality(Record):
    status: Literal["marker_readable", "not_visually_confirmable"]
    value: Literal["LEFT", "RIGHT", "UNKNOWN"]
    evidence: Text
    view_id: str = Field(pattern=r"^V[0-9]+$")

    @model_validator(mode="after")
    def marker(self):
        if (self.status == "not_visually_confirmable") != (self.value == "UNKNOWN"):
            raise ValueError("visual_laterality_status_value")
        return self


class Uncertainty(Record):
    field: Literal["laterality", "physical_scale", "anatomy", "visibility", "orientation", "other"]
    reason: Text
    view_id: str = Field(pattern=r"^V[0-9]+$")


class VisualResultV2(VisualResult):
    schema_version: Literal["qualitative-visual-result/1.1.0"]
    anatomy: list[Anatomy] = Field(max_length=24)
    visual_laterality: VisualLaterality
    uncertainties: list[Uncertainty] = Field(max_length=16)


class AnalysisConfig(Record):
    schema_version: Literal[
        "qualitative-analysis-config/1.0.0", "qualitative-analysis-config/1.1.0"
    ]
    transport: LLMConfig
    network_route: NetworkRoute | None = None
    http_client: CurlClient | None = None
    prompt: str
    prompt_checksum: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    max_requests: int = Field(ge=1, le=100)
    duration_seconds: int = Field(ge=60, le=14400)
    views_per_component: int = Field(ge=1, le=5)
    image_edge: int = Field(ge=256, le=1536)
    max_output_tokens: int = Field(ge=512, le=8192)
    max_response_bytes: int = Field(ge=4096, le=262144)

    @model_validator(mode="after")
    def bounded(self):
        if self.transport.endpoint != "chat_completions" or self.transport.max_retries:
            raise ValueError("qualitative_requires_single_dispatch_chat_transport")
        return self


class MaterialScope(Record):
    schema_version: Literal["qualitative-material-scope/1.0.0"] = "qualitative-material-scope/1.0.0"
    mode: Literal["archive_description", "explicit_selection"]
    component_ids: list[str] = Field(min_length=1)
    background_ids: list[str]
    purpose: Literal["data_description_only"] = "data_description_only"

    @model_validator(mode="after")
    def unique(self):
        for values in (self.component_ids, self.background_ids):
            if len(values) != len(set(values)):
                raise ValueError("qualitative_duplicate_scope")
        return self
