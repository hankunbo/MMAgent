"""Reuse the existing non-secret config loader; role limits are separately versioned."""

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Token, Value, digest, validate
from packages.knowledge_claims.ports import LLMConfig, load_llm_config

from .curl_client import CurlClient
from .models import Role
from .network_route import NetworkRoute, read_route

ROOT = Path(__file__).resolve().parents[2]


class RoleConfig(Value):
    role: Role
    prompt_version: Token
    prompt_checksum: Checksum
    max_output_tokens: int = Field(ge=1, le=16384)
    max_output_bytes: int = Field(ge=256, le=262144)
    max_input_bytes: int = Field(ge=256, le=262144)
    images: bool


class ProviderConfig(Value):
    schema_version: Literal["agent-provider-config/1.0.0"] = "agent-provider-config/1.0.0"
    transport: LLMConfig
    mode: Literal["fake", "live"]
    allowed_endpoints: tuple[Literal["chat_completions", "responses"], ...]
    max_requests: int = Field(ge=1, le=100)
    max_concurrency: int = Field(ge=1, le=8)
    retry_delay_seconds: float = Field(ge=0, le=5)
    image_capable: bool
    roles: tuple[RoleConfig, ...]
    independence: Literal["same_model_roles_are_correlated_not_clinical_review"]

    @model_validator(mode="after")
    def capabilities(self):
        if self.transport.endpoint not in self.allowed_endpoints:
            raise ValueError("endpoint_not_allowed")
        if self.transport.provider == "openai_compatible" and self.allowed_endpoints != (
            "chat_completions",
        ):
            raise ValueError("compatible_endpoint_unverified")
        if (
            set(r.role for r in self.roles) != {"trace", "expert", "source", "vision", "review"}
            or len(self.roles) != 5
        ):
            raise ValueError("role_inventory")
        return self


class ProviderConfigV2(ProviderConfig):
    """Opt-in confirmed HTTP recovery; legacy configuration remains exact."""

    schema_version: Literal["agent-provider-config/2.0.0"] = "agent-provider-config/2.0.0"
    recovery: Literal["confirmed_http_once/1.0.0"]


class ProviderConfigV3(ProviderConfigV2):
    schema_version: Literal["agent-provider-config/3.0.0"] = "agent-provider-config/3.0.0"
    wire_policy: Literal["normalized_arrays/1.0.0"]
    wire_mode: Literal["normalized_strict", "json_object"]


class ReviewBatchPolicy(Value):
    schema_version: Literal["review-batch-policy/1.0.0"] = "review-batch-policy/1.0.0"
    batch_size: int = Field(ge=1, le=5)
    max_batches: int = Field(ge=1, le=20)
    planner_actions: Literal[1] = 1
    partition: Literal["sorted_task_ids_full_parent_material"] = (
        "sorted_task_ids_full_parent_material"
    )


class BatchedProviderConfig(ProviderConfigV3):
    schema_version: Literal["agent-provider-config/4.0.0"] = "agent-provider-config/4.0.0"
    review_batching: ReviewBatchPolicy
    wire_mode: Literal["json_object"] = "json_object"


class ReviewShapeProviderConfig(BatchedProviderConfig):
    """Explicit response-only compatibility, never a change to scoring semantics."""

    schema_version: Literal["agent-provider-config/5.0.0"] = "agent-provider-config/5.0.0"
    review_response_policy: Literal["review-missing-method/1.0.0"]


class IntegratedReviewProviderConfig(ReviewShapeProviderConfig):
    schema_version: Literal["agent-provider-config/6.0.0"] = "agent-provider-config/6.0.0"
    review_correction_policy: Literal["schema-feedback-once/1.0.0"]
    max_extra_requests: Literal[3] = 3


class SourceQuoteProviderConfig(IntegratedReviewProviderConfig):
    schema_version: Literal["agent-provider-config/7.0.0"] = "agent-provider-config/7.0.0"
    source_quote_selection: Literal["exact-snippets/1.0.0"]


class DirectionalProviderConfig(SourceQuoteProviderConfig):
    schema_version: Literal["agent-provider-config/8.0.0"] = "agent-provider-config/8.0.0"
    direction_policy: Literal["direction-assessment/1.0.0"] = "direction-assessment/1.0.0"


class NullableReviewProviderConfig(DirectionalProviderConfig):
    """Lossless optional-null interpretation and one separately budgeted correction."""

    schema_version: Literal["agent-provider-config/9.0.0"] = "agent-provider-config/9.0.0"
    review_response_policy: Literal["review-optional-null/1.1.0"] = "review-optional-null/1.1.0"
    format_correction_budget: Literal[1] = 1


class RoutedSourceProviderConfig(SourceQuoteProviderConfig):
    """Network-only opt-in; preserve legacy 7.0 medical/wire semantics."""

    schema_version: Literal["agent-provider-config/7.1.0"] = "agent-provider-config/7.1.0"
    network_route: NetworkRoute


class RoutedNullableProviderConfig(NullableReviewProviderConfig):
    """Network-only opt-in; preserve legacy 9.0 medical/wire semantics."""

    schema_version: Literal["agent-provider-config/9.1.0"] = "agent-provider-config/9.1.0"
    network_route: NetworkRoute


class CurlSourceProviderConfig(RoutedSourceProviderConfig):
    schema_version: Literal["agent-provider-config/7.2.0"] = "agent-provider-config/7.2.0"
    http_client: CurlClient


class CurlNullableProviderConfig(RoutedNullableProviderConfig):
    schema_version: Literal["agent-provider-config/9.2.0"] = "agent-provider-config/9.2.0"
    http_client: CurlClient


def with_http_client(config, client):
    """Explicit new transport version; never mutate an existing run/configuration."""
    origin = read_provider_config(config)
    version = {
        "agent-provider-config/7.1.0": "agent-provider-config/7.2.0",
        "agent-provider-config/9.1.0": "agent-provider-config/9.2.0",
    }.get(origin.schema_version)
    if version is None:
        raise ValueError("http_client_origin_version")
    raw = origin.model_dump(mode="json")
    raw.update(
        schema_version=version, http_client=validate(CurlClient, client).model_dump(mode="json")
    )
    return read_provider_config(raw)


def with_network_route(config, route):
    origin = read_provider_config(config)
    version = {
        "agent-provider-config/7.0.0": "agent-provider-config/7.1.0",
        "agent-provider-config/9.0.0": "agent-provider-config/9.1.0",
    }.get(origin.schema_version)
    if version is None or route is None:
        raise ValueError("network_route_origin_version")
    raw = origin.model_dump(mode="json")
    raw.update(schema_version=version, network_route=read_route(route).model_dump(mode="json"))
    return read_provider_config(raw)


def with_replacement_network_route(config, route):
    """Explicit routed-client transition; preserve every inference/medical field.

    This derives a new immutable configuration. Its hash must be bound in a new
    campaign, never substituted into a previously frozen request or protocol.
    """
    origin = read_provider_config(config)
    selected = read_route(route)
    if (
        origin.schema_version
        not in {
            "agent-provider-config/7.1.0",
            "agent-provider-config/7.2.0",
            "agent-provider-config/9.1.0",
            "agent-provider-config/9.2.0",
        }
        or selected is None
        or selected == origin.network_route
    ):
        raise ValueError("network_route_replacement_origin")
    raw = origin.model_dump(mode="json")
    raw["network_route"] = selected.model_dump(mode="json")
    return read_provider_config(raw)


def qualified_citation_upgrade(config):
    """Explicit new qualified-Review wire contract; do not reinterpret old 2.2 bytes."""
    old = read_provider_config(config)
    if old.schema_version != "agent-provider-config/8.0.0":
        raise ValueError("qualified_citation_origin_version")
    raw = old.model_dump(mode="json")
    role = next(r for r in raw["roles"] if r["role"] == "review")
    if role["prompt_version"] != "review-direction-2.2.0":
        raise ValueError("qualified_citation_origin_prompt")
    role["prompt_version"] = "review-direction-2.3.0"
    role["prompt_checksum"] = digest(
        (ROOT / "configs/prompts/agents/review-direction-2.3.0.txt").read_text()
    )
    return read_provider_config(raw)


def review_format_upgrade(config):
    old = read_provider_config(config)
    if old.schema_version != "agent-provider-config/8.0.0":
        raise ValueError("review_format_origin_version")
    raw = old.model_dump(mode="json")
    raw.update(
        schema_version="agent-provider-config/9.0.0",
        review_response_policy="review-optional-null/1.1.0",
        format_correction_budget=1,
    )
    return read_provider_config(raw)


def read_provider_config(value):
    raw = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    model = {
        "agent-provider-config/1.0.0": ProviderConfig,
        "agent-provider-config/2.0.0": ProviderConfigV2,
        "agent-provider-config/3.0.0": ProviderConfigV3,
        "agent-provider-config/4.0.0": BatchedProviderConfig,
        "agent-provider-config/5.0.0": ReviewShapeProviderConfig,
        "agent-provider-config/6.0.0": IntegratedReviewProviderConfig,
        "agent-provider-config/7.0.0": SourceQuoteProviderConfig,
        "agent-provider-config/7.1.0": RoutedSourceProviderConfig,
        "agent-provider-config/7.2.0": CurlSourceProviderConfig,
        "agent-provider-config/8.0.0": DirectionalProviderConfig,
        "agent-provider-config/9.0.0": NullableReviewProviderConfig,
        "agent-provider-config/9.1.0": RoutedNullableProviderConfig,
        "agent-provider-config/9.2.0": CurlNullableProviderConfig,
    }.get(raw.get("schema_version", "agent-provider-config/1.0.0"))
    if model is None:
        raise ValueError("provider_configuration_version")
    return validate(model, raw)


def load_provider_config(
    path=ROOT / "configs/agents/provider.json", *, environ=None, llm_path=None
):
    try:
        if path.stat().st_size > 65536:
            raise ValueError("configuration_limit")
        raw = json.loads(path.read_text())
        raw["transport"] = load_llm_config(
            **({"path": llm_path} if llm_path else {}), environ=environ
        )
        return validate(ProviderConfig, raw)
    except (ValueError, OSError, TypeError):
        raise ValueError("agent_configuration_invalid") from None


def prompt(config, role):
    setting = next(r for r in config.roles if r.role == role)
    path = ROOT / "configs/prompts/agents" / (setting.prompt_version + ".txt")
    text = path.read_text()
    if digest(text) != setting.prompt_checksum:
        raise ValueError("prompt_version_drift")
    return setting, text
