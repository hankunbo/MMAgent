"""Explicit model-only selection for new research campaigns, never request replay.

The public profile supplies a model identity, not its prompts or retry policy.
The complete selection and public profile checksums are bound into the entry.
"""

import json
from pathlib import Path
from typing import Literal

from packages.grounding.models import Value, digest, validate

from .config import read_provider_config

ROOT = Path(__file__).resolve().parents[2]


class Selection(Value):
    schema_version: Literal["provider-model-selection/1.0.0"]
    id: Literal["relay-existing-alternate-v1"]
    profile: Literal["configs/agents/relayrouter_generic_evidence_v1.json"]
    original_model: Literal["gpt-4o-2024-08-06"]
    selected_model: Literal["gpt-6-sol"]
    scope: Literal["fresh_research_campaign"]
    automatic_unknown_replay: Literal[False]


def load_selection(path):
    selected = Path(path)
    selected = selected if selected.is_absolute() else ROOT / selected
    selected = selected.resolve()
    if not selected.is_relative_to(ROOT / "configs/agents") or selected.suffix != ".json":
        raise ValueError("model_selection_public_configuration_required")
    selection = validate(Selection, json.loads(selected.read_bytes()))
    profile = read_provider_config(json.loads((ROOT / selection.profile).read_bytes()))
    if profile.transport.model != selection.selected_model:
        raise ValueError("model_selection_profile_model_changed")
    return dict(
        schema_version="provider-model-binding/1.0.0",
        path=str(selected.relative_to(ROOT)),
        selection=selection.model_dump(mode="json"),
        selection_checksum=digest(selection),
        profile_checksum=digest(profile),
    )


def apply_selection(config, binding):
    """Change only transport.model; every role, limit and wire option is retained."""
    if not isinstance(binding, dict) or binding != load_selection(binding.get("path", "")):
        raise ValueError("model_selection_binding_changed")
    selection = validate(Selection, binding["selection"])
    profile = read_provider_config(json.loads((ROOT / selection.profile).read_bytes()))
    if (
        config.transport.model != selection.original_model
        or getattr(config, "network_route", None) is None
        or any(getattr(role, "model", None) is not None for role in config.roles)
        or any(
            getattr(config.transport, field) != getattr(profile.transport, field)
            for field in ("provider", "base_url", "endpoint", "api_key_env", "store")
        )
    ):
        raise ValueError("model_selection_original_route_or_model")
    raw = config.model_dump(mode="json")
    raw["transport"]["model"] = selection.selected_model
    return read_provider_config(raw)


def check_entry_binding(spec, protocol=None):
    """An opt-in versioned extension; old entries have no such field."""
    binding = spec.get("model_selection")
    if protocol is not None and binding != protocol.get("model_selection"):
        raise ValueError("model_selection_protocol_binding")
    if binding is None:
        return None
    if (
        spec.get("schema_version")
        not in {
            "multimodal-live-entry/1.29.0",
            "multimodal-live-entry/1.30.0",
        }
        or spec.get("execution_profile") != "relay-root-question-v1"
    ):
        raise ValueError("model_selection_entry_scope")
    from packages.multimodal_research.campaign import configuration

    origin = configuration(
        resilient_vision=True,
        directional=True,
        exact_primary_status=True,
        qualified_observations=bool(spec.get("input_semantics")),
        execution_profile=spec["execution_profile"],
        review_batch_size=spec.get("review_batch_size", 1),
        network_route=spec.get("network_route"),
        vision_feedback=spec.get("vision_feedback"),
        review_contract=spec.get("review_contract"),
        model_selection=binding,
        http_client_selection=spec.get("http_client_selection"),
    )
    if origin.model_dump(mode="json") != spec["provider"]:
        raise ValueError("model_selection_entry_provider")
    return binding
