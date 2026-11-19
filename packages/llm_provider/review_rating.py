"""Explicit Review-only presence contract; old configurations remain unchanged."""

from pathlib import Path

from packages.grounding.models import digest

POLICY = "rating-presence/1.0.0"
PROMPT = "review-direction-2.9.0"
POLICIES = {POLICY: PROMPT, "rating-presence/1.1.0": "review-direction-2.10.0"}


def apply(config, policy):
    if policy is None:
        return config
    from packages.evidence_details.live_permission import validate_live_config

    from .config import read_provider_config

    raw = config.model_dump(mode="json")
    review = next(r for r in raw["roles"] if r["role"] == "review")
    prompts = Path(__file__).resolve().parents[2] / "configs/prompts/agents"
    if (
        policy not in POLICIES
        or config.schema_version
        not in {"agent-provider-config/9.0.0", "agent-provider-config/9.1.0"}
        or review["prompt_version"]
        not in (
            {"review-direction-2.7.0", "review-direction-2.8.0"}
            | ({PROMPT} if policy == "rating-presence/1.1.0" else set())
        )
        or review["prompt_checksum"]
        != digest((prompts / (review["prompt_version"] + ".txt")).read_text())
        or config.format_correction_budget != 1
        or config.review_correction_policy != "schema-feedback-once/1.0.0"
        or config.transport.max_retries != 0
    ):
        raise ValueError("review_rating_configuration")
    selected = POLICIES[policy]
    review.update(
        prompt_version=selected, prompt_checksum=digest((prompts / (selected + ".txt")).read_text())
    )
    return validate_live_config(read_provider_config(raw))


def check_entry_binding(spec, protocol=None):
    policy = spec.get("review_contract")
    if protocol is not None and policy != protocol.get("review_contract"):
        raise ValueError("review_rating_protocol_binding")
    prompts = [
        r["prompt_version"]
        for r in spec.get("provider", {}).get("roles", [])
        if r["role"] == "review"
    ]
    if policy is None:
        if set(prompts) & set(POLICIES.values()):
            raise ValueError("review_rating_missing_entry_policy")
        return None
    if (
        policy not in POLICIES
        or spec.get("schema_version")
        not in {"multimodal-live-entry/1.29.0", "multimodal-live-entry/1.30.0"}
        or spec.get("execution_profile") != "relay-root-question-v1"
        or prompts != [POLICIES.get(policy)]
        or spec.get("network_route") != spec.get("provider", {}).get("network_route")
        or not spec.get("network_route")
    ):
        raise ValueError("review_rating_entry_binding")
    return policy
