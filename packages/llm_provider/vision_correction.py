"""Explicit once-only state consistency feedback, never medical result repair."""

from pathlib import Path

from packages.grounding.models import digest

POLICY = "state-consistency-once/1.0.0"
PROMPT = "vision-partial-multimodal-1.3.0"
CONSTRAINTS = {"partial_status_signal", "partial_observed_requires_visible_region"}


def enabled(config, role):
    return bool(
        role == "vision"
        and getattr(config, "format_correction_budget", 0) == 1
        and getattr(config, "review_correction_policy", None) == "schema-feedback-once/1.0.0"
        and next(r for r in config.roles if r.role == "vision").prompt_version == PROMPT
    )


def apply(config, policy):
    if policy is None:
        return config
    from .config import read_provider_config

    raw = config.model_dump(mode="json")
    vision = next(r for r in raw["roles"] if r["role"] == "vision")
    if (
        policy != POLICY
        or vision["prompt_version"] != "vision-partial-multimodal-1.2.0"
        or getattr(config, "format_correction_budget", 0) != 1
    ):
        raise ValueError("vision_feedback_configuration")
    vision["prompt_version"] = PROMPT
    path = Path(__file__).resolve().parents[2] / "configs/prompts/agents" / (PROMPT + ".txt")
    vision["prompt_checksum"] = digest(path.read_text())
    return read_provider_config(raw)


def check_entry_binding(spec, protocol=None):
    policy = spec.get("vision_feedback")
    if protocol is not None and policy != protocol.get("vision_feedback"):
        raise ValueError("vision_feedback_protocol_binding")
    roles = spec.get("provider", {}).get("roles", [])
    prompts = [r["prompt_version"] for r in roles if r["role"] == "vision"]
    if policy is None:
        if PROMPT in prompts:
            raise ValueError("vision_feedback_missing_entry_policy")
        return None
    if (
        policy != POLICY
        or spec.get("schema_version")
        not in {
            "multimodal-live-entry/1.29.0",
            "multimodal-live-entry/1.30.0",
        }
        or spec.get("execution_profile") != "relay-root-question-v1"
        or prompts != [PROMPT]
    ):
        raise ValueError("vision_feedback_entry_binding")
    return policy


def eligible(config, scope, role, retry, error, reply, observation, failures):
    return bool(
        enabled(config, role)
        and scope.case_code != "CASE:PUBLIC_PROBE"
        and retry == 0
        and error == "schema"
        and failures
        and all(f.get("constraint") in CONSTRAINTS for f in failures)
        and reply.content
        and reply.finish_reason == "stop"
        and reply.response_confirmed is True
        and (observation is None or observation.get("http_status") == 200)
    )
