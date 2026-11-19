"""Explicit Source-definition format feedback, never a candidate repair.

Only the new prompt opts in. The original request budget/journal owns the one
extra dispatch; a confirmed bad answer is retained, and unknown work is not retried.
"""

from copy import deepcopy

from packages.grounding.models import canonical


def enabled(config, role):
    return bool(
        role == "source"
        and getattr(config, "format_correction_budget", 0) == 1
        and getattr(config, "review_correction_policy", None) == "schema-feedback-once/1.0.0"
        and next(r for r in config.roles if r.role == "source").prompt_version
        in {"definition-draft-6.5.0", "definition-draft-6.6.0", "definition-draft-6.7.0"}
    )


def eligible(config, scope, role, retry, error, reply, observation):
    return bool(
        enabled(config, role)
        and scope.case_code != "CASE:PUBLIC_PROBE"
        and retry == 0
        and error == "schema"
        and reply.content
        and reply.finish_reason == "stop"
        and reply.response_confirmed is True
        and (observation is None or observation.get("http_status") == 200)
    )


def corrected_wire(body, failures):
    result = deepcopy(body)
    result["messages"].append(
        dict(
            role="user",
            content=canonical(
                dict(
                    task="Regenerate the same research-definition-proposal/5.0.0 once.",
                    errors=failures,
                    contract=(
                        "Use only candidate_contract_choices in the original input. "
                        "A 6.1.0 raw-input draft uses input_aliases, never requirement_alias, "
                        "confirmed_method or confirmed_technical_qc. For vision use "
                        "live_image_observation/qualitative, not a derived measurement method. "
                        "A 7.1.0 confirmed-observation draft requires an actual matching "
                        "confirmed_observation choice with its exact requirement_alias, method "
                        "and unit. An empty confirmed list forbids that draft type. "
                        "Copy only a legitimate matching choice; do not relabel one kind as "
                        "another. Preserve every required unresolved condition. Dependencies "
                        "and dependency_contracts have matching text, unique consecutive "
                        "indices, and the original exact source excerpts."
                    ),
                    constraints=(
                        "This feedback supplies no medical answer, observation or authorization. "
                        "Do not invent a capability, remove a required condition or change a "
                        "medical relationship merely to pass. If no supported expressible "
                        "proposal exists, return an empty definitions array and a specific "
                        "no_addition_reason. Return the whole original object, no markdown."
                    ),
                )
            ).decode(),
        )
    )
    return result
