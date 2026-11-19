"""Versioned deterministic templates and an optional, allowlisted synonym port."""

from __future__ import annotations

from typing import Protocol

from packages.knowledge_connectors import ProviderConfig
from packages.knowledge_connectors.ports import checked, digest, model_checksum

from .models import (
    SYNONYM_PROMPT,
    ExpansionConfig,
    ExpansionOutput,
    ExpansionTrace,
    KnowledgeQueryPlan,
    ProviderIdentity,
    ProviderQuery,
)


class SynonymExpander(Protocol):
    def suggest(self, plan: KnowledgeQueryPlan, config: ExpansionConfig) -> str: ...


def expand_plan(
    plan: KnowledgeQueryPlan, config: ExpansionConfig, expander: SynonymExpander | None = None
) -> KnowledgeQueryPlan:
    plan = checked(KnowledgeQueryPlan, plan)
    config = checked(ExpansionConfig, config)
    if not config.enabled:
        return plan
    if plan.expansion is not None or expander is None:
        raise ValueError("expansion requires an unexpanded plan and an explicit port")
    output = expander.suggest(plan, config)
    if type(output) is not str or len(output.encode()) > 8192:
        raise ValueError("bounded JSON expansion required")
    # Duplicate JSON keys must not silently override earlier instructions.
    import json

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate expansion field")
            result[key] = value
        return result

    json.loads(output, object_pairs_hook=pairs)
    parsed = ExpansionOutput.model_validate_json(output)
    suggestions = tuple(
        sorted(set(parsed.suggestions), key=lambda s: (s.concept_code, s.kind, s.term))
    )
    if any(s not in config.allowlist for s in suggestions):
        raise ValueError("expansion is outside the configured concept/term allowlist")
    trace = ExpansionTrace(
        suggestion_policy="set/2.0.0",
        prompt_version=config.prompt_version,
        model=config.model,
        config_version=config.config_version,
        config_checksum=model_checksum(config),
        input_plan_checksum=plan.checksum(),
        prompt_checksum=digest((SYNONYM_PROMPT + config.prompt_version + plan.checksum()).encode()),
        config=config,
        output_checksum=model_checksum(ExpansionOutput(suggestions=suggestions)),
        suggestions=suggestions,
    )
    return KnowledgeQueryPlan.model_validate(plan.model_copy(update={"expansion": trace}))


def provider_query(plan: KnowledgeQueryPlan, config: ProviderConfig, policy) -> ProviderQuery:
    plan = checked(KnowledgeQueryPlan, plan)
    config = checked(ProviderConfig, config)
    from .provider import configuration_proof

    identity = ProviderIdentity(
        configuration=configuration_proof(config, policy),
        provider_code=config.provider_code,
        provider_version=config.provider_version,
        adapter=config.adapter,
        config_checksum=config.checksum(),
        identity_policy=config.identity_policy,
        contact_environment=config.contact_email_env,
        auth_environment_name=config.api_key_env,
        identity_configured=config.contact_email_env is not None,
        policy_version=policy.policy_version,
        policy_checksum=model_checksum(policy),
        deployment_workers=policy.deployment_workers,
    )
    texts, mode = query_texts(plan, config.adapter)
    return ProviderQuery(
        identity=identity,
        plan_checksum=plan.checksum(),
        template_version=plan.template_version,
        texts=texts,
        constraints_mode=mode,
    )


def query_texts(plan, adapter):
    from .template import render

    return render(plan.model_dump(mode="json"), adapter)
