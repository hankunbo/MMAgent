"""Roles only propose values; installed readers and original domains retain authority."""

from packages.grounding.models import digest, validate
from packages.llm_provider.models import (
    ExpertProposal,
    Material,
    PartialVisionProposal,
    ReviewProposal,
    ReviewTemplate,
    Scope,
    SourceProposal,
    VisionProposal,
)
from packages.llm_provider.provider import ProviderError
from packages.planner_control.models import LegalActionSet, Reference, TraceRanking

OUTPUTS = {
    "expert": ExpertProposal,
    "source": SourceProposal,
    "vision": VisionProposal,
    "review": ReviewProposal,
}


def provider_locks(config):
    return (
        Reference(
            role="model",
            code="CONFIG:AGENT_PROVIDER",
            version=config.schema_version,
            checksum=digest(config),
        ),
        Reference(
            role="prompt",
            code="CONFIG:AGENT_PROMPTS",
            version="agent-prompts/1.0.0",
            checksum=digest(config.roles),
        ),
    )


def action_projection(actions):
    return [
        dict(
            id=a.id,
            kind=a.kind,
            target=a.target.model_dump(mode="json"),
            required_inputs=[r.model_dump(mode="json") for r in a.required_inputs],
            components=[r.model_dump(mode="json") for r in a.components],
            expires_at=a.expires_at.isoformat(),
        )
        for a in actions.actions
    ]


class ProviderTraceAgent:
    """The existing TraceAgent.rank(LegalActionSet, *, deadline) port, unchanged."""

    def __init__(self, *, provider, materials, budget):
        self.provider, self.materials, self.budget = provider, materials, budget

    def rank(self, actions, *, deadline):
        actions = validate(LegalActionSet, actions)
        locks = {r.role: r for r in actions.binding.locks}
        if any(locks.get(r.role) != r for r in provider_locks(self.provider.config)):
            raise ProviderError("scope")
        material = validate(Material, self.materials.read(actions.binding))
        scope = material.scope
        if (
            scope.case_code,
            scope.run_id,
            scope.graph,
            scope.input_checksum,
            scope.visible_checksum,
        ) != (
            actions.binding.case_code,
            actions.binding.run_id,
            actions.graph,
            actions.binding.input_checksum,
            actions.binding.visible_checksum,
        ):
            raise ProviderError("scope")
        payload = dict(
            role="trace",
            material=material.model_dump(mode="json"),
            locks=[r.model_dump(mode="json") for r in actions.binding.locks],
            set_checksum=actions.checksum,
            policy_checksum=digest(actions.policy),
            actions=action_projection(actions),
        )

        def admit(result):
            if (
                result.set_checksum != actions.checksum
                or result.policy_checksum != digest(actions.policy)
                or not set(result.ranked_ids) <= {a.id for a in actions.actions}
            ):
                raise ProviderError("scope")
            if validate(Material, self.materials.read(actions.binding)) != material:
                raise ProviderError("scope")

        try:
            return self.provider.generate_structured(
                role="trace",
                input_payload=payload,
                output_schema=TraceRanking,
                scope=scope,
                budget=self.budget,
                deadline=min(deadline, actions.expires_at),
                admit=admit,
            )
        except ProviderError as exc:
            if exc.code in {"timeout", "rate_limit", "transport"}:
                # Agent retries share the SAME RequestBudget, never create another budget.
                raise ConnectionError("provider_unavailable") from None
            raise


class StructuredRoleAgent:
    """Expert is only a proposal port here. 08D owns its business orchestration."""

    def __init__(self, role, *, provider, materials, budget):
        if role not in OUTPUTS:
            raise ProviderError("configuration")
        self.role, self.provider, self.materials, self.budget = role, provider, materials, budget

    def call(self, scope, *, deadline, legal_actions=None, round_id=None, before_request=None):
        scope = validate(Scope, scope)
        material = validate(Material, self.materials.read(scope))
        if material.scope != scope:
            raise ProviderError("scope")
        payload = dict(role=self.role, material=material.model_dump(mode="json"))
        images = ()
        allowed = set()
        output_schema = OUTPUTS[self.role]
        if self.role == "expert":
            actions = validate(LegalActionSet, legal_actions)
            if (
                actions.binding.run_id,
                actions.binding.case_code,
                actions.graph,
                actions.binding.input_checksum,
                actions.binding.visible_checksum,
            ) != (
                scope.run_id,
                scope.case_code,
                scope.graph,
                scope.input_checksum,
                scope.visible_checksum,
            ):
                raise ProviderError("scope")
            payload["actions"] = action_projection(actions)
            allowed = {a.id for a in actions.actions}
            deadline = min(deadline, actions.expires_at)
        if self.role == "vision":
            bound = [o.image for o in material.observations if o.image is not None]
            if len(bound) != 1:
                raise ProviderError("unsupported_image")
            images = ((bound[0], self.materials.image(bound[0])),)
            # Explicit opt-in: legacy Vision wire stays byte-for-byte unchanged.
            if callable(getattr(self.materials, "vision_task", None)):
                from packages.llm_provider.models import ReviewTask

                task = validate(ReviewTask, self.materials.vision_task(scope))
                template = self.materials.review_template(scope)
                observed = next(o for o in material.observations if o.image is not None)
                if task not in template.tasks or task.observation != observed.reference:
                    raise ProviderError("scope", fields=("vision_task",))
                payload["schema_version"] = "targeted-vision-input/1.0.0"
                payload["vision_task"] = task.model_dump(mode="json")
                if callable(getattr(self.materials, "input_scope_task", None)):
                    from packages.llm_provider.input_scope import (
                        PROMPT,
                        STORAGE_PROMPT,
                        TEXT_PROMPT,
                        ScopeTask,
                        ScopeVisionPayload,
                        proposal_type,
                    )

                    setting = next(r for r in self.provider.config.roles if r.role == "vision")
                    if (
                        setting.prompt_version not in {PROMPT, STORAGE_PROMPT, TEXT_PROMPT}
                        or task.method != "input_scope_survey"
                    ):
                        raise ProviderError("configuration", fields=("input_scope_prompt",))
                    scope_task = validate(ScopeTask, self.materials.input_scope_task(scope))
                    payload = validate(
                        ScopeVisionPayload,
                        dict(
                            payload,
                            schema_version="input-scope-vision-input/1.0.0",
                            scope_task=scope_task,
                        ),
                    ).model_dump(mode="json")
                    output_schema = proposal_type(setting.prompt_version)
                elif getattr(self.materials, "partial_observation_contract", None) == "1.0.0":
                    from packages.llm_provider.vision_correction import PROMPT as FEEDBACK_PROMPT

                    setting = next(r for r in self.provider.config.roles if r.role == "vision")
                    if setting.prompt_version not in {
                        "vision-partial-1.0.0",
                        "vision-partial-multimodal-1.0.0",
                        "vision-partial-multimodal-1.1.0",
                        "vision-partial-multimodal-1.2.0",
                        FEEDBACK_PROMPT,
                    }:
                        raise ProviderError("configuration", fields=("partial_prompt",))
                    output_schema = PartialVisionProposal
        if self.role == "review":
            if round_id not in {"A", "B", "C"}:
                raise ProviderError("scope")
            from packages.evidence_details.review_tasks import review_payload

            template = validate(ReviewTemplate, self.materials.review_template(scope))
            payload = review_payload(material, template, round_id)
            if not template.tasks:
                raise ProviderError("scope")  # Fixed-only policy needs no LLM evaluation.

        def admit(result):
            if result.scope_checksum != digest(scope):
                raise ProviderError("scope", fields=("scope_checksum",))
            refs = {s.reference.code for s in material.sources} | {
                o.reference.code for o in material.observations
            }
            if self.role == "expert":
                if not set(result.evidence) <= refs or len(set(result.evidence)) != len(
                    result.evidence
                ):
                    raise ProviderError("scope")
                if result.action in {"accept", "alternative"} and result.action_id not in allowed:
                    raise ProviderError("scope")
                if result.action in {"reject", "safe_stop"} and result.action_id is not None:
                    raise ProviderError("scope")
            if self.role == "source":
                from packages.knowledge_claims.research_quote import (
                    LAYOUT_POLICY,
                    align_research_quote,
                    quote_policy,
                )

                sources = {s.reference.code: s for s in material.sources}
                mismatches = []
                if result.source_code not in sources:
                    mismatches.append("source_code")
                elif (
                    align_research_quote(
                        result.quote,
                        sources[result.source_code].text,
                        allow_layout=quote_policy(self.materials) == LAYOUT_POLICY,
                    )
                    is None
                ):
                    mismatches.append("quote")
                if result.target != material.goal_code:
                    mismatches.append("target")
                if mismatches:
                    raise ProviderError("scope", fields=mismatches)
            if self.role == "vision":
                if payload.get("schema_version") == "input-scope-vision-input/1.0.0":
                    from packages.llm_provider.input_scope import ScopeTask, check_regions

                    check_regions(result, scope_task)
                    if validate(ScopeTask, self.materials.input_scope_task(scope)) != scope_task:
                        raise ProviderError("scope", fields=("input_scope_task",))
                b = images[0][0]
                mismatches = [
                    name
                    for name, expected in (
                        ("image_checksum", b.reference.checksum),
                        ("preprocessing_checksum", b.preprocessing.checksum),
                        ("location", b.location),
                    )
                    if getattr(result, name) != expected
                ]
                if mismatches:
                    raise ProviderError("scope", fields=mismatches)
                if (
                    "vision_task" in payload
                    and self.materials.vision_task(scope).model_dump(mode="json")
                    != payload["vision_task"]
                ):
                    raise ProviderError("scope", fields=("vision_task",))
            if self.role == "review":
                mismatches = []
                if result.round != round_id:
                    mismatches.append("round")
                if {i.assessment_id for i in result.items} != set(template.assessment_ids):
                    mismatches.append("items.assessment_id")
                if mismatches:
                    raise ProviderError("scope", fields=mismatches)
                tasks = {t.assessment_id: t for t in template.tasks}
                for item in result.items:
                    task = tasks[item.assessment_id]
                    if (
                        len(set(item.observation_citations)) != len(item.observation_citations)
                        or len(set(item.reference_citations)) != len(item.reference_citations)
                        or not set(item.observation_citations) <= {task.observation.code}
                        or not set(item.reference_citations)
                        <= {r.code for r in task.allowed_sources}
                    ):
                        raise ProviderError("scope", fields=("items.citations",))
                if validate(ReviewTemplate, self.materials.review_template(scope)) != template:
                    raise ProviderError("scope", fields=("review_tasks",))
            if validate(Material, self.materials.read(scope)) != material:
                raise ProviderError("scope", fields=("material",))

        return self.provider.generate_structured(
            role=self.role,
            input_payload=payload,
            output_schema=output_schema,
            scope=scope,
            budget=self.budget,
            deadline=deadline,
            images=images,
            admit=admit,
            before_request=before_request,
        )
