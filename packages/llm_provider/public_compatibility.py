"""Frozen finite public experiments on original Provider/Runtime budgets."""

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from agents.provider_agents import StructuredRoleAgent
from agents.provider_agents.service import OUTPUTS
from packages.evidence_details.live_permission import validate_live_config
from packages.evidence_scoring.accepted import build_accepted
from packages.grounding.models import Checksum, Code, Token, Value, digest, validate

from .config import prompt, read_provider_config
from .http_recovery import format_rejected, observation_for
from .live_probes import PublicProbeMaterials
from .models import RolePayload
from .provider import ProviderError, StructuredProvider
from .public_shape import PublicReviewMaterials
from .transport import wire_request


class ProbeWire(Value):
    role: Literal["source", "vision", "review"]
    wire_checksum: Checksum
    schema_checksum: Checksum
    prompt_checksum: Checksum
    input_checksum: Checksum
    output_schema_checksum: Checksum
    remote_constraint: Literal["normalized_strict", "json_object"]


class PublicCandidate(Value):
    id: Token
    config_path: str
    config_checksum: Checksum
    requests: tuple[ProbeWire, ...] = Field(min_length=2, max_length=3)
    source_reused: bool
    hypothesis: str


class CompatibilityPlan(Value):
    schema_version: Literal["public-compatibility-plan/1.0.0"] = "public-compatibility-plan/1.0.0"
    session_id: Token
    run_id: Code
    candidates: tuple[PublicCandidate, ...] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def exact(self):
        if len({c.id for c in self.candidates}) != len(self.candidates) or len(
            {c.config_checksum for c in self.candidates}
        ) != len(self.candidates):
            raise ValueError("public_candidate_duplicate")
        for candidate in self.candidates:
            expected = (
                {"vision", "review"} if candidate.source_reused else {"vision", "review", "source"}
            )
            if {r.role for r in candidate.requests} != expected or len(candidate.requests) != len(
                expected
            ):
                raise ValueError("public_role_inventory")
        return self


def public_materials(run_id, config, role):
    cls = PublicReviewMaterials if role == "review" else PublicProbeMaterials
    return cls(run_id, config)


def construct(config, materials, role):
    from packages.evidence_details.review_tasks import project_review_tasks, review_payload

    material = materials.state.material
    payload = (
        review_payload(material, project_review_tasks(materials.state), "A")
        if role == "review"
        else dict(role=role, material=material.model_dump(mode="json"))
    )
    validate(RolePayload, payload)
    images = ((material.observations[1].image, materials.raw),) if role == "vision" else ()
    setting, text = prompt(config, role)
    schema = OUTPUTS[role].model_json_schema()
    body = wire_request(
        config.transport,
        setting,
        text,
        payload,
        schema,
        images,
        scope_binding=digest(material.scope),
        wire_policy=getattr(config, "wire_policy", None),
        wire_mode=getattr(config, "wire_mode", None),
    )
    return body, ProbeWire(
        role=role,
        wire_checksum=digest(body),
        schema_checksum=digest(body["response_format"]),
        prompt_checksum=setting.prompt_checksum,
        input_checksum=digest(payload),
        output_schema_checksum=digest(schema),
        remote_constraint=getattr(config, "wire_mode", "normalized_strict"),
    )


def load_plan(round_config):
    stem = "vectorengine_" + round_config.session_id.replace("-", "_")
    path = Path("configs/research") / (stem + "_candidates.json")
    plan = validate(CompatibilityPlan, json.loads(path.read_text()))
    if plan.session_id != round_config.session_id or plan.run_id != round_config.run_id:
        raise ValueError("public_plan_identity")
    allowed = {"configs/agents/" + stem + suffix for suffix in (".json", "_json.json")}
    for candidate in plan.candidates:
        if candidate.config_path not in allowed:
            raise ValueError("public_config_path")
        config = validate_live_config(
            read_provider_config(json.loads(Path(candidate.config_path).read_text()))
        )
        if digest(config) != candidate.config_checksum:
            raise ValueError("public_config_lock")
        for request in candidate.requests:
            _, actual = construct(
                config, public_materials(plan.run_id, config, request.role), request.role
            )
            if actual != request:
                raise ValueError("public_wire_lock")
    if plan.candidates[0].config_checksum != round_config.provider_checksum:
        raise ValueError("public_initial_config")
    return plan


def controlled_probe(*, candidate, request, config, budget, documents):
    materials = public_materials(budget.run_id, config, request.role)
    _, actual = construct(config, materials, request.role)
    if actual != request or digest(config) != candidate.config_checksum:
        raise ValueError("public_wire_lock")
    slot = "compat2:" + candidate.id + ":" + request.role
    lock = dict(config_checksum=candidate.config_checksum, wire=request)
    with documents.session():
        prior = documents.read(slot + ":lock")
        if prior and digest(prior[0]) != digest(lock):
            raise ValueError("public_slot_conflict")
        saved = documents.read(slot + ":result")
        if saved:
            if saved[0]["wire"] != request.model_dump(mode="json"):
                raise ValueError("public_saved_binding")
            return saved[0]
        budget.before_reserve(digest(config))
        if documents.read(slot + ":claim"):
            raise ValueError("public_attempt_unknown_no_repeat")
        documents.write(slot + ":lock", (lock,))
        documents.write(
            slot + ":claim",
            (dict(config_checksum=digest(config), request_checksum=request.wire_checksum),),
        )
    provider = StructuredProvider(config, live_opt_in=True)
    used_before = budget.used
    try:
        proposal = StructuredRoleAgent(
            request.role, provider=provider, budget=budget, materials=materials
        ).call(
            materials.state.material.scope,
            deadline=budget.deadline,
            round_id="A" if request.role == "review" else None,
        )
        if request.role == "review":
            build_accepted(materials.state, proposal)
        elif request.role == "vision":
            if proposal.status != "observed" or not any(
                s in proposal.finding.lower() for s in ("red", "红")
            ):
                raise ProviderError("scope")
        elif proposal.quote != "Two plus two equals four.":
            raise ProviderError("scope")
        budget.check_control()
        result = dict(status="passed", candidate=proposal, error=None)
    except Exception as error:
        result = dict(
            status="failed",
            error=error.code if isinstance(error, ProviderError) else type(error).__name__,
        )
        if not isinstance(error, ProviderError):
            documents.write(slot + ":technical", (dict(exception_type=type(error).__name__),))
    record = budget.records()[-1] if budget.used > used_before and budget.records() else None
    event = observation_for(budget.journal, record.request_id) if record else None
    if event is not None and (
        event.get("wire_checksum") != request.wire_checksum
        or event.get("config_checksum") != digest(config)
    ):
        raise ValueError("public_http_binding")
    result.update(
        wire=request,
        config_checksum=digest(config),
        candidate_id=candidate.id,
        request_id=record.request_id if record else None,
        http=event,
        patient_material=False,
        remote_constraint=config.wire_mode,
    )
    with documents.session():
        documents.write(slot + ":result", (result,))
    return json.loads(json.dumps(result, default=lambda x: x.model_dump(mode="json")))


def run_comparisons(plan, budget, documents, round_checksum):
    """One active Control across predeclared candidates; no terminal stop on a format rejection."""
    saved = documents.read("public-qualification")
    if saved:
        if (
            saved[0].get("plan_checksum") != digest(plan)
            or saved[0].get("round_checksum") != round_checksum
        ):
            raise ValueError("public_qualification_conflict")
        return saved[0]
    for candidate in plan.candidates:
        config = validate_live_config(
            read_provider_config(json.loads(Path(candidate.config_path).read_text()))
        )
        results = []
        for request in candidate.requests:
            result = controlled_probe(
                candidate=candidate,
                request=request,
                config=config,
                budget=budget,
                documents=documents,
            )
            if result["status"] != "passed":
                if format_rejected(result.get("http")):
                    break
                raise ProviderError(result["error"])
            results.append(result)
        else:
            budget.check_control()
            qualification = dict(
                schema_version="bounded-public-qualification/2.0.0",
                session_id=plan.session_id,
                run_id=plan.run_id,
                round_checksum=round_checksum,
                plan_checksum=digest(plan),
                candidate_id=candidate.id,
                config_checksum=digest(config),
                roles=results,
                source_reused=candidate.source_reused,
                patient_requests=0,
            )
            documents.write("public-qualification", (qualification,))
            return json.loads(
                json.dumps(qualification, default=lambda x: x.model_dump(mode="json"))
            )
    raise ProviderError("configuration")


def check_qualification(plan, qualification, documents, config_checksum):
    if (
        qualification.get("schema_version") != "bounded-public-qualification/2.0.0"
        or qualification.get("plan_checksum") != digest(plan)
        or qualification.get("config_checksum") != config_checksum
    ):
        raise ValueError("public_qualification_binding")
    chosen = next((c for c in plan.candidates if c.id == qualification["candidate_id"]), None)
    if chosen is None or chosen.config_checksum != config_checksum:
        raise ValueError("public_qualified_config")
    if len(qualification["roles"]) != len(chosen.requests):
        raise ValueError("public_qualified_roles")
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    journal = RuntimeAgentJournal(documents)
    for request, result in zip(chosen.requests, qualification["roles"], strict=True):
        stored = documents.read("compat2:" + chosen.id + ":" + request.role + ":result")
        event = observation_for(journal, result.get("request_id"))
        if (
            event is None
            or event != result.get("http")
            or event.get("http_status") != 200
            or event.get("wire_checksum") != request.wire_checksum
            or event.get("config_checksum") != config_checksum
            or event.get("prompt_checksum") != request.prompt_checksum
        ):
            raise ValueError("public_qualified_http")
        if (
            stored is None
            or stored[0] != result
            or result["status"] != "passed"
            or result["wire"] != request.model_dump(mode="json")
            or result["config_checksum"] != chosen.config_checksum
            or result["remote_constraint"] != request.remote_constraint
        ):
            raise ValueError("public_qualified_request")
    return chosen
