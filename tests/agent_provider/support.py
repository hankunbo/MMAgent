"""Approved synthetic materials, never old case answers or medical assessments."""

import copy
import json
from contextlib import nullcontext

from packages.evidence_details.acceptance_models import DomainState, Submission, TaskPolicyV2
from packages.evidence_scoring.continuous_policy import POLICY
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider import DeterministicFakeProvider, MemoryJournal, RequestBudget
from packages.llm_provider.config import load_provider_config
from packages.llm_provider.models import Material, ReviewProposal
from packages.planner_control.models import Reference
from tests.planner.support import binding


def ref(role, code, version="1.0.0", checksum=None):
    return Reference(role=role, code=code, version=version, checksum=checksum or digest(code))


def definition(
    source="SOURCE:ONE", *, interpretation="Synthetic comparison, no medical threshold."
):
    raw = dict(
        examination="Compare the synthetic observed feature to the installed source.",
        applicability_conditions="Synthetic unitless fixture only; no patient use.",
        interpretation=interpretation,
        interpretation_status="installed" if interpretation else "missing",
        source_refs=[ref("source", source).model_dump(mode="json")],
    )
    return dict(
        reference=ref("definition", "POLICY:SYNTHETIC_DEFINITION", "2.0.0", digest(raw)).model_dump(
            mode="json"
        ),
        **raw,
    )


def materials(b=None, *, role="review", config=None):
    b = b or binding()
    config = config or load_provider_config(environ={})
    policy = validate(
        TaskPolicyV2,
        dict(
            version="synthetic-task-1.0.0",
            scale=ref("scale", "POLICY:SCALE"),
            aggregation_checksum=digest(POLICY),
            items=[
                dict(
                    definition=definition(),
                    id="item:one",
                    node="GOAL:SYNTHETIC",
                    kind="visual",
                    role="visual",
                    observation_code="OBS:ONE",
                    method="synthetic",
                    unit="unitless",
                    laterality="UNKNOWN",
                    source_codes=["SOURCE:ONE"],
                    applicable=True,
                    correlation_group="image:one",
                    dependence_block="image:one",
                    dependency_paths=[["INPUT:ONE"]],
                )
            ],
            groups=[
                dict(
                    id="group:one",
                    slots=[["item:one"]],
                    importance=dict(slo_image_research=1.0, comprehensive_research=1.0),
                )
            ],
        ),
    )
    scope = dict(
        case_code=b.case_code,
        run_id=b.run_id,
        graph=b.graph.model_dump(mode="json"),
        visible_checksum=b.visible_checksum,
        input_checksum=b.input_checksum,
        policy_checksum=digest(policy),
    )
    raw = dict(
        scope=scope,
        goal_code="GOAL:SYNTHETIC",
        goal_summary="Synthetic approved research task",
        visibility="approved_visible_projection/1.0.0",
        sources=[
            dict(
                reference=ref("source", "SOURCE:ONE").model_dump(mode="json"),
                locator="Synthetic paragraph 1",
                purpose="research_material",
                state="research_only",
                release=None,
                text="Synthetic source supports a bounded research comparison.",
            )
        ],
        observations=[
            dict(
                reference=ref("observation", "OBS:ONE").model_dump(mode="json"),
                status="obtained",
                method="synthetic",
                unit="unitless",
                value="Synthetic observation",
                image=None,
            )
        ],
        policy_ref=ref("policy", "POLICY:TASK", checksum=digest(policy)).model_dump(mode="json"),
        missing=[],
        memory=[],
    )
    scope["material_checksum"] = digest(raw)
    m = validate(Material, raw)
    state = DomainState(
        material=m,
        policy=policy,
        laterality="UNKNOWN",
        resources=(ref("input", "INPUT:ONE"),),
        prompt_checksum=next(r.prompt_checksum for r in config.roles if r.role == role),
        config_checksum=digest(config),
        nodes=("GOAL:SYNTHETIC",),
    )
    return Reader(state), b


class Reader:
    def __init__(self, state):
        self.state = state
        self.image_bytes = b""
        self.withdrawn = False

    def read(self, _):
        if self.withdrawn:
            raise ValueError("source_withdrawn")
        return self.state.material

    def current(self):
        if self.withdrawn:
            raise ValueError("source_withdrawn")
        return self.state

    def image(self, _):
        return self.image_bytes

    def verify_image(self, binding):
        from packages.object_store import checksum

        if checksum(self.image_bytes) != binding.reference.checksum:
            raise ValueError("image_changed")

    def review_template(self, scope):
        from packages.evidence_details.review_tasks import project_review_tasks

        return project_review_tasks(self.current())


def review(scope, *, value=0.3):
    return validate(
        ReviewProposal,
        dict(
            scope_checksum=digest(scope),
            round="A",
            items=[
                dict(
                    assessment_id="item:one",
                    status="assessed",
                    observation_available=True,
                    observation_citations=["OBS:ONE"],
                    reference_citations=["SOURCE:ONE"],
                    comparison="Synthetic comparison",
                    observation=dict(value=0.7, reason="Synthetic O"),
                    relation=dict(value=0.6, reason="Synthetic L"),
                    applicability=dict(value=0.8, reason="Synthetic A"),
                    judgment=dict(
                        value=value,
                        lower=min(0.0, value),
                        upper=max(0.6, value),
                        reason="Synthetic e",
                        method="manual_source_bound_review",
                    ),
                    limitation_allocation="Synthetic limitation",
                    unresolved=[],
                )
            ],
        ),
    )


def budget_for(b, config=None, journal=None):
    config = config or load_provider_config(environ={})
    return RequestBudget(
        run_id=b.run_id,
        deadline=b.deadline,
        max_requests=config.max_requests,
        journal=journal or MemoryJournal(),
    )


def submission(reader, candidate, budget):
    s = reader.state
    return Submission(
        scope=s.material.scope,
        actor="PLANNER",
        correlation="SOURCE:AGENT_RESEARCH",
        request_id=budget.records()[-1].request_id,
        prompt_checksum=s.prompt_checksum,
        config_checksum=s.config_checksum,
        observation_checksums=tuple(o.reference for o in s.material.observations),
        source_checksums=tuple(o.reference for o in s.material.sources),
        candidate=candidate,
    )


class Documents:
    """Explicit memory Runtime double; PG tests use actual original repositories."""

    def __init__(self):
        self.saved = {}
        self.fail = None
        self.audit = []

    def session(self):
        return nullcontext()

    def validate_scope(self, scope):
        # Explicit in-memory owner; actual RuntimeDocuments checks persisted scope/graph.
        if scope.graph.kind != "research_artifact":
            raise ValueError("fake_domain_scope")

    def failure(self, key, sha, reason):
        self.audit.append("failure:" + key + sha + reason)

    def read(self, key):
        return copy.deepcopy(self.saved.get(key))

    def reference(self, key, index, value, *, role, version):
        return ref(
            role, "ASSET:FAKE_" + digest([key, index, value])[7:].upper(), version, digest(value)
        )

    def write(self, key, values):
        normalized = tuple(json.loads(canonical(v)) for v in values)
        if key in self.saved:
            if self.saved[key] != normalized:
                raise ValueError("agent_idempotency_conflict")
            return
        if self.fail == "transaction":
            raise ValueError("transaction_failure")
        if self.fail == "object":
            raise OSError("object_failure")
        self.saved[key] = normalized
        self.audit.append(key)
        if self.fail == "lost_reply":
            raise OSError("lost_reply")

    def consumed(self, graph):
        return any(
            v.get("schema_version") == "research-acceptance/1.0.0"
            and v["status"] == "accepted"
            and v["parent_graph"]["code"] == graph.code
            for values in self.saved.values()
            for v in values
        )


def generated(*, role="review", candidate=None, config=None):
    from agents.provider_agents import StructuredRoleAgent

    config = config or load_provider_config(environ={})
    reader, b = materials(role=role, config=config)
    candidate = candidate or review(reader.state.material.scope)
    provider = DeterministicFakeProvider(config, [candidate.model_dump(mode="json")])
    budget = budget_for(b, config)
    result = StructuredRoleAgent(role, provider=provider, materials=reader, budget=budget).call(
        reader.state.material.scope, deadline=b.deadline, round_id="A" if role == "review" else None
    )
    return reader, b, budget, submission(reader, result, budget)


def review_from_wire(body):
    """Synthetic ratings only; task and reference mapping must come from the sent payload."""
    from packages.llm_provider.models import Scope

    payload = json.loads(body["messages"][1]["content"][0]["text"])
    scope = validate(Scope, payload["material"]["scope"])
    raw = review(scope).model_dump(mode="json")
    raw["round"] = payload["round"]
    prototype = raw["items"][0]
    observations = {o["reference"]["code"]: o for o in payload["material"]["observations"]}
    raw["items"] = []
    for task in payload["review_tasks"]["tasks"]:
        o = observations[task["observation"]["code"]]
        row = copy.deepcopy(prototype)
        row.update(
            assessment_id=task["assessment_id"],
            observation_citations=[task["observation"]["code"]],
            reference_citations=[s["code"] for s in task["allowed_sources"]],
        )
        if o["status"] != "obtained":
            row.update(
                status="observation_ungradable" if o["status"] == "ungradable" else "input_missing",
                observation_available=False,
                judgment=dict(
                    value=None,
                    lower=None,
                    upper=None,
                    reason="Synthetic unresolved fact",
                    method="not_assessed",
                    missing_reason="Unknown acquisition",
                ),
            )
        raw["items"].append(row)
    return raw
