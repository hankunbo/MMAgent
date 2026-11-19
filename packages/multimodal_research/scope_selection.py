"""Current input capabilities/scope -> append-only original campaign method plan.

This is an execution prerequisite projection, not a medical definition or vote.
Unknown visibility retains the question and requests local clarification; it
does not authorize a measurement or infer normality. Original owners still
validate inputs, sources, tools, observations and every actual dispatch.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from apps.vectorengine_single_case import write_once
from packages.grounding.models import Token, Value, canonical, digest, validate


class ScopeSelectionPolicy(Value):
    schema_version: Literal["input-scope-method-policy/1.0.0"]
    regions_by_question: dict[Token, tuple[Token, ...]] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def inventory(self):
        if any(
            not r or len(r) > 3 or len(set(r)) != len(r) for r in self.regions_by_question.values()
        ):
            raise ValueError("scope_selection_region_inventory")
        return self


def enabled(base):
    owner = getattr(base, "active_owner", None)
    return bool(owner and owner.profile.get("scope_selection_policy"))


def read_policy(profile):
    from .input_scope import read_policy as scope_policy

    ref = profile["scope_selection_policy"]
    if set(ref) != {"path", "checksum"}:
        raise ValueError("scope_selection_policy_reference")
    raw = json.loads(Path(ref["path"]).read_bytes())
    if digest(raw) != ref["checksum"]:
        raise ValueError("scope_selection_policy_changed")
    policy = validate(ScopeSelectionPolicy, raw)
    questions = {c["id"]: c for c in profile["checks"]}
    survey = {
        q.modality: set(q.regions) for q in scope_policy(profile["input_scope_policy"]).questions
    }
    for key, regions in policy.regions_by_question.items():
        question = questions.get(key)
        if (
            question is None
            or question["method"] not in {"vision", "disc_cup_measurement"}
            or not set(regions) <= survey.get(question["modality"], set())
        ):
            raise ValueError("scope_selection_question_binding")
    return policy


def question_inputs(base):
    # Quality and tool artifacts can be added after the selection, but the
    # medical questions/definitions and original component binding cannot drift.
    return [
        {
            k: row.get(k)
            for k in ("id", "component", "definition", "acquisition_method", "acquisition_allowed")
        }
        for row in base.ledger
    ]


def project(base, facts):
    """Pure projection; callers must obtain facts from read_results(original owners)."""
    policy = read_policy(base.active_owner.profile)
    by_component = {f["task"]["component_id"]: f for f in facts}
    if len(by_component) != len(facts):
        raise ValueError("scope_selection_duplicate_component")
    rows = []
    for row in base.ledger:
        method = row.get("acquisition_method", "vision")
        question = next(
            (
                q
                for q in policy.regions_by_question
                if row["id"] in {"active:" + q, "active:" + q + ":" + str(row.get("component"))}
            ),
            None,
        )
        required = policy.regions_by_question.get(question, ())
        fact = by_component.get(row.get("component"))
        regions = {r["region"]: r for r in fact["regions"]} if fact else {}
        selected = [regions[r] for r in required if r in regions]
        native = False
        if not row.get("component") or not row.get("acquisition_allowed", True):
            state, action = "missing_input", "retain_gap"
        elif method == "source_measurement":
            state, action = "source_measurement", "original_numeric_owner"
        elif method == "paired_observation":
            state, action = "pair_binding_pending", "original_pair_owner"
        elif not required or len(selected) != len(required):
            state, action = "scope_not_assessed", "vision_scope_clarification"
        elif any(r["visibility"] in {"uncertain", "not_identified"} for r in selected):
            state, action = "scope_clarification_needed", "vision_scope_clarification"
        else:
            state = (
                "partly_observable"
                if any(r["visibility"] == "partly_visible" for r in selected)
                else "observable_in_supplied_view"
            )
            native = method == "disc_cup_measurement"
            action = "current_native_owner_then_qualitative_fallback" if native else "scoped_vision"
        rows.append(
            dict(
                id=row["id"],
                component=row.get("component"),
                question=question,
                required_regions=list(required),
                regions=selected,
                state=state,
                action=action,
                native_candidate=native,
                scope_records_checksum=fact["records_checksum"] if fact else None,
                scope_observation=fact["observation"]["reference"] if fact else None,
                disease_votes=0,
            )
        )
    input_only = not base.input_scope_plan["entries"]
    return json.loads(
        canonical(
            dict(
                schema_version=(
                    "input-scope-method-plan/1.1.0"
                    if input_only
                    else "input-scope-method-plan/1.0.0"
                ),
                **(
                    {"storage_basis": "original_input_capabilities_without_visual_survey"}
                    if input_only
                    else {}
                ),
                input_checksum=base.manifest.checksum,
                target=base.task.target_code,
                parent_plan_checksum=digest(base.active_owner.plan),
                definition_revision_checksum=digest(getattr(base, "definition_revision", None)),
                question_checksum=digest(question_inputs(base)),
                policy=base.active_owner.profile["scope_selection_policy"],
                scope_plan_checksum=digest(base.input_scope_plan),
                scope_records=[
                    dict(
                        phase=f["phase"],
                        records_checksum=f["records_checksum"],
                        receipt_checksum=f["receipt_checksum"],
                    )
                    for f in facts
                ],
                rows=rows,
                medical_requirement_count=len(base.ledger),
                removed_requirements=[],
                added_disease_votes=0,
                physical_scale_authorized=False,
            )
        )
    )


def storage_identity(plan, campaign_id):
    """Exact original Runtime location, shared by storage and historical readback."""
    from db.runtime.repository.case_pipeline import identity

    key = "input-scope-method-plan:" + digest(plan)[7:]
    if plan["schema_version"] == "input-scope-method-plan/1.0.0" and plan["scope_records"]:
        return plan["scope_records"][0]["phase"]["run_id"], key
    if (
        plan["schema_version"] != "input-scope-method-plan/1.1.0"
        or plan.get("storage_basis") != "original_input_capabilities_without_visual_survey"
        or plan["scope_records"]
    ):
        raise ValueError("scope_selection_requires_accepted_survey")
    request_key = "mm-live:" + digest((campaign_id, "input-capability-plan:" + digest(plan)))[7:55]
    return identity("RUN", "task-research:" + request_key), key


class SelectionOwner:
    def __init__(self, campaign):
        self.campaign = campaign

    def current(self):
        from .input_scope import read_results

        c = self.campaign
        c.base.active_owner.check()
        return project(c.base, read_results(c, require_all=True))

    def input_context(self, plan, *, create=False):
        """Original task/input binding for metadata, never a survey acceptance."""
        from db.runtime.repository.case_pipeline import identity
        from packages.research_runtime.task_context import check_current_inputs, project_request

        c = self.campaign
        if (
            plan["schema_version"] != "input-scope-method-plan/1.1.0"
            or plan.get("storage_basis") != "original_input_capabilities_without_visual_survey"
            or plan["scope_records"]
            or c.base.input_scope_plan["entries"]
            or c.scope_targets
        ):
            raise ValueError("scope_selection_requires_accepted_survey")
        key = c.request_key("input-capability-plan:" + digest(plan))
        run_id = identity("RUN", "task-research:" + key)
        policy = dict(
            kind="policy",
            code="POLICY:INPUT_CAPABILITY_METHOD_PLAN",
            version="1.0.0",
            checksum=digest((plan, c.config)),
        )
        catalog = dict(
            kind="catalog", code="CATALOG:NO_NATIVE_TOOL", version="1.0.0", checksum=digest([])
        )
        if create:
            c.guard_control()
            context = c.runs.create(
                case_code=c.base.binding["case_code"],
                key=key,
                task=c.base.task,
                policy_ref=policy,
                catalog_ref=catalog,
                synthetic=c.synthetic,
            )
        else:
            context = c.contexts.read_research_run(run_id, missing_ok=True)
            if context is None:
                raise ValueError("scope_selection_saved_plan_missing")
        visible = check_current_inputs(context, c.runs.inputs)
        expected = project_request(
            visible, run_id=run_id, task=c.base.task, policy_ref=policy, catalog_ref=catalog
        )
        if (
            context.request != expected
            or context.synthetic != c.synthetic
            or expected.scope.case_code != c.base.binding["case_code"]
            or expected.input_checksum != c.base.manifest.checksum
        ):
            raise ValueError("scope_selection_input_context_binding")
        return context

    def location(self, plan, *, create=False):
        run_id, key = storage_identity(plan, self.campaign.campaign_id)
        if not plan["scope_records"]:
            context = self.input_context(plan, create=create)
            if context.run_id != run_id:
                raise ValueError("scope_selection_input_context_binding")
        docs = self.campaign.documents(run_id)
        path = self.campaign.output / "scope-selection" / (digest(plan)[7:] + ".json")
        return docs, key, path

    def save(self):
        plan = self.current()
        docs, key, path = self.location(plan, create=True)
        if docs.read(key) is None:
            self.campaign.guard_control()
            with docs.session():
                docs.write(key, (plan,))
        path.parent.mkdir(parents=True, exist_ok=True)
        write_once(path, plan)
        return self.check()

    def check(self):
        plan = self.current()
        docs, key, path = self.location(plan)
        if docs.read(key) != (plan,) or json.loads(path.read_bytes()) != plan:
            raise ValueError("scope_selection_current_binding")
        return plan

    def restore(self):
        """Require the committed Runtime plan; repair only its local mirror."""
        plan = self.current()
        docs, key, path = self.location(plan)
        if docs.read(key) != (plan,):
            raise ValueError("scope_selection_saved_plan_missing")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_once(path, plan)
        return self.check()


def ensure(campaign, *, allow_native=True):
    """Consume accepted survey facts before any new medical phase dispatch."""
    from .active_acquisition import install_native

    base = campaign.base
    if not enabled(base):
        return None
    owner = SelectionOwner(campaign)
    plan = owner.save() if allow_native else owner.restore()
    base.scope_selection_owner = owner
    eligible = frozenset(r["id"] for r in plan["rows"] if r["native_candidate"])
    # A new survey qualifies new execution; it does not erase a current,
    # owner-confirmed measurement already obtained from this exact input.
    from .native_fundus import records

    existing = records(base.active_owner.native)
    retained = frozenset(
        r["plan"]["target"]
        for r in existing
        if r.get("status") == "obtained" and r["plan"]["target"] in base.observations
    )
    base.scope_native_targets = eligible | retained
    if getattr(base, "scope_native_completed", None) == eligible:
        base.active_owner.check()
        campaign.targets = campaign.scoped_acquisition_inventory()
        return plan
    if existing and len(retained) == len(existing) and eligible <= retained:
        base.active_owner.check()
        base.scope_native_completed = eligible
        campaign.targets = campaign.scoped_acquisition_inventory()
        return plan

    def guard():
        owner.check()
        campaign.guard_control()
        base.check()

    native_runner = None
    if not allow_native:
        from .native_fundus import read_saved

        def native_runner(base, target, *, before_dispatch, **kwargs):
            return read_saved(base, target, **kwargs)

    native, methods = install_native(
        base,
        base.active_owner.profile,
        base.ledger,
        base.active_owner.plan,
        guard=guard,
        native_runner=native_runner,
        eligible_targets=eligible,
        read_only=not allow_native,
    )
    base.active_owner.native, base.active_owner.methods = native, methods
    base.active_owner.check()
    base.scope_native_completed = eligible
    campaign.targets = campaign.scoped_acquisition_inventory()
    return plan


def context_observations(base, target=None):
    from apps.vectorengine_case_materials import ref
    from packages.llm_provider.models import ObservationInput

    if not enabled(base):
        return ()
    owner = getattr(base, "scope_selection_owner", None)
    if owner is None:
        raise ValueError("scope_selection_before_medical_phase_required")
    plan = owner.check()
    result = []
    for item in plan["rows"]:
        if target is not None and item["id"] != target:
            continue
        if item["action"] in {"retain_gap", "original_numeric_owner", "original_pair_owner"}:
            continue
        context = dict(
            requirement=item["id"],
            state=item["state"],
            action=item["action"],
            scope_records_checksum=item["scope_records_checksum"],
            regions=[
                {k: r[k] for k in ("region", "visibility", "coverage", "clarity", "limitation")}
                for r in item["regions"]
            ],
            limits="Scope estimates only. Clarify uncertain regions; use interpretable portions. "
            "No normality, disease direction, physical scale or extra vote is implied.",
        )
        body = dict(
            status="obtained",
            method="input_scope_execution_context",
            unit="research_context",
            value=canonical(context).decode(),
        )
        result.append(
            ObservationInput(
                reference=ref(
                    "observation", "OBS:SCOPE_PLAN:" + digest(item["id"])[7:31], digest(body)
                ),
                **body,
            )
        )
    return tuple(result)
