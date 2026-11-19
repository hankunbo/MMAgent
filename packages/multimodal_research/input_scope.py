"""Input survey phases in the original campaign, separate from disease votes.

Questions come from a frozen public policy and actual components/views. Dataset
names and expected diagnoses never select regions. A region is a question, not
an assertion of visible anatomy or a qualified measurement.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from apps.vectorengine_case_materials import ref
from apps.vectorengine_single_case import write_once
from packages.evaluation_preparation.dynamic_stages import TargetReader
from packages.grounding.models import Token, Value, canonical, digest, validate
from packages.llm_provider.input_scope import PROMPT, STORAGE_PROMPT, TEXT_PROMPT, ScopeTask
from packages.llm_provider.models import ObservationInput, Source

PREFIX = "input-scope:"


class RegionQuestions(Value):
    modality: Token
    representations: tuple[Literal["image", "image_series", "volume", "map"], ...] = Field(
        min_length=1, max_length=4
    )
    regions: tuple[Token, ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique(self):
        if len(set(self.regions)) != len(self.regions) or len(set(self.representations)) != len(
            self.representations
        ):
            raise ValueError("input_scope_policy_duplicate")
        return self


class ScopePolicy(Value):
    schema_version: Literal["input-scope-phase-policy/1.0.0"]
    max_components: int = Field(ge=1, le=8)
    questions: tuple[RegionQuestions, ...] = Field(min_length=1, max_length=32)
    procedure: str = Field(min_length=20, max_length=1000)

    @model_validator(mode="after")
    def unique(self):
        if len({q.modality for q in self.questions}) != len(self.questions):
            raise ValueError("input_scope_policy_modality_duplicate")
        return self


class ScopePolicyV2(ScopePolicy):
    schema_version: Literal["input-scope-phase-policy/1.1.0"]
    observation_storage: Literal["bound_candidate_reference"]


class ScopePolicyV3(ScopePolicyV2):
    schema_version: Literal["input-scope-phase-policy/1.2.0"]
    region_text_contract: Literal["bounded_full_text_512"]


def read_policy(reference):
    if set(reference) != {"path", "checksum"}:
        raise ValueError("input_scope_policy_reference")
    raw = json.loads(Path(reference["path"]).read_bytes())
    if digest(raw) != reference["checksum"]:
        raise ValueError("input_scope_policy_changed")
    cls = {
        "input-scope-phase-policy/1.0.0": ScopePolicy,
        "input-scope-phase-policy/1.1.0": ScopePolicyV2,
        "input-scope-phase-policy/1.2.0": ScopePolicyV3,
    }.get(raw.get("schema_version"), ScopePolicy)
    return validate(cls, raw)


def is_target(target):
    return isinstance(target, str) and target.startswith(PREFIX)


def project_plan(base, reference):
    from .active_acquisition import input_capabilities

    policy = read_policy(reference)
    prompt = (
        TEXT_PROMPT
        if isinstance(policy, ScopePolicyV3)
        else STORAGE_PROMPT
        if isinstance(policy, ScopePolicyV2)
        else PROMPT
    )
    owner = getattr(base, "active_owner", None)
    if owner is None or "requirement_catalog" not in owner.profile:
        raise ValueError("input_scope_requires_original_catalog_owner")
    capabilities = input_capabilities(base, owner.profile)
    views = base.views["views"]
    if len({v["component"] for v in views}) != len(views):
        raise ValueError("input_scope_view_inventory")
    entries, gaps = [], []
    for component in capabilities.components:
        question = next((q for q in policy.questions if q.modality == component.modality), None)
        if question is None or component.representation not in question.representations:
            continue
        view = next((v for v in views if v["component"] == component.component_id), None)
        if (
            component.input_status not in {"obtained", "partial"}
            or not view
            or view["status"] != "obtained"
        ):
            gaps.append(dict(component=component.component_id, reason="no_available_input_view"))
            continue
        if len(entries) >= policy.max_components:
            gaps.append(dict(component=component.component_id, reason="scope_component_budget"))
            continue
        task = ScopeTask(
            component_id=component.component_id,
            component_checksum=component.component_checksum,
            input_checksum=base.visible_value.components.checksum,
            modality=component.modality,
            representation=component.representation,
            requested_regions=question.regions,
            sampling_limits=(
                "Supplied view only; sampling metadata attached. "
                "No whole-volume or physical-scale claim."
            ),
        )
        entries.append(
            dict(
                target=PREFIX + digest(component.component_id)[7:31],
                task=task.model_dump(mode="json"),
                view_checksum=digest(view),
            )
        )
    return dict(
        schema_version="input-scope-phase-plan/1.0.0",
        input_checksum=base.manifest.checksum,
        capability_checksum=capabilities.checksum,
        policy=reference,
        prompt_version=prompt,
        prompt_checksum=digest(Path("configs/prompts/agents", prompt + ".txt").read_text()),
        entries=entries,
        gaps=gaps,
        disease_votes=0,
        estimates="model_estimates_not_pixel_measurements",
    )


def install(campaign):
    owner = getattr(campaign.base, "active_owner", None)
    reference = owner.profile.get("input_scope_policy") if owner else None
    if reference is None:
        return
    plan = project_plan(campaign.base, reference)
    write_once(campaign.output / "input-scope-plan.json", plan)
    campaign.base.input_scope_plan = plan
    campaign.scope_targets = tuple(e["target"] for e in plan["entries"])


def current_plan(base):
    saved = getattr(base, "input_scope_plan", None)
    if saved is None:
        raise ValueError("input_scope_target_not_installed")
    current = project_plan(base, saved["policy"])
    if (
        current != saved
        or json.loads((base.output / "input-scope-plan.json").read_bytes()) != saved
    ):
        raise ValueError("input_scope_plan_changed")
    return saved


def current_entry(base, target):
    if not is_target(target):
        raise ValueError("input_scope_target_not_installed")
    saved = current_plan(base)
    entries = [e for e in saved["entries"] if e["target"] == target]
    if len(entries) != 1:
        raise ValueError("input_scope_target_not_planned")
    return entries[0]


def configuration(config, plan):
    raw = config.model_dump(mode="json")
    role = next(r for r in raw["roles"] if r["role"] == "vision")
    role.update(prompt_version=plan["prompt_version"], prompt_checksum=plan["prompt_checksum"])
    # The same provider, model, timeouts, correction limits and request cap apply.
    return validate(type(config), raw)


class ScopeTargetReader(TargetReader):
    def __init__(self, owner, evidence):
        super().__init__(owner)
        self.evidence = evidence

    def input_scope_task(self, scope):
        state = self.owner.current()
        if state.material.scope != scope:
            raise ValueError("input_scope_owner_scope")
        self.evidence.verify(state)
        entry = current_entry(self.evidence.base, self.evidence.target)
        return validate(ScopeTask, entry["task"])


def projection(evidence):
    from . import acquisition_scope, material

    base, target = evidence.base, evidence.target
    entry = current_entry(base, target)
    task = validate(ScopeTask, entry["task"])
    policy = read_policy(base.input_scope_plan["policy"])
    source = Source(
        reference=ref("source", "POLICY:INPUT_SCOPE_PROCEDURE", digest(policy)),
        locator=base.input_scope_plan["policy"]["path"],
        purpose="research_material",
        state="research_only",
        text=policy.procedure,
    )
    definition = material.definition(
        target,
        "Input visibility and sampling scope survey; not a disease interpretation",
        "Describe only the requested visible regions and limitations. "
        "Do not infer disease, normality, physical calibration or whole-volume coverage.",
        (source,),
    )
    image, _, view = base.view(target)
    body = dict(
        status="unknown",
        method="input_scope_survey",
        unit="qualitative",
        value=canonical(
            {
                k: view[k]
                for k in (
                    "modality",
                    "indices",
                    "available_members",
                    "inspected_fraction",
                    "transform",
                    "limits",
                )
            }
        ).decode(),
        image=image,
    )
    observation = ObservationInput(
        reference=ref("observation", "OBS:SCOPE:" + digest(target)[7:19], digest(body)), **body
    )
    row = dict(
        id=target,
        component=task.component_id,
        category="input_scope",
        role="visual",
        definition=definition,
        input_modalities=(task.modality,),
        basis_kind="qualitative",
        missing_reason=None,
    )
    return material.state(
        context=evidence.context,
        visible=base.visible_value,
        root=base.root,
        config=evidence.config,
        medical={"scope_procedure": source},
        ledger=[row],
        observations={target: observation},
        resources=acquisition_scope.resources(base, target),
        acquisition=target,
    )


def accepted_fact(values, parent, result, task, control):
    """Shared exact accepted survey projection; no medical vote or dispatch grant.

    Callers read values/parent/Control from their original durable owners. Live
    consumption additionally reprojects current input in read_results; historical
    Memory archival checks the original campaign and task context instead.
    """
    from packages.evidence_details.acceptance_models import read_receipt, read_records
    from packages.evidence_details.input_scope import LiveScopeRecords, ScopeRecords

    records, receipt = read_records(values[0]), read_receipt(values[3])
    if (
        not isinstance(records, (ScopeRecords, LiveScopeRecords))
        or records.scope_task.model_dump(mode="json") != task
        or records.scope != parent.material.scope
        or records.scope.run_id != result["run_id"]
        or not result["accepted"]
        or receipt.status != "accepted"
        or receipt.records_checksum != digest(records)
        or len(records.observations) != 1
        or records.observations[0].image != parent.material.observations[0].image
        or records.observations[0].reference.code != parent.material.observations[0].reference.code
        or values[2]["scores"] is not None
        or control is None
        or control.pending is not None
        or control.phase != "stopped"
    ):
        raise ValueError("input_scope_acceptance_binding")
    return dict(
        phase=result,
        task=task,
        records_checksum=digest(records),
        receipt_checksum=digest(receipt),
        observation=records.observations[0].model_dump(mode="json"),
        regions=[r.model_dump(mode="json") for r in records.candidate.regions],
        disease_votes=0,
    )


def terminal_dispatch_phases(campaign, path, protocol, result):
    """Account for this window's surveys, separately from inherited facts.

    Case results store surveys outside ``phases``. Only the original Runtime
    phase at the deterministic current-window identity can add a request to
    this window. Current medical qualification remains read_results' job.
    """
    from db.runtime.repository.case_pipeline import identity

    scope = protocol.get("input_scope", {})
    targets = scope.get("targets", [])
    dispatched = scope.get("dispatch_targets", targets)
    if len(set(targets)) != len(targets) or len(set(dispatched)) != len(dispatched):
        raise ValueError("input_scope_accounting_duplicate_target")
    if not set(dispatched).issubset(targets):
        raise ValueError("input_scope_accounting_unknown_target")
    rows = []
    for target in dispatched:
        key = "mm-live:" + digest((Path(path).name, target))[7:55]
        run = identity("RUN", "task-research:" + key)
        saved = campaign.documents(run).read("dynamic-phase-result")
        if (
            not saved
            or len(saved) != 1
            or saved[0].get("target") != target
            or saved[0].get("run_id") != run
        ):
            raise ValueError("input_scope_accounting_original_phase")
        matching = [
            fact
            for fact in result.get("input_scopes", [])
            if fact.get("phase", {}).get("target") == target
        ]
        if len(matching) != 1 or matching[0]["phase"] != saved[0]:
            raise ValueError("input_scope_accounting_result_binding")
        rows.append(saved[0])
    return rows


def read_results(campaign, *, require_all=False):
    """Current owner readback, not authority inferred from a local JSON file."""
    from packages.evidence_details.acceptance_models import DomainState

    from .campaign import Phase

    rows = []
    plan = current_plan(campaign.base)
    if tuple(e["target"] for e in plan["entries"]) != campaign.scope_targets:
        raise ValueError("input_scope_plan_changed")
    config = configuration(campaign.config, plan)
    from .scope_successor import inherited

    reused = inherited(campaign)
    for target in campaign.scope_targets:
        entry = current_entry(campaign.base, target)
        path = campaign.phase_path(target) / "result.json"
        if target in reused:
            if path.exists():
                raise ValueError("input_scope_reused_target_dispatched")
            rows.append(reused[target])
            continue
        if not path.exists():
            if require_all:
                raise ValueError("input_scope_phase_not_finished")
            continue
        result = json.loads(path.read_bytes())
        docs = campaign.documents(result["run_id"])
        if docs.read("dynamic-phase-result") != (result,) or result["target"] != target:
            raise ValueError("input_scope_phase_owner_binding")
        if not result["accepted"]:
            if docs.read("recorded-input-scope:result:1") is not None:
                from .finalization import read_finalized_scope

                rows.append(read_finalized_scope(campaign, target))
                continue
            if require_all:
                raise ValueError("input_scope_phase_not_accepted")
            rows.append(dict(phase=result, observation=None))
            continue
        parent = validate(DomainState, docs.read("dynamic-phase-input")[0])
        context = campaign.contexts.read_research_run(result["run_id"])
        if context is None or context.request.scope.case_code != campaign.base.binding["case_code"]:
            raise ValueError("input_scope_original_run_binding")
        campaign.base.contexts[context.run_id] = context
        phase = Phase(campaign.base, target=target)
        expected, _ = phase.project(run_id=context.run_id, config=config)
        if expected != parent:
            raise ValueError("input_scope_current_parent_changed")
        phase.verify(parent)
        values = docs.read("accepted:" + result["acceptance_key"])
        rows.append(
            accepted_fact(
                values, parent, result, entry["task"], campaign.sessions.read(context.run_id)
            )
        )
    return rows


def execute(campaign):
    """Use original phase executor/budget; never retry an unaccepted phase here."""
    if not getattr(campaign, "scope_targets", ()):
        return []
    from .scope_successor import inherited

    reused = inherited(campaign)
    for target in campaign.scope_targets:
        if target in reused:
            print(
                json.dumps(dict(phase=target, accepted=True, reused=True, new_requests=0)),
                flush=True,
            )
            continue
        phase = campaign.execute_phase(target)
        print(
            json.dumps(dict(phase=target, accepted=phase["accepted"], error=phase["error"])),
            flush=True,
        )
        if not phase["accepted"]:
            raise ValueError("input_scope_phase_not_accepted")
    return read_results(campaign, require_all=True)


def attach_report(campaign, report, result):
    """Save current, exact survey provenance in the original six-section report."""
    if "input_scopes" not in result:
        return
    rows = read_results(campaign, require_all=True)
    if rows != result["input_scopes"] or result["input_scope_plan_checksum"] != digest(
        campaign.base.input_scope_plan
    ):
        raise ValueError("input_scope_report_result_binding")
    report["input_scope_assessment"] = dict(
        schema_version="input-scope-report/1.0.0",
        plan=campaign.base.input_scope_plan,
        records=rows,
        disease_votes=0,
        physical_calibration="not_established_by_visual_survey",
    )
    selection = result.get("scope_method_plan")
    if selection is not None:
        from .scope_selection import ensure

        if ensure(campaign, allow_native=False) != selection:
            raise ValueError("scope_method_report_result_binding")
        report["input_scope_assessment"]["method_plan"] = selection
    by_component = {r["task"]["component_id"]: (i, r) for i, r in enumerate(rows)}
    requirements = {r["id"]: r for r in campaign.base.ledger}
    for row in report["requirements"]:
        component = requirements.get(row["id"], {}).get("component")
        if component not in by_component:
            continue
        index, fact = by_component[component]
        evidence = dict(
            pointer="/input_scope_assessment/records/" + str(index),
            records_checksum=fact["records_checksum"],
            observation=fact["observation"]["reference"],
            regions=fact["regions"],
            purpose="input_visibility_context_not_additional_disease_vote",
        )
        row["input_scope_context"] = evidence
        if selection is not None:
            selected = next(r for r in selection["rows"] if r["id"] == row["id"])
            evidence["method_selection"] = selected
        if row.get("presentation_sections") is not None:
            row["presentation_sections"]["acquisition"]["input_scope_context"] = evidence
            row["presentation_sections"]["dependencies"]["input_scope_context"] = evidence
