"""Bounded application service over existing 03/04/05/06 and patient Runtime ports."""

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from apps.graph_versioning import GraphVersionAuditContext, build_graph_version_commit
from apps.ophthalmology_disease_packs import build_rule_package
from db.audit.repository import (
    AuditEvent,
    ConcurrencyConflictError,
    GraphVersionAppend,
    RunStatusChange,
)
from db.runtime.repository import RunRecord
from db.runtime.repository.case_pipeline import identity
from packages.contracts import (
    ImmutablePolicyReference,
    RunStatus,
    VersionedReference,
)
from packages.dataset_adapters.models import ObjectReference, VisibleManifest
from packages.graph_engine import (
    InstantiationContext,
    accepted_delta_event_from_payload,
    bind_artifacts,
    canonical_checksum,
    instantiate_template,
    plan_graph_delta,
    replay_accepted_deltas,
    replay_artifact_binding,
)
from packages.grounding import GroundingError
from packages.grounding.models import Ref
from packages.object_store import checksum
from packages.rule_engine import PropagationBudget, PropagationContext, snapshot_binding_for
from packages.tool_execution.models import WorkflowResult, validate_workflow_plan
from packages.tool_workflow import ExecutionPlan

from .demands import envelope, mapped_node_resolution
from .knowledge_rejection import KnowledgeRejection, knowledge_read, require_admitted
from .models import CaseInput, PipelineConfig, PipelineError
from .reporting import REPORT_VERSION, accepted_validations, observation_effects


def wire(value):
    def convert(v):
        if hasattr(v, "model_dump"):
            return v.model_dump(mode="json")
        if hasattr(v, "__dataclass_fields__"):
            return asdict(v)
        if isinstance(v, datetime):
            return v.isoformat()
        raise TypeError(type(v).__name__)

    return json.dumps(
        value, default=convert, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


class DeterministicCasePipeline:
    def __init__(
        self,
        *,
        visible_reader,
        runtime,
        audit,
        audit_reader,
        objects,
        disease,
        catalog,
        locks,
        mappings,
        tools,
        validator,
        configs,
        synthetic_test_mode=False,
        demand_resolver=None,
        queries=None,
    ):
        self.visible_reader = visible_reader
        self.runtime, self.audit, self.audit_reader = runtime, audit, audit_reader
        self.objects, self.disease, self.catalog = objects, disease, catalog
        self.locks, self.mappings, self.tools, self.validator = locks, mappings, tools, validator
        self.synthetic = synthetic_test_mode
        self.demand_resolver = demand_resolver
        self.queries = queries if queries is not None else tools
        self.configs = {
            c.checksum: PipelineConfig.model_validate_json(c.model_dump_json()) for c in configs
        }

    def run(self, case_code, *, input_checksum, config_checksum, key):
        if self.synthetic:
            raise PipelineError("real_input_forbidden_in_synthetic_service")
        try:
            v = self.visible_reader.read(case_code)
            v = VisibleManifest.model_validate_json(v.model_dump_json(warnings="error"))
        except Exception:
            raise PipelineError("visible_input_unavailable_or_invalid") from None
        if v.case.case_code != case_code or v.checksum != input_checksum:
            raise PipelineError("visible_input_version_conflict")
        return self._run(
            CaseInput(
                synthetic=False,
                case=v.case,
                pack_ref=v.pack_ref,
                input_checksum=v.checksum,
                visible=v,
            ),
            config_checksum,
            key,
        )

    def run_synthetic(self, case, *, pack_ref, config_checksum, key):
        if not self.synthetic:
            raise PipelineError("synthetic_input_not_enabled")
        return self._run(
            CaseInput(
                synthetic=True,
                case=case,
                pack_ref=pack_ref,
                input_checksum=canonical_checksum(case),
                visible=None,
            ),
            config_checksum,
            key,
        )

    def _put(self, value):
        data = wire(value)
        if not 0 < len(data) <= 4194304:
            raise PipelineError("trace_object_budget")
        staged = self.objects.stage(data)
        uri = self.objects.promote(staged, checksum(data))
        if self.objects.read_bounded(uri, len(data)) != data:
            raise PipelineError("trace_object_verification")
        return ObjectReference(uri=uri, checksum=checksum(data), byte_size=len(data))

    def _read(self, ref):
        data = self.objects.read_bounded(ref.uri, ref.byte_size)
        if len(data) != ref.byte_size or checksum(data) != ref.checksum:
            raise PipelineError("trace_object_checksum_mismatch")
        return json.loads(data)

    def _trace(self, key, state, kind, **data):
        if len(state.trace) >= 250:
            raise PipelineError("trace_budget_exhausted")
        ref = self._put(dict(kind=kind, **data))
        # Ambiguous DB failures preserve the promoted object. Retry reads committed state.
        return self.runtime.checkpoint(key, state, trace=(*state.trace, ref))

    def _admit(self, i, c, state):
        return self.disease.admit(
            i.pack_ref,
            c.query,
            c.context,
            c.retrieval,
            **(dict(runtime=self.runtime, run_id=state.run_id) if state.run_id else {}),
        )

    def _run(self, i, config_checksum, user_key):
        i = CaseInput.model_validate_json(i.model_dump_json())
        c = self.configs.get(config_checksum)
        if c is None or c.pack_ref != i.pack_ref:
            raise PipelineError("exact_registered_config_required")
        from packages.disease_packs.catalog import validate_prepared_case_contract

        pack = self.catalog.pack(i.pack_ref)
        validate_prepared_case_contract(pack, i.case)
        if pack.definition.synthetic != i.synthetic:
            raise PipelineError("input_knowledge_synthetic_scope_mismatch")
        if self.tools is not None and self.tools.reference not in c.service_refs:
            raise PipelineError("toolchain_version_not_locked")
        if self.validator.reference not in c.service_refs:
            raise PipelineError("validator_version_not_locked")
        if self.queries is not None and self.queries.reference not in c.service_refs:
            raise PipelineError("query_service_version_not_locked")
        with self.runtime.session(i.case.case_code, user_key) as key:
            state = self.runtime.begin(key, i, c)
            if state.report is not None:
                return self._read(state.report)
            if not state.trace:
                state = self._trace(key, state, "visible_input", input=i, config=c)
            try:
                admission = require_admitted(
                    knowledge_read("initial_admission", self._admit, i, c, state),
                    "initial_admission",
                )
            except KnowledgeRejection as rejection:
                state = self._record_knowledge_rejection(key, state, i, c, rejection)
                return self._finish(
                    key, state, i, c, "knowledge_not_admitted", gaps=[rejection.reason]
                )
            if c.mapping_refs and (
                self.mappings is None or self.locks is None or self.queries is None
            ):
                return self._finish(
                    key,
                    state,
                    i,
                    c,
                    "environment_error",
                    gaps=["mapped_query_dependency_unavailable"],
                )
            if (
                not c.mapping_refs
                and self.demand_resolver is not None
                and self.demand_resolver.reference not in c.service_refs
            ):
                raise PipelineError("demand_resolver_version_not_locked")
            if not c.mapping_refs and self.demand_resolver is None:
                return self._finish(
                    key, state, i, c, "environment_error", gaps=["demand_resolver_unavailable"]
                )
            if state.run_id is None:
                try:
                    resolved = (
                        knowledge_read(
                            "demand_resolution", self.demand_resolver.resolve, i, admission, c
                        )
                        if not c.mapping_refs
                        else None
                    )
                except KnowledgeRejection as rejection:
                    state = self._record_knowledge_rejection(key, state, i, c, rejection)
                    return self._finish(
                        key, state, i, c, "knowledge_not_admitted", gaps=[rejection.reason]
                    )
                run_id = identity("RUN", key)
                ctx = InstantiationContext(
                    graph_code=identity("GRAPH", key),
                    graph_version=1,
                    template_checksum=canonical_checksum(admission.template.graph_template),
                    created_at=state.started_at,
                    generator_version="07B.1.0.0",
                    source_codes=tuple(r.code for r in admission.pack.bindings.claim_refs),
                    resolutions=resolved[1]
                    if resolved is not None
                    else tuple(
                        mapped_node_resolution(slot.node_code, synthetic=i.synthetic)
                        for slot in admission.pack.definition.evidence_slots
                    ),
                    rule_versions=tuple(
                        r.to_versioned_reference() for r in admission.pack.bindings.rule_refs
                    ),
                    run_id=run_id,
                )
                graph = instantiate_template(
                    admission.template.graph_template, i.case, ctx
                ).instance
                refs = [
                    i.pack_ref,
                    *c.service_refs,
                    VersionedReference(
                        code="POLICY:PIPELINE_CONFIG",
                        version=c.version.rsplit("/", 1)[-1],
                        checksum=c.checksum,
                    ),
                    VersionedReference(
                        code="POLICY:PIPELINE_BUDGET",
                        version=c.policy.version.rsplit("/", 1)[-1],
                        checksum=c.policy.checksum,
                    ),
                    VersionedReference(
                        code=c.grounding.code,
                        version=c.grounding.version,
                        checksum=c.grounding.checksum,
                    ),
                ]
                policies = [
                    ImmutablePolicyReference(
                        policy_code=r.code, version=r.version, checksum=r.checksum
                    )
                    for r in refs
                ]
                if c.mapping_refs:
                    mp = self.mappings.policy
                    policies.append(
                        ImmutablePolicyReference(
                            policy_code=mp.code, version=mp.version, checksum=mp.checksum
                        )
                    )
                run = RunRecord(
                    run_id=run_id,
                    case_code=i.case.case_code,
                    graph_code=graph.graph_code,
                    graph_version=1,
                    tool_catalog_ref=VersionedReference(
                        **c.catalog_ref.model_dump(exclude={"kind"})
                    ),
                    policy_refs=tuple(policies),
                    template_ref=graph.template_ref,
                    rule_refs=ctx.rule_versions,
                )
                lock = (
                    self.locks.prepare(
                        run_id=run_id,
                        graph_code=graph.graph_code,
                        graph_version=1,
                        graph_checksum=canonical_checksum(graph),
                        references=c.mapping_refs,
                        mappings=self.mappings,
                        context=c.context,
                    )
                    if c.mapping_refs
                    else None
                )
                resolution = envelope(
                    input_value=i,
                    config=c,
                    admission=admission,
                    graph=graph,
                    lock=lock,
                    resolver=self.demand_resolver if resolved is not None else None,
                    resolved=resolved,
                )
                state = self.runtime.start_run(
                    key,
                    state,
                    graph,
                    run,
                    admission.pack.bindings.release_ref,
                    references=c.mapping_refs,
                    mappings=self.mappings,
                    context=c.context,
                    demand_resolution=resolution,
                    research_pack=i.pack_ref
                    if getattr(c.context, "use", None) is not None
                    else None,
                )
                state = self._trace(key, state, "instantiation", graph=graph)
            self._planning(state)
            if (
                state.demand_resolution is not None
                and state.demand_resolution.state == "unresolved"
            ):
                return self._unresolved(key, state, i, c)
            # Resume the last compiled action from durable plan/attempt facts before a new query.
            try:
                state = self._recover_actions(key, state, i, c)
            except PipelineError as error:
                state = self.runtime.read_request(key)
                return self._finish(key, state, i, c, str(error), gaps=[str(error)])
            gaps = []
            for round_index in range(c.policy.max_graph_rounds):
                graph = self.runtime.current_graph(state.run_id)
                if self._deadline_reached(state, c):
                    return self._finish(key, state, i, c, "time_budget", gaps=gaps)
                try:
                    admission = self._admit(i, c, state)
                except (ValueError, KeyError, RuntimeError):
                    return self._finish(
                        key,
                        state,
                        i,
                        c,
                        "knowledge_not_admitted",
                        gaps=[*gaps, "knowledge_live_read_rejected"],
                    )
                if admission.knowledge_gap:
                    return self._finish(
                        key, state, i, c, "knowledge_not_admitted", gaps=["knowledge_revoked"]
                    )
                try:
                    state = self._validate_results(key, state, i, c, admission, graph)
                    graph = self.runtime.current_graph(state.run_id)
                    package = build_rule_package(admission, graph)
                    pc = PropagationContext(
                        package.package_version,
                        package.package_checksum,
                        snapshot_binding_for(graph),
                        budget=PropagationBudget(
                            c.policy.max_graph_rounds,
                            round_index,
                            max(0, c.policy.max_actions - len(state.actions)),
                            True,
                        ),
                    )
                    _, propagation = self.disease.propagate_for_run(
                        i.pack_ref,
                        self.runtime,
                        state.run_id,
                        graph,
                        package,
                        pc,
                        c.query,
                        c.context,
                        c.retrieval,
                    )
                    # Prefer gathering available evidence before deriving an insufficient state.
                    goal = next(n for n in graph.nodes if n.node_code == graph.goal_node_code)
                    if goal.status.value in {"SUPPORTED", "REFUTED", "CONFLICTED"} and (
                        propagation.candidate_delta is not None
                        or propagation.stop_suggestion is not None
                    ):
                        if propagation.candidate_delta is not None:
                            self._commit_delta(graph, propagation.candidate_delta)
                            continue
                        return self._finish(
                            key,
                            state,
                            i,
                            c,
                            "graph_" + goal.status.value.lower(),
                            gaps=gaps,
                            components=propagation.evaluation,
                        )
                    # Evidence status commits can now propagate to the goal through 03.
                    if any(
                        n.status.value in {"SUPPORTED", "REFUTED", "CONFLICTED"}
                        for n in graph.nodes
                    ):
                        if propagation.candidate_delta is not None:
                            self._commit_delta(graph, propagation.candidate_delta)
                            continue
                    if (
                        len(state.actions) >= c.policy.max_actions
                        or state.cost >= c.policy.max_cost
                    ):
                        return self._finish(
                            key,
                            state,
                            i,
                            c,
                            "action_or_cost_budget",
                            gaps=gaps,
                            components=propagation.evaluation,
                        )
                    lock = self.locks.read(state.run_id)
                    artifacts = self.runtime.artifacts(i.case.case_code, state.run_id)
                    selected = None
                    for requirement in admission.pack.bindings.evidence_requirement_refs:
                        requirement_ref = Ref.model_validate_json(requirement.model_dump_json())
                        q, result = self.queries.query(
                            i, graph, lock, artifacts, requirement_ref, c, state.started_at
                        )
                        state = self._trace(
                            key,
                            state,
                            "grounding",
                            graph=snapshot_binding_for(graph),
                            result=result,
                        )
                        target_node = next(
                            n for n in graph.nodes if n.node_code == q.target_node_code
                        )
                        if target_node.status.value in {"SUPPORTED", "REFUTED"}:
                            continue
                        if result.existing_artifacts:
                            past = [self._read(r) for r in state.trace]
                            reusable = next(
                                (
                                    e
                                    for e in past
                                    if e["kind"] == "execution"
                                    and any(
                                        a["asset_code"] in result.existing_artifacts
                                        for step in e["workflow"]["steps"]
                                        for a in step["result"]["artifacts"]
                                    )
                                ),
                                None,
                            )
                            already = any(
                                e["kind"] == "execution" and e["target"] == q.target_node_code
                                for e in past
                            )
                            if reusable and not already:
                                state = self._trace(
                                    key,
                                    state,
                                    "execution",
                                    action=reusable["action"],
                                    target=q.target_node_code,
                                    requirement=requirement_ref,
                                    workflow=reusable["workflow"],
                                    reuse_only=True,
                                )
                                selected = ("recompute",)
                                break
                            gaps.append("existing_product_without_decisive_qualification")
                            continue
                        for finding in result.gaps:
                            gaps.append(finding.category)
                        # 05 already ranks candidates; preserve that order.
                        for candidate in result.candidates:
                            if self.tools is None:
                                gaps.append("execution_dependency_unavailable")
                                continue
                            action = self.tools.action_key(q.target_node_code, candidate)
                            if action in state.actions:
                                continue
                            cost = next(v for v in candidate.components if v.name == "cost")
                            # Unknown costs use the explicit baseline reservation.
                            charge = (
                                cost.value
                                if cost.state == "known"
                                else c.policy.unknown_cost_reservation
                            )
                            if state.cost + charge > c.policy.max_cost:
                                gaps.append("cost_budget")
                                continue
                            try:
                                plan = self.tools.compile(q, result, candidate)
                            except ValueError as error:
                                from packages.tool_admission.models import AdmissionError
                                from packages.tool_workflow.models import CompileError

                                reason = (
                                    str(error)
                                    if isinstance(
                                        error, (AdmissionError, CompileError, PipelineError)
                                    )
                                    else "invalid_tool_input"
                                )
                                state = self._trace(
                                    key,
                                    state,
                                    "tool_rejected",
                                    reason=reason,
                                    candidate_ref=candidate.identity,
                                )
                                gaps.append("tool_admission_or_compile_rejected")
                                continue
                            selected = (action, charge, plan, q.target_node_code, requirement_ref)
                            break
                        if selected:
                            break
                    if selected is None:
                        if propagation.candidate_delta is not None:
                            self._commit_delta(graph, propagation.candidate_delta)
                            graph = self.runtime.current_graph(state.run_id)
                        return self._finish(
                            key,
                            state,
                            i,
                            c,
                            "paths_exhausted",
                            gaps=gaps or ["insufficient_evidence"],
                            components=propagation.evaluation,
                        )
                    if selected[0] == "recompute":
                        continue
                    action, charge, plan, target, requirement = selected
                    # Action intent + complete plan persisted before any model can be reserved.
                    ref = self._put(
                        dict(
                            kind="plan",
                            action=action,
                            target=target,
                            requirement=requirement,
                            source_graph=snapshot_binding_for(graph),
                            bundle=json.loads(plan.to_bundle_json()),
                        )
                    )
                    state = self.runtime.checkpoint(
                        key,
                        state,
                        trace=(*state.trace, ref),
                        actions=(*state.actions, action),
                        cost=state.cost + charge,
                    )
                    state = self._recover_actions(key, state, i, c)
                except ConcurrencyConflictError:
                    # Validation trace may already be committed before the graph CAS fails.
                    state = self.runtime.read_request(key)
                    state = self._trace(
                        key,
                        state,
                        "graph_conflict",
                        discarded_base=snapshot_binding_for(graph),
                        decision="reload_and_recompute_no_validation_rebinding",
                    )
                    continue
                except GroundingError as error:
                    state = self.runtime.read_request(key)
                    category = (
                        "mapping_missing"
                        if error.reason == "requirement_mapping_missing"
                        else "grounding_blocked"
                    )
                    return self._finish(key, state, i, c, category, gaps=gaps + [error.reason])
                except PipelineError as error:
                    state = self.runtime.read_request(key)
                    return self._finish(key, state, i, c, str(error), gaps=gaps + [str(error)])
            return self._finish(key, state, i, c, "graph_round_budget", gaps=gaps)

    def _record_knowledge_rejection(self, key, state, i, c, rejection):
        if rejection.gap is not None:
            state = self._trace(
                key, state, "knowledge_gap", gap=asdict(rejection.gap), pack_ref=i.pack_ref
            )
        binding = self.catalog.pack(i.pack_ref).bindings
        return self._trace(
            key,
            state,
            "knowledge_rejection",
            phase=rejection.phase,
            reason=rejection.reason,
            error_type=rejection.error_type,
            binding_guard=rejection.binding_guard,
            pack_ref=i.pack_ref,
            release_ref=binding.release_ref if binding else None,
            requirement_refs=binding.evidence_requirement_refs if binding else (),
            config_checksum=c.checksum,
            demand_checksum=state.demand_resolution.checksum if state.demand_resolution else None,
            run_id=state.run_id,
        )

    def _reviewed_propagation(self, i, c, state, graph, budget, phase):
        admission = require_admitted(
            knowledge_read(phase + "_admission", self._admit, i, c, state),
            phase + "_admission",
        )
        package = knowledge_read(phase + "_rule_package", build_rule_package, admission, graph)
        pc = PropagationContext(
            package.package_version,
            package.package_checksum,
            snapshot_binding_for(graph),
            budget=budget,
        )
        admitted, propagation = knowledge_read(
            phase + "_propagation",
            self.disease.propagate_for_run,
            i.pack_ref,
            self.runtime,
            state.run_id,
            graph,
            package,
            pc,
            c.query,
            c.context,
            c.retrieval,
        )
        require_admitted(admitted, phase + "_propagation")
        return package, propagation

    def _unresolved(self, key, state, i, c):
        """Only 03 propagation/acceptance. This branch has no Tool/Validation calls."""
        resolution = state.demand_resolution
        if c.mapping_refs or state.actions:
            raise PipelineError("unresolved_execution_forbidden")
        gaps = sorted({r for d in resolution.demands for r in d.reasons})
        for round_index in range(c.policy.max_graph_rounds):
            graph = self.runtime.current_graph(state.run_id)
            if self._deadline_reached(state, c):
                return self._finish(key, state, i, c, "time_budget", gaps=gaps)
            try:
                package, propagation = self._reviewed_propagation(
                    i,
                    c,
                    state,
                    graph,
                    PropagationBudget(
                        c.policy.max_graph_rounds, round_index, c.policy.max_actions, True
                    ),
                    "unresolved",
                )
            except KnowledgeRejection as rejection:
                state = self._record_knowledge_rejection(key, state, i, c, rejection)
                return self._finish(
                    key, state, i, c, "knowledge_not_admitted", gaps=[*gaps, rejection.reason]
                )
            state = self._trace(
                key,
                state,
                "demand_propagation",
                demand_checksum=resolution.checksum,
                source_graph=snapshot_binding_for(graph),
                package=package,
                propagation=propagation,
                delta_reason="reviewed_propagation_candidate"
                if propagation.candidate_delta is not None
                else "no_status_change_required_by_reviewed_propagation",
                evidence_validation="not_applicable_no_executed_or_accepted_evidence",
            )
            if propagation.candidate_delta is not None:
                try:
                    self._commit_delta(graph, propagation.candidate_delta)
                except ConcurrencyConflictError:
                    state = self._trace(
                        key,
                        state,
                        "graph_conflict",
                        discarded_base=snapshot_binding_for(graph),
                        decision="reload_unresolved_propagation",
                    )
                continue
            reason = (
                "demands_unresolved"
                if resolution.catalog_state == "readable"
                else "environment_error"
            )
            if propagation.stop_suggestion is None:
                reason = "reviewed_stop_basis_missing"
            return self._finish(
                key, state, i, c, reason, gaps=gaps, components=propagation.evaluation
            )
        return self._finish(key, state, i, c, "graph_round_budget", gaps=gaps)

    @staticmethod
    def _deadline(state, config):
        return state.started_at + timedelta(seconds=config.policy.max_seconds)

    def _deadline_reached(self, state, config):
        return datetime.now(UTC) >= self._deadline(state, config)

    def _recover_actions(self, key, state, i, c):
        events = [self._read(r) for r in state.trace]
        complete = {e["action"] for e in events if e["kind"] == "execution"}
        stopped = {e["action"] for e in events if e["kind"] == "execution_not_started"}
        expired_unstarted = False
        for event in events:
            if event["kind"] != "plan" or event["action"] in complete:
                continue
            plan = ExecutionPlan.from_bundle_json(json.dumps(event["bundle"]))
            executor = self.tools.executor_for(plan)
            stored = executor.recover(state.run_id, event["action"])
            if stored is not None:
                if stored["result"] is None:
                    # Only 06's explicit recovery can finish a journal; it never relaunches.
                    try:
                        workflow = executor.recover_commit(plan, event["action"])
                    except (ValueError, RuntimeError):
                        raise PipelineError("execution_unknown_no_automatic_rerun") from None
                else:
                    workflow = validate_workflow_plan(
                        stored["result"], plan.document, stored["decision"]
                    )
            else:
                # The case deadline forbids new starts, not reads or 06 journal commits.
                # This also fences a durable plan resumed before its first reservation.
                if event["action"] in stopped or self._deadline_reached(state, c):
                    expired_unstarted = True
                    if event["action"] not in stopped:
                        state = self._trace(
                            key,
                            state,
                            "execution_not_started",
                            action=event["action"],
                            target=event["target"],
                            plan_checksum=plan.checksum,
                            reason="case_deadline_reached_before_execute",
                            deadline=self._deadline(state, c).isoformat(),
                        )
                    continue
                workflow = executor.execute(plan, event["action"])
            state = self._trace(
                key,
                state,
                "execution",
                action=event["action"],
                target=event["target"],
                requirement=event["requirement"],
                workflow=workflow,
            )
        if expired_unstarted:
            raise PipelineError("time_budget")
        return state

    def _validate_results(self, key, state, i, c, admission, graph):
        events = [self._read(r) for r in state.trace]
        plans = {
            e["action"]: ExecutionPlan.from_bundle_json(json.dumps(e["bundle"]))
            for e in events
            if e["kind"] == "plan"
        }
        for e in events:
            if e["kind"] != "execution":
                continue
            node = next(n for n in graph.nodes if n.node_code == e["target"])
            if node.status.value != "PENDING":
                continue
            from db.knowledge.models import KnowledgeReference

            requirement, rule = self.validator.rule(
                admission,
                KnowledgeReference.model_validate_json(json.dumps(e["requirement"])),
                c.context,
                c.query,
                c.retrieval,
            )
            workflow = WorkflowResult.model_validate_json(json.dumps(e["workflow"]))
            plan = plans[e["action"]]
            workflow = validate_workflow_plan(workflow, plan.document, workflow.decision)
            from packages.tool_admission.models import AdmissionError
            from packages.tool_workflow import CompileError

            try:
                workflow = self.tools.executor_for(plan).validate_evidence_source(
                    plan, e["action"], workflow
                )
            except (AdmissionError, CompileError) as error:
                if str(error) not in {
                    "stale_or_forged_admission_result",
                    "stale_or_modified_plan",
                    "admission_not_prepared",
                    "catalog_binding_mismatch",
                    "deployment_changed_during_compile",
                    "safety_changed_during_admission",
                }:
                    raise
                raise PipelineError("tool_source_not_current") from None
            root_steps = {
                s.step_id
                for s in plan.document.steps
                if s.candidate_ref in plan.document.selection.roots
            }
            for step in workflow.steps:
                if step.step_id not in root_steps:
                    continue
                from .tool_evidence import inspect_tool_evidence

                check = inspect_tool_evidence(step, plan)
                if not any(
                    x["kind"] == "tool_validation"
                    and x["action"] == e["action"]
                    and x["step_id"] == step.step_id
                    for x in events
                ):
                    state = self._trace(
                        key,
                        state,
                        "tool_validation",
                        action=e["action"],
                        target=e["target"],
                        step_id=step.step_id,
                        check=check,
                    )
                eligible, reason = (
                    self.validator.eligible(i, admission, rule, step, plan)
                    if check.evidence_candidate
                    else (False, check.reason)
                )
                classified = any(
                    x["kind"] == "observation_classification"
                    and x["action"] == e["action"]
                    and x["target"] == e["target"]
                    and x["step_id"] == step.step_id
                    for x in events
                )
                if not classified:
                    state = self._trace(
                        key,
                        state,
                        "observation_classification",
                        action=e["action"],
                        target=e["target"],
                        step_id=step.step_id,
                        eligible=eligible,
                        reason=reason,
                        source_graph=snapshot_binding_for(graph),
                    )
                if not eligible:
                    continue
                for artifact in step.result.artifacts:
                    if artifact.artifact_type_code != requirement.artifact_type_code:
                        continue
                    if artifact.asset_code not in node.evidence_bindings:
                        updated, event = bind_artifacts(
                            graph,
                            ((e["target"], artifact),),
                            expected_checksum=canonical_checksum(graph),
                            rule_ref=rule.reference().to_versioned_reference(),
                        )
                        self.audit.append_graph_version(
                            GraphVersionAppend(
                                run_id=state.run_id,
                                base_graph_version=graph.graph_version,
                                graph=updated,
                                audit_event=AuditEvent(
                                    event_id=identity("MESSAGE", event["event_checksum"]),
                                    event_type="GRAPH_VERSION_APPENDED",
                                    actor="GRAPH_ENGINE",
                                    case_code=i.case.case_code,
                                    run_id=state.run_id,
                                    graph_code=graph.graph_code,
                                    graph_version=updated.graph_version,
                                    correlation_code="SOURCE:DETERMINISTIC_PIPELINE",
                                    idempotency_key=identity("TASK", event["event_checksum"]),
                                    occurred_at=state.started_at,
                                    payload=event,
                                ),
                            )
                        )
                        graph = updated
                    bound = self.validator.evaluate(graph, e["target"], artifact, rule, step)
                    # Bind once where produced. Graph conflicts discard this package.
                    state = self._trace(key, state, "validation", validation=bound)
                    self._commit_delta(graph, bound.item.graph_delta_candidate, bound.item)
                    return state
        return state

    def _commit_delta(self, graph, delta, validation=None):
        plan = plan_graph_delta(graph, delta, validation_result=validation)
        commit = build_graph_version_commit(
            graph,
            plan,
            GraphVersionAuditContext(
                event_id=identity("MESSAGE", plan.plan_checksum),
                correlation_code="SOURCE:DETERMINISTIC_PIPELINE",
                occurred_at=graph.provenance.created_at,
            ),
        )
        self.audit.append_graph_version(commit.command)

    def _planning(self, state):
        status = self.runtime.run_status(state.run_id)
        if status == "CREATED":
            self._status(state, "NORMALIZING")
            status = "NORMALIZING"
        if status == "NORMALIZING":
            self._status(state, "PLANNING")

    def _status(self, state, target):
        old = self.runtime.run_status(state.run_id)
        if old == target:
            return
        token = state.run_id + ":" + str(state.revision) + ":" + old + ":" + target
        self.audit.transition_run_status(
            RunStatusChange(
                run_id=state.run_id,
                expected_status=RunStatus(old),
                new_status=RunStatus(target),
                audit_event=AuditEvent(
                    event_id=identity("MESSAGE", token),
                    event_type="RUN_STATUS_CHANGED",
                    actor="PLANNER",
                    case_code=state.case_code,
                    run_id=state.run_id,
                    correlation_code="SOURCE:DETERMINISTIC_PIPELINE",
                    idempotency_key=identity("TASK", token),
                    occurred_at=state.started_at,
                    payload=dict(from_status=old, to_status=target),
                ),
            )
        )

    def _finish(self, key, state, i, c, reason, *, gaps, components=None):
        graph = self.runtime.current_graph(state.run_id) if state.run_id else None
        if reason == "knowledge_not_admitted":
            # A refusal is terminal for this request. Do not retry admission or reuse a package.
            components = None
        elif graph is not None:
            try:
                _, current = self._reviewed_propagation(
                    i, c, state, graph, PropagationBudget(1, 0, 1, True), "final"
                )
                components = current.evaluation
                if reason.startswith("graph_") and current.stop_suggestion is None:
                    reason = "final_state_requires_reassessment"
            except KnowledgeRejection as rejection:
                state = self._record_knowledge_rejection(key, state, i, c, rejection)
                components = None
                reason = "knowledge_not_admitted"
                gaps = [*gaps, rejection.reason, "final_knowledge_revalidation_failed"]
        conclusion = {"graph_supported": "evidence_supported", "graph_refuted": "unsupported"}.get(
            reason, "insufficient"
        )
        if getattr(c.context, "use", None) is not None:
            conclusion = "insufficient"
        self.disease.validate_conclusion(i.pack_ref, conclusion)
        events = [self._read(r) for r in state.trace]
        observations = []
        validations = accepted_validations(
            graph,
            events,
            self.audit_reader.list_correlation("SOURCE:DETERMINISTIC_PIPELINE")
            if state.run_id
            else (),
            self.runtime.get_graph_snapshot,
        )
        plans = {
            e["action"]: ExecutionPlan.from_bundle_json(json.dumps(e["bundle"]))
            for e in events
            if e["kind"] == "plan"
        }
        for e in events:
            if e["kind"] != "execution":
                continue
            w = WorkflowResult.model_validate_json(json.dumps(e["workflow"]))
            for step in w.steps:
                observations.append(
                    dict(
                        target=e["target"],
                        attempt_id=w.attempt_id,
                        step_id=step.step_id,
                        plan_checksum=w.plan_checksum,
                        execution_status=step.result.status.value,
                        implementation=step.implementation_ref.model_dump(mode="json"),
                        qc_status=step.result.qc_status.value,
                        decisive_evidence=step.result.decisive_evidence,
                        synthetic=i.synthetic,
                        artifacts=[a.model_dump(mode="json") for a in step.result.artifacts],
                        **observation_effects(
                            graph,
                            e["target"],
                            step,
                            plans[e["action"]],
                            validations,
                            synthetic=i.synthetic,
                        ),
                        qc_records=[r.model_dump(mode="json") for r in step.qc_records],
                    )
                )
        evaluation = asdict(components) if components else None
        # Only the four registered components, never an invented scalar/probability.
        four = (
            {k: evaluation[k] for k in ("support", "refutation", "completeness", "conflict")}
            if evaluation
            else {k: None for k in ("support", "refutation", "completeness", "conflict")}
        )
        report = dict(
            schema_version=REPORT_VERSION,
            case_code=i.case.case_code,
            run_id=state.run_id,
            synthetic=i.synthetic,
            input_checksum=i.input_checksum,
            config_checksum=c.checksum,
            visible_manifest=i.visible.model_dump(mode="json") if i.visible else None,
            graph=graph.model_dump(mode="json") if graph else None,
            conclusion_status=conclusion,
            scope_limitations=[
                "evidence_planning_only",
                "no_definitive_poag",
                "no_treatment_orders",
            ],
            stop_reason=reason,
            gaps=sorted(set(gaps)),
            knowledge_gap_details=[e["gap"] for e in events if e["kind"] == "knowledge_gap"],
            knowledge_rejections=[e for e in events if e["kind"] == "knowledge_rejection"],
            demand_resolution=state.demand_resolution.model_dump(mode="json")
            if state.demand_resolution
            else None,
            demand_propagations=[e for e in events if e["kind"] == "demand_propagation"],
            observation_classifications=[
                e for e in events if e["kind"] == "observation_classification"
            ],
            failure_channels=sorted(
                {
                    e["reason"]
                    for e in events
                    if e["kind"] == "observation_classification" and not e["eligible"]
                }
            ),
            observations=observations,
            unstarted_actions=[e for e in events if e["kind"] == "execution_not_started"],
            execution_start_budget=dict(
                started_at=state.started_at.isoformat(),
                deadline=self._deadline(state, c).isoformat(),
                semantics="no_new_execute_at_or_after_deadline",
            ),
            components=four,
            components_source_graph=asdict(snapshot_binding_for(graph)) if graph else None,
            unresolved_nodes=[
                n.node_code for n in graph.nodes if n.status.value not in ("SUPPORTED", "REFUTED")
            ]
            if graph
            else [],
            trace=[r.model_dump(mode="json") for r in state.trace],
            actions=len(state.actions),
            cost=state.cost,
            network_policy="deny",
            execution_kind="synthetic_only" if i.synthetic else "real_or_none",
            cost_semantics="estimated_or_configured_reservation_units",
            graph_node_scope={str(n.node_code): dict(synthetic=i.synthetic) for n in graph.nodes}
            if graph
            else {},
            product_demands=[
                v
                for e in events
                if e["kind"] == "grounding"
                for v in e["result"]["product_demands"]
            ],
            tool_suggestions=[
                v for e in events if e["kind"] == "grounding" for v in e["result"]["suggestions"]
            ],
            real_medical_graph_complete=False,
        )
        if getattr(c.context, "use", None) is not None:
            report["knowledge_use"] = c.context.use.model_dump(mode="json")
            report["clinical_level"] = "pending_L4"
            report["scope_limitations"].append("limited_research_evidence_planning")
        report = json.loads(wire(report))
        ref = self._put(report)
        # Terminal status + report reference + Audit share the existing Runtime transaction.
        if state.run_id:
            if self.runtime.run_status(state.run_id) in {"PLANNING", "REPLANNING"}:
                self._status(state, "VERIFYING")
            target = {
                "graph_supported": "COMPLETED_SUPPORTED",
                "graph_refuted": "COMPLETED_REFUTED",
            }.get(reason, "COMPLETED_INSUFFICIENT_EVIDENCE")
            if getattr(c.context, "use", None) is not None:
                target = "COMPLETED_INSUFFICIENT_EVIDENCE"
            old = self.runtime.run_status(state.run_id)
            token = key + ":terminal:" + ref.checksum
            change = RunStatusChange(
                run_id=state.run_id,
                expected_status=RunStatus(old),
                new_status=RunStatus(target),
                audit_event=AuditEvent(
                    event_id=identity("MESSAGE", token),
                    event_type="RUN_STATUS_CHANGED",
                    actor="PLANNER",
                    case_code=state.case_code,
                    run_id=state.run_id,
                    correlation_code="SOURCE:DETERMINISTIC_PIPELINE",
                    idempotency_key=identity("TASK", token),
                    occurred_at=state.started_at,
                    payload=dict(from_status=old, to_status=target),
                ),
            )
            self.runtime.finish(key, state, ref, change)
        else:
            self.runtime.checkpoint(key, state, report=ref)
        return report

    def trace(self, case_code, key):
        state = self.runtime.read_request(self.runtime.key(case_code, key))
        if state is None:
            raise PipelineError("pipeline_request_missing")
        return tuple(self._read(ref) for ref in state.trace)

    def replay(self, run_id):
        """Replay binding and original 03 accepted events in Audit sequence order."""
        graph = self.runtime.current_graph(run_id)
        base = self.runtime.get_graph_snapshot(graph.graph_code, 1)
        events = (
            e
            for e in self.audit_reader.list_correlation("SOURCE:DETERMINISTIC_PIPELINE")
            if e.run_id == run_id
        )
        for event in events:
            if event.event_type.value != "GRAPH_VERSION_APPENDED":
                continue
            payload = event.payload
            if payload.get("payload_version") == "03_ARTIFACT_BINDING_V1":
                base = replay_artifact_binding(base, payload)
            else:
                accepted = accepted_delta_event_from_payload(payload["accepted_delta_event"])
                base = replay_accepted_deltas(base, (accepted,))
        if base != graph:
            raise PipelineError("graph_replay_mismatch")
        return base
