"""Bounded registration and consumption; all owner I/O precedes short SQL transactions."""

from packages.grounding.models import digest, validate
from packages.grounding.query_models import QueryRequest
from packages.planner_control.models import Gap
from packages.tool_admission.models import AdmissionSelection, ResearchRequest
from packages.validation_recovery.models import ToolRequirement
from packages.validation_recovery.requirements import requirement_key
from packages.validation_recovery.service import scope

from .classification import SOURCE_BLOCKS, admission, grounding
from .models import Application, Finding, QueuePolicy


class GapQueue:
    def __init__(
        self,
        *,
        repository,
        policy=None,
        grounding_owner=None,
        admission_owner=None,
        contexts=None,
        planner=None,
        documents=None,
        resolver=None,
        memory=None,
        execution_reader=None,
        source_reader=None,
        grounding_request_reader=None,
    ):
        self.repository = repository
        self.policy = validate(QueuePolicy, policy or QueuePolicy())
        self.grounding, self.admission = grounding_owner, admission_owner
        self.contexts, self.planner, self.documents = contexts, planner, documents
        # Installed adapter returns original request coordinates, never a gap classification.
        self.resolver, self.memory = resolver, memory
        self.execution_reader = execution_reader
        self.source_reader = source_reader
        self.grounding_request_reader = grounding_request_reader

    @property
    def repository(self):
        return self._repository

    @repository.setter
    def repository(self, repository):
        from db.runtime.repository.tool_gap_queue import GapQueueRepository

        self._repository = (
            repository.with_owner(self)
            if isinstance(repository, GapQueueRepository)
            else repository
        )

    def prepare_grounding(self, request, *, key, selection=None):
        request = validate(QueryRequest, request)
        if self.grounding_request_reader is None:
            raise ValueError("gap_grounding_request_owner_required")
        owned = validate(
            QueryRequest,
            self.grounding_request_reader(request.lock.run_id, request.requirement_ref),
        )
        if owned != request:
            raise ValueError("gap_grounding_request_changed")
        result = self.grounding.query(request)
        material = dict(
            request=request.model_dump(mode="json"), result=result.model_dump(mode="json")
        )
        findings = grounding(result, self.policy)
        owner = "05"
        if selection is not None:
            selection = validate(AdmissionSelection, selection)
            admitted = self.admission.admit(request, result, selection)
            material.update(
                selection=selection.model_dump(mode="json"),
                admission=admitted.model_dump(mode="json"),
            )
            findings += admission(admitted, self.policy)
            blocked = {
                r.implementation_ref
                for r in admitted.rejected
                if r.category == "policy_rejection" or r.reason in SOURCE_BLOCKS
            }
            refused = {
                digest(f)
                for f in result.gaps
                if f.source and f.source.candidate.implementation_ref in blocked
            }
            refused.update(digest(r) for r in admitted.rejected if r.implementation_ref in blocked)
            findings = tuple(
                f.model_copy(update={"queueable": False}) if f.source_checksum in refused else f
                for f in findings
            )
            owner = "06_formal"
        return Application(
            run_id=request.lock.run_id,
            case_code=request.scope.case_code,
            key=key,
            owner=owner,
            purpose="decisive_research",
            policy=self.policy,
            requirement=request.requirement_ref.model_dump(mode="json"),
            catalog=request.lock.catalog_ref.model_dump(mode="json"),
            material=material,
            findings=findings,
        )

    def prepare_research(self, run_id, *, key):
        # Read registered request from original Runtime, not caller-supplied patient facts.
        context = self.contexts.read_research_run(run_id)
        if not context.synthetic:
            raise ValueError("gap_synthetic_research_required")
        request = validate(ResearchRequest, context.request)
        result = self.admission.admit_research(request)
        inputs = self.admission.inspect_research_gap_inputs(request)
        findings = admission(result, self.policy)
        if inputs["missing"]:
            findings = tuple(f.model_copy(update={"queueable": False}) for f in findings) + tuple(
                Finding(
                    kind="input_unsatisfied", source_checksum=digest(item), reason=item["reason"]
                )
                for item in inputs["missing"]
            )
        return Application(
            run_id=run_id,
            case_code=request.scope.case_code,
            key=key,
            owner="06_research",
            purpose=request.purpose,
            policy=self.policy,
            requirement=dict(code=request.code, version=request.version, checksum=request.checksum),
            catalog=request.lock.catalog_ref.model_dump(mode="json"),
            material=dict(
                context_checksum=context.checksum,
                request=request.model_dump(mode="json"),
                result=result.model_dump(mode="json"),
                inputs=inputs,
            ),
            findings=findings,
        )

    def prepare_execution(self, run_id, *, execution_key, key):
        """Original 06 read/confirmation only; never recover_commit, launch or retry."""
        from packages.tool_execution.models import WorkflowResult

        base = self.prepare_research(run_id, key=key)
        plan, executor = self.execution_reader(run_id, execution_key)
        self.contexts.read_research_run(run_id).validate_plan(plan)
        record = executor.recover(run_id, execution_key)
        workflow = None
        if record is not None and record["result"] is not None:
            workflow = executor.confirmed_result(
                plan, execution_key, validate(WorkflowResult, record["result"])
            )
        findings = (
            Finding(
                kind="execution_unknown",
                source_checksum=digest(record),
                reason="original_06_confirmation_required",
            ),
        )
        if workflow is not None:
            findings = tuple(
                Finding(
                    kind="environment_unavailable"
                    if step.error_code == "resource_limit"
                    else "execution_failed",
                    source_checksum=digest(step),
                    reason=step.error_code or "confirmed_no_failure",
                    capability=base.material["request"]["candidate"]["capability_ref"],
                    # Confirmed execution errors remain diagnostic in 11A. A fresh 05/06
                    # owner result is required to resolve an extensible cause, not an error string.
                    queueable=False,
                )
                for step in workflow.steps
                if step.error_code
            )
        return validate(
            Application,
            base.model_copy(
                update=dict(
                    owner="06_execution",
                    findings=findings,
                    material=dict(
                        base=base.model_dump(mode="json"),
                        execution_key=execution_key,
                        plan_checksum=plan.checksum,
                        attempt=record,
                        workflow=workflow.model_dump(mode="json") if workflow else None,
                    ),
                )
            ),
        )

    def _control(self, run_id):
        state = self.planner.read(run_id)
        self.documents.validate_scope(scope(state))
        if self.planner.owner.current(state.binding) != state.graph:
            raise ValueError("gap_current_graph_changed")
        return state

    def prepare_control(self, run_id, *, gap_checksum, key, intent_key=None):
        state = self._control(run_id)
        intent = None
        if intent_key is not None:
            saved = self.documents.read("recovery:" + intent_key)
            if not saved:
                raise ValueError("gap_intent_missing")
            intent = validate(ToolRequirement, saved[0])
            if (
                intent.origin_run_id != run_id
                or intent.binding_checksum != digest(state.binding)
                or requirement_key(intent) != intent_key
                or digest(intent.gap) != gap_checksum
                or intent.catalog != next(r for r in state.binding.locks if r.role == "catalog")
            ):
                raise ValueError("gap_intent_binding")
        gaps = state.gaps + ((intent.gap,) if intent else ())
        gap = next((validate(Gap, g) for g in gaps if digest(g) == gap_checksum), None)
        if gap is None:
            raise ValueError("gap_not_owned_by_control")
        blocked = (
            state.cancelled
            or state.pending is not None
            or any(e.outcome.status == "unknown" for e in state.events[-1:])
        )
        resolved = None
        source_proofs = []
        if intent and self.source_reader is not None:
            for ref in intent.sources:
                material = self.source_reader(state, ref)
                if digest(material) != ref.checksum:
                    raise ValueError("gap_intent_source_changed")
                source_proofs.append(ref.model_dump(mode="json"))
        if (
            self.resolver is not None
            and not blocked
            and (intent is None or (self.source_reader is not None and bool(intent.sources)))
        ):
            coordinates = self.resolver(state, gap)
            if coordinates is not None:
                if coordinates["owner"] == "06_research":
                    resolved = self.prepare_research(run_id, key=key)
                elif coordinates["owner"] == "05":
                    resolved = self.prepare_grounding(coordinates["request"], key=key)
                else:
                    raise ValueError("gap_owner_not_installed")
                if resolved.run_id != run_id or resolved.case_code != state.binding.case_code:
                    raise ValueError("gap_resolver_scope")
                catalog = next(r for r in state.binding.locks if r.role == "catalog")
                if any(
                    resolved.catalog.get(k) != getattr(catalog, k)
                    for k in ("code", "version", "checksum")
                ):
                    raise ValueError("gap_resolver_catalog")
                ref = gap.requirement_ref
                if ref is None or any(
                    resolved.requirement.get(k) != getattr(ref, k)
                    for k in ("code", "version", "checksum")
                ):
                    raise ValueError("gap_resolver_requirement")
                if gap.capability_code is not None and not any(
                    f.capability and f.capability["code"] == gap.capability_code
                    for f in resolved.findings
                ):
                    raise ValueError("gap_resolver_capability")
        fields = dict(
            run_id=run_id,
            case_code=state.binding.case_code,
            key=key,
            owner="09" if intent else "08",
            purpose="research_offline",
            policy=self.policy,
            requirement=gap.requirement_ref.model_dump(mode="json") if gap.requirement_ref else {},
            catalog=next(r for r in state.binding.locks if r.role == "catalog").model_dump(
                mode="json"
            ),
            material=dict(
                gap_checksum=gap_checksum,
                gap=gap.model_dump(mode="json"),
                intent_key=intent_key,
                resolved=resolved.model_dump(mode="json") if resolved else None,
                source_proofs=source_proofs,
            ),
            control=state.model_dump(mode="json"),
            intent=intent.model_dump(mode="json") if intent else None,
            findings=resolved.findings
            if resolved
            else (
                Finding(
                    kind="execution_unknown" if blocked else gap.kind,
                    source_checksum=digest(gap),
                    reason="original_06_confirmation_required"
                    if blocked
                    else "owner_resolution_required",
                ),
            ),
        )
        return Application(**fields)

    def recheck(self, application):
        a = validate(Application, application)
        required = {
            "05": (self.grounding, self.grounding_request_reader),
            "06_formal": (self.grounding, self.grounding_request_reader, self.admission),
            "06_research": (self.contexts, self.admission),
            "06_execution": (self.contexts, self.admission, self.execution_reader),
            "08": (self.planner, self.documents),
            "09": (self.planner, self.documents),
        }[a.owner]
        if any(owner is None for owner in required):
            raise ValueError("gap_owner_required")
        if a.policy != self.policy:
            raise ValueError("gap_policy_changed")
        if a.owner in {"05", "06_formal"}:
            current = self.prepare_grounding(
                a.material["request"], key=a.key, selection=a.material.get("selection")
            )
        elif a.owner == "06_research":
            current = self.prepare_research(a.run_id, key=a.key)
        elif a.owner == "06_execution":
            current = self.prepare_execution(
                a.run_id, key=a.key, execution_key=a.material["execution_key"]
            )
        else:
            current = self.prepare_control(
                a.run_id,
                key=a.key,
                gap_checksum=a.material["gap_checksum"],
                intent_key=a.material["intent_key"],
            )
            # Progress in unrelated Control steps need not invalidate historical registration.
            # Graph/source/version changes and pending/UNKNOWN still invalidate fresh consumption.
            if (
                current.control["binding"] == a.control["binding"]
                and current.control["graph"] == a.control["graph"]
                and current.findings == a.findings
                and not current.control["cancelled"]
                and current.control["pending"] is None
            ):
                current = current.model_copy(update={"control": a.control})
        if current != a:
            raise ValueError("gap_owner_material_changed")
        return a

    def register(self, application):
        return self.repository.register(application)

    def read(self, run_id, key):
        """Historical read is not authorization; also resolves uncertain registration replies."""
        return self.repository.read(run_id, key)

    def read_command(self, run_id, key, command_key):
        """Historical explanation only, even if the saved receipt says leased."""
        return self.repository.read_command(run_id, key, command_key)

    def report(self, run_id, key):
        """Deterministic historical explanation, not a complete v1 medical requirement."""
        receipt = self.read(run_id, key)
        if receipt is None:
            raise ValueError("gap_application_missing")
        a = receipt.application
        return dict(
            schema_version="tool-gap-report/1.0.0",
            task_id=receipt.task_id,
            application_checksum=digest(a),
            receipt=receipt.model_dump(mode="json"),
            audit_correlation="SOURCE:TOOL_GAP_QUEUE",
            historical_read_not_authorization=True,
            collection_required=any(
                f.kind in {"patient_data_missing", "input_unsatisfied"} for f in a.findings
            ),
            original_contracts_and_impacts=a.material,
            discovered_candidates=[],
            machine_gate=None,
            missing=list(a.missing),
            formal_document_prerequisites=[
                "formal_graph_and_node",
                "resolved_capability",
                "real_extension_task",
                "input_output_contracts",
                "honest_validation_plan",
            ],
            replay=a.replay,
            live_domain_acceptance=a.live_domain_acceptance,
        )

    def claim(self, run_id, key, *, command_key, worker):
        receipt = self.read(run_id, key)
        if receipt is None:
            raise ValueError("gap_application_missing")
        return self.repository.command(receipt, "claim", command_key, worker=worker)

    def acknowledge(self, run_id, key, *, command_key, lease_token, worker):
        receipt = self.read(run_id, key)
        return self.repository.command(
            receipt, "ack", command_key, lease_token=lease_token, worker=worker
        )

    def fail(self, run_id, key, *, command_key, lease_token, worker, reason):
        return self.repository.command(
            self.read(run_id, key),
            "fail",
            command_key,
            lease_token=lease_token,
            worker=worker,
            reason=reason,
        )

    def cancel(self, run_id, key, *, command_key, reason="cancelled_by_origin"):
        return self.repository.command(self.read(run_id, key), "cancel", command_key, reason=reason)

    def memory_diagnostics(self, run_id):
        if self.memory is None:
            return dict(status="not_requested", hits=[], extension_authority=False)
        try:
            result = self.memory.retrieve(run_id)
            document = result.document()
            hits = [h for values in result.groups.values() for h in values]
            if result.status != "technical_failure":
                self.memory.recheck(result.query, hits)
            return dict(status=result.status, receipt=document, extension_authority=False)
        except Exception:
            return dict(
                status="technical_failure",
                reason="memory_retrieval_unavailable",
                hits=[],
                extension_authority=False,
            )
