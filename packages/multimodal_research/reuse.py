"""Optional procedure-only Memory hints; patient observations never cross cases."""

from contextlib import nullcontext

from packages.grounding.models import digest

INVALID = {"memory_parent_invalid", "memory_source_withdrawn", "memory_source_missing"}
OPERATIONS = ("read_input", "review", "report")
CAPABILITY_POLICY = "proposition-input-method-reuse/1.0.0"
CAPABILITY_PROFILE = "configs/research/multimodal_catalog_acquisition_v6.json"
ROOT_CONTRACT = "root-input-method-reuse/1.0.0"
METHOD_GAPS = {
    "memory_method_context_missing",
    "memory_method_component_binding_missing",
    "memory_method_input_unavailable",
    "memory_method_current_requirement_missing",
}


def root_entry_contract(profile, task, input_checksum):
    """Explicit composition contract, not current patient/tool qualification."""
    if (
        not profile
        or profile.get("procedure_reuse_policy") != CAPABILITY_POLICY
        or not all(
            profile.get(key)
            for key in ("requirement_catalog", "input_scope_policy", "scope_selection_policy")
        )
        or task.target_code not in profile["goals"]
    ):
        raise ValueError("root_method_reuse_capability_profile")
    return dict(
        schema_version=ROOT_CONTRACT,
        policy=CAPABILITY_POLICY,
        profile_checksum=digest(profile),
        task_checksum=task.checksum,
        input_checksum=input_checksum,
        qualification="method_hints_require_current_input_source_and_tool_owners",
    )


def check_root_entry_contract(spec):
    """Validate before any credential/DB assembly; old root+reuse stays refused."""
    from packages.grounding.models import validate
    from packages.research_runtime.task_context import TaskIntent

    from .active_requirements import load

    proof = spec.get("procedure_reuse_contract")
    if proof is None:
        return False
    if (
        spec.get("execution_profile") != "relay-root-question-v1"
        or spec.get("reuse_procedures") is not True
        or not spec.get("active_acquisition")
        or spec.get("definition_failure_of")
    ):
        raise ValueError("root_method_reuse_entry_binding")
    profile = load(spec["active_acquisition"]["path"])
    if digest(profile) != spec["active_acquisition"]["checksum"] or proof != root_entry_contract(
        profile, validate(TaskIntent, spec["intent"]), spec["input_checksum"]
    ):
        raise ValueError("root_method_reuse_contract_changed")
    return True


def capability_policy(base):
    owner = getattr(base, "active_owner", None)
    value = owner.profile.get("procedure_reuse_policy") if owner else None
    if value not in {None, CAPABILITY_POLICY}:
        raise ValueError("memory_method_policy_version")
    return value


def targets_for(base, row):
    targets = [row["id"], "check:" + str(row["component"])]
    if getattr(base.task, "code", None) == "TASK:GLAUCOMA_FOUR_HYPOTHESES":
        from .hypotheses import LABELS, enabled

        enabled(base.task)  # Original registry, not caller-invented hypothesis labels.
        targets.extend(row["id"] + ":hyp:" + label for label, _, _ in LABELS)
    return tuple(dict.fromkeys(targets))


def repository(campaign):
    """Use the installed archive owner for both input and historical tool objects.

    The campaign's visible-input store alone cannot read every archived workflow.
    This is read-side legacy provenance reconstruction, never model installation
    or patient tool execution. Current methods still use this campaign's owners.
    """
    from apps.case_memory_store import PRIVATE, assembly

    if not PRIVATE.exists():
        return None
    reader, _ = assembly("reader")
    target = reader.target
    if (target.host, target.port, target.database) != (
        campaign.target.host,
        campaign.target.port,
        campaign.target.database,
    ):
        raise ValueError("multimodal_memory_target")
    return reader


class ProcedureReuse:
    def __init__(self, campaign, reader):
        self.campaign, self.reader, self.events = campaign, reader, []
        self.selected = {}
        self.capability_policy = capability_policy(campaign.base)
        self.method_bindings = {}

    def method_context(self, record, row):
        from packages.memory.execution_archive import ArchiveOwners
        from packages.memory.procedure_context import mismatch, query, read

        owners = self.reader.sources
        source = owners.artifacts if isinstance(owners, ArchiveOwners) else owners
        previous = read(
            source,
            record,
            inputs=self.campaign.base.inputs,
            contexts=self.campaign.contexts,
        )
        current = query(self.campaign.base, row)
        reason = mismatch(previous, current)
        if reason:
            return None, reason
        return dict(
            contract=previous["contract"],
            context_checksum=digest(previous),
            provenance=previous["provenance"],
        ), None

    def archive_read_scope(self):
        """One candidate/check only; no archive proof survives into dispatch."""
        from packages.memory.archive import ArchiveSources
        from packages.memory.execution_archive import ArchiveOwners

        owners = getattr(self.reader, "sources", None)
        source = owners.artifacts if isinstance(owners, ArchiveOwners) else owners
        return source.read_scope() if isinstance(source, ArchiveSources) else nullcontext()

    def prepare(self):
        base, plans = self.campaign.base, []
        self.selected.clear()
        self.method_bindings.clear()
        for row in base.ledger:
            targets = targets_for(base, row)
            modality = "+".join(row["input_modalities"])
            keys = (
                self.reader.procedure_keys(targets=targets, modality=modality, limit=3)
                if self.reader and row["component"] is not None
                else ()
            )
            choice, reasons = None, []
            for key in keys:
                with self.archive_read_scope():
                    try:
                        result = self.reader.read(key)
                    except ValueError as error:
                        if str(error) not in INVALID:
                            raise
                        reasons.append(dict(key=key, reason=str(error)))
                        continue
                    record = result.record
                    if record.case_code == base.binding["case_code"]:
                        reasons.append(dict(key=key, reason="same_case_not_external_experience"))
                        continue
                    if not self.compatible(record, targets, modality):
                        reasons.append(dict(key=key, reason="procedure_or_modality_not_installed"))
                        continue
                    method = None
                    if self.capability_policy:
                        try:
                            method, reason = self.method_context(record, row)
                        except ValueError as error:
                            if str(error) not in METHOD_GAPS:
                                raise
                            reason = str(error)
                        if reason:
                            reasons.append(dict(key=key, reason=reason))
                            continue
                    # Never expose artifact codes, patient scores or old clinical judgments.
                    choice = dict(
                        key=key,
                        checksum=result.checksum,
                        operations=list(OPERATIONS),
                        tool="RESEARCH:MULTIMODAL_TASK",
                        version="1.0.0",
                    )
                    self.selected[key] = (result.checksum, targets, modality)
                    if method is not None:
                        choice["method_context"] = method
                        self.method_bindings.setdefault(key, []).append((row, method))
                    break
            plans.append(
                dict(
                    requirement_id=row["id"],
                    selected=choice,
                    rejected=reasons,
                    origin="research_procedure_hint" if choice else "current_owner_baseline",
                )
            )
        self.plan = dict(
            schema_version="multimodal-procedure-reuse/2.0.0"
            if self.capability_policy
            else "multimodal-procedure-reuse/1.0.0",
            **({"capability_policy": self.capability_policy} if self.capability_policy else {}),
            plans=plans,
            input_checksum=base.manifest.checksum,
            task_checksum=base.task.checksum,
            old_patient_values_copied=False,
            clinical_evidence=False,
            published_planning_eligibility=False,
            search_limit_per_requirement=3,
        )
        self.plan_checksum = digest(self.plan)
        return self.plan

    def restore(self, plan):
        """Resume the same selection; do not silently query a newer Memory pool."""
        base = self.campaign.base
        if (
            plan.get("schema_version")
            != (
                "multimodal-procedure-reuse/2.0.0"
                if self.capability_policy
                else "multimodal-procedure-reuse/1.0.0"
            )
            or plan.get("capability_policy") != self.capability_policy
            or plan.get("input_checksum") != base.manifest.checksum
            or plan.get("task_checksum") != base.task.checksum
            or [p["requirement_id"] for p in plan["plans"]] != [r["id"] for r in base.ledger]
        ):
            raise ValueError("multimodal_reuse_restore_binding")
        self.selected.clear()
        self.method_bindings.clear()
        for saved, row in zip(plan["plans"], base.ledger, strict=True):
            choice = saved["selected"]
            if choice:
                if (choice["operations"], choice["tool"], choice["version"]) != (
                    list(OPERATIONS),
                    "RESEARCH:MULTIMODAL_TASK",
                    "1.0.0",
                ) or self.reader is None:
                    raise ValueError("multimodal_reuse_restore_port")
                targets = targets_for(base, row)
                self.selected[choice["key"]] = (
                    choice["checksum"],
                    targets,
                    "+".join(row["input_modalities"]),
                )
                if self.capability_policy:
                    if not choice.get("method_context"):
                        raise ValueError("multimodal_reuse_restore_method_context")
                    self.method_bindings.setdefault(choice["key"], []).append(
                        (row, choice["method_context"])
                    )
        self.plan, self.plan_checksum = plan, digest(plan)
        self.check()
        return self.plan

    @staticmethod
    def compatible(record, targets, modality):
        return (
            record.kind == "tool_experience"
            and record.detail.target in targets
            and record.detail.modality == modality
            and tuple(s.operation for s in record.detail.steps) == OPERATIONS
            and all(
                s.tool == "RESEARCH:MULTIMODAL_TASK" and s.version == "1.0.0"
                for s in record.detail.steps
            )
        )

    def check(self):
        from psycopg import OperationalError

        if digest(self.plan) != self.plan_checksum:
            raise ValueError("multimodal_reuse_plan_changed")
        for key, (expected, targets, modality) in tuple(self.selected.items()):
            try:
                with self.archive_read_scope():
                    result = self.reader.read(key)
                    if result.checksum != expected or not self.compatible(
                        result.record, targets, modality
                    ):
                        raise ValueError("memory_parent_invalid")
                    for row, saved in self.method_bindings.get(key, ()):
                        method, reason = self.method_context(result.record, row)
                        if reason or method != saved:
                            raise ValueError("memory_parent_invalid")
            except ValueError as error:
                if str(error) not in INVALID and not (
                    self.capability_policy and str(error) in METHOD_GAPS
                ):
                    raise
                self.events.append(
                    dict(key=key, reason=str(error), action="current_owner_baseline")
                )
                del self.selected[key]
            except (OSError, OperationalError) as error:
                self.events.append(
                    dict(
                        key=key,
                        reason="memory_technical_unavailable",
                        error_type=type(error).__name__,
                        action="current_owner_baseline",
                    )
                )
                del self.selected[key]
        # This is a hint to the already installed pipeline, not a new execution grant.
        return tuple(self.selected)

    def report(self):
        active = self.check()
        return dict(
            plan=self.plan,
            checks=self.events,
            active_keys=list(active),
            semantics="method_references_only_not_published_memory_or_patient_evidence",
        )
