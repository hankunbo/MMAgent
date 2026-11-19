"""Source-rebuilt case and branch archives in the original Memory repository.

Historical workflow hints do not confer publication, execution or clinical authority.
Every query rechecks the original object bytes; new patient values are never copied.
"""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

from packages.grounding.models import digest
from packages.memory.archive_models import ArchiveRecord, ArtifactSource, archive_record_class
from packages.object_store import checksum

_reads = ContextVar("memory_archive_synchronous_reads", default=None)
MAX_READ_BYTES = 64 * 1024 * 1024
MAX_READ_KEYS = 512


class ArchiveSources:
    def __init__(self, *, runtime, read_bytes, read_run_case, read_view):
        self.runtime, self.read_bytes, self.read_run_case = runtime, read_bytes, read_run_case
        self.read_view = read_view

    @contextmanager
    def read_scope(self):
        """Synchronous read work only; no qualification survives this context.

        Every retained artifact is read and hashed by the original owners at
        entry and on successful return. Capacity overflow falls back to ordinary
        reads, not acceptance from a missing proof. No SQL transaction is held.
        """
        active = _reads.get()
        if active is not None and active["owner"] is self:
            yield
            return
        state = dict(owner=self, artifacts={}, cases={}, size=0)
        token = _reads.set(state)
        try:
            yield
            # Rechecks deliberately bypass this read scope. It is not a cache
            # of withdrawal, execution, Control or publication authorization.
            for (case, code), expected in state["artifacts"].items():
                if self._artifact(case, code) != expected:
                    raise ValueError("memory_source_changed_during_read")
            for run, case in state["cases"].items():
                if self.read_run_case(run) != case:
                    raise ValueError("memory_scope_changed_during_read")
        finally:
            _reads.reset(token)

    def _artifact(self, case, code):
        asset = self.runtime.read_case_artifact(case, code)
        if asset is None or asset.scope.case_code != case or not asset.research_only:
            raise ValueError("memory_source_binding")
        raw = self.read_bytes(asset)
        if checksum(raw) != asset.checksum:
            raise ValueError("memory_source_binding")
        return asset, raw

    def artifact(self, case, code):
        state = _reads.get()
        if state is None or state["owner"] is not self:
            return self._artifact(case, code)
        key = (case, code)
        if key in state["artifacts"]:
            asset, raw = state["artifacts"][key]
            return deepcopy(asset), raw
        asset, raw = self._artifact(case, code)
        if len(state["artifacts"]) < MAX_READ_KEYS and state["size"] + len(raw) <= MAX_READ_BYTES:
            state["artifacts"][key] = (deepcopy(asset), raw)
            state["size"] += len(raw)
        return asset, raw

    def case_for_run(self, run):
        state = _reads.get()
        if state is None or state["owner"] is not self:
            return self.read_run_case(run)
        if run in state["cases"]:
            return state["cases"][run]
        case = self.read_run_case(run)
        if len(state["cases"]) < MAX_READ_KEYS:
            state["cases"][run] = case
        return case

    def reference(self, *, case_code, run_id, code):
        if self.case_for_run(run_id) != case_code:
            raise ValueError("memory_scope")
        asset, _ = self.artifact(case_code, code)
        return ArtifactSource(
            case_code=case_code,
            run_id=run_id,
            key=code,
            object_code=code,
            checksum=digest(asset),
            object_checksum=asset.checksum,
            dependencies=(digest(["case_artifact", case_code, code]),),
            applicability=("USE:RESEARCH_WORKFLOW",),
            nature="source_fact",
        )

    def verify(self, source):
        if (
            self.reference(
                case_code=source.case_code, run_id=source.run_id, code=source.object_code
            )
            != source
        ):
            raise ValueError("memory_source_binding")

    def prepare(self, *, key, case_code, run_id, origin_code, kind, target="glaucoma"):
        root, raw = self.artifact(case_code, origin_code)
        if root.media_type != "application/json":
            raise ValueError("memory_source_binding")
        value = json.loads(raw)
        if root.artifact_type_code == "ARTIFACT:AGENT_RECORD":
            from .runtime_graph_archive import prepare

            return prepare(
                self,
                key=key,
                case_code=case_code,
                run_id=run_id,
                origin=root,
                envelope=value,
                kind=kind,
                target=target,
            )
        if (
            kind not in {"graph_snapshot", "tool_experience"}
            or root.artifact_type_code != "ARTIFACT:CASE_EVIDENCE_GRAPH"
        ):
            raise ValueError("memory_source_binding")
        view = self.read_view(case_code, origin_code)
        case = view["case_evidence"]
        if case["case_code"] != case_code:
            raise ValueError("memory_scope")
        # Caller only chooses a locator; no caller-provided workflow/score is saved.
        refs = {root.asset_code: root}
        # Original graph reader validates the graph/records/scores chain. Keep
        # exact current graph and visible input/output references, not text copies.
        for expected in [case["input"], *case["artifacts"]]:
            asset, _ = self.artifact(case_code, expected["asset_code"])
            if asset.model_dump(mode="json") != expected:
                raise ValueError("memory_source_binding")
            refs[asset.asset_code] = asset
        steps = []
        if kind == "tool_experience":
            if target != "vertical_cdr":
                raise ValueError("memory_source_binding")
            supplement = case.get("supplement", {})
            metrics = supplement.get("mask_check", {}).get("metrics", {})
            model = supplement.get("registered_tool")
            if "vcdr" not in metrics or not model:
                raise ValueError("memory_source_missing")
            masks = {a.extensions.get("output_port"): a for a in refs.values()}
            if not {"segman_disc", "segman_cup", "segman_observation"} <= masks.keys():
                raise ValueError("memory_source_binding")
            algorithm = next(
                (
                    s
                    for s in supplement["mask_check"]["algorithm_sources"]
                    if s["path"].endswith("/Algorithm_Tool/cdr_metrics/runner.py")
                ),
                None,
            )
            if algorithm is None:
                raise ValueError("memory_source_missing")
            steps = [
                dict(
                    operation="read_input",
                    tool="case_artifact_reader",
                    version="1.0.0",
                    implementation="original_runtime_reader",
                    source_checksum=case["input"]["checksum"],
                    input_roles=("current_case",),
                    output_roles=("fundus_image",),
                    artifact_codes=(case["input"]["asset_code"],),
                ),
                dict(
                    operation="segment",
                    tool=model["code"],
                    version=model["version"],
                    implementation=model["implementation"],
                    source_checksum=model["manifest_checksum"],
                    input_roles=("fundus_image",),
                    output_roles=("disc_mask", "cup_mask"),
                    artifact_codes=(
                        masks["segman_disc"].asset_code,
                        masks["segman_cup"].asset_code,
                    ),
                ),
                dict(
                    operation="measure",
                    tool="cdr_metrics",
                    version="historical_source_checksum",
                    implementation="Algorithm_Tool/cdr_metrics/runner.py",
                    source_checksum="sha256:" + algorithm["sha256"],
                    input_roles=("disc_mask", "cup_mask"),
                    output_roles=("vertical_cdr",),
                    artifact_codes=(masks["segman_observation"].asset_code,),
                ),
            ]
        outcome = "recorded"
        failure = None
        if kind == "recovery_experience":
            status = value.get("technical_status") or value.get("status")
            if status not in {"SUCCEEDED", "FAILED", "CANCELLED", "UNKNOWN"}:
                raise ValueError("memory_source_binding")
            outcome = "success" if status == "SUCCEEDED" else "failure"
            failure = status if outcome == "failure" else None
        sources = tuple(
            self.reference(case_code=case_code, run_id=run_id, code=code) for code in sorted(refs)
        )
        return archive_record_class(sources)(
            key=key,
            kind=kind,
            case_code=case_code,
            run_id=run_id,
            applicability=("USE:RESEARCH_WORKFLOW",),
            sources=sources,
            nature="technical_failure" if outcome == "failure" else "source_fact",
            detail=dict(
                type={
                    "graph_snapshot": "case_snapshot",
                    "tool_experience": "branch_workflow",
                    "recovery_experience": "execution_outcome",
                }[kind],
                target=target,
                modality="SLO",
                modality_source="historical_fairvlmed_slo_case",
                origin_code=origin_code,
                steps=tuple(steps),
                outcome=outcome,
                failure_reason=failure,
            ),
        )

    def verify_record(self, record):
        from packages.memory.writer_models import parse_record

        r = parse_record(record)
        if not isinstance(r, ArchiveRecord):
            raise ValueError("memory_archive_version")
        if (
            self.prepare(
                key=r.key,
                case_code=r.case_code,
                run_id=r.run_id,
                origin_code=r.detail.origin_code,
                kind=r.kind,
                target=r.detail.target,
            )
            != r
        ):
            raise ValueError("memory_projection_binding")


class WorkflowMemory:
    def __init__(self, repository):
        self.repository = repository

    def suggest(
        self, *, target, modality, legal_tools, query_case, limit=10, candidate_policy=None
    ):
        """Return procedure hints after current owner reads, never authorization.

        The legacy default retains its exact global catalog semantics. The explicit
        prefilter limits discovery by metadata before expensive current reads; every
        returned record still passes the same source, case and tool checks below.
        """
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("memory_archive_limit")
        if candidate_policy is None:
            keys = self.repository.archive_keys()
        elif candidate_policy == "target-modality-prefilter/1.0.0":
            keys = self.repository.procedure_keys(targets=(target,), modality=modality, limit=100)
            if (
                not isinstance(keys, (tuple, list))
                or len(keys) > 100
                or any(not isinstance(key, str) or not key for key in keys)
                or len(set(keys)) != len(keys)
            ):
                raise ValueError("memory_candidate_catalog")
        else:
            raise ValueError("memory_candidate_policy")
        matches, rejected = [], []
        for key in keys:
            result = self.repository.read(key)
            r = result.record
            if r.kind != "tool_experience":
                continue
            reason = None
            if r.case_code == query_case:
                reason = "same_case_not_external_experience"
            elif (r.detail.target, r.detail.modality) != (target, modality):
                reason = "target_or_modality_mismatch"
            elif any(
                s.tool not in legal_tools for s in r.detail.steps if s.operation != "read_input"
            ):
                reason = "current_tool_not_admitted"
            if reason:
                rejected.append(dict(key=key, reason=reason))
                continue
            matches.append(
                dict(
                    key=key,
                    checksum=result.checksum,
                    steps=[
                        dict(
                            operation=s.operation,
                            tool=s.tool,
                            input_roles=s.input_roles,
                            output_roles=s.output_roles,
                        )
                        for s in r.detail.steps
                    ],
                    reuse="procedure_only_recompute_values",
                    execution_authorized=False,
                    clinical_evidence=False,
                    publication_state=result.state,
                )
            )
        value = dict(matches=matches[:limit], rejected=rejected, clinical_evidence=False)
        if candidate_policy is not None:
            value["candidate_selection"] = dict(
                policy=candidate_policy,
                candidate_limit=100,
                candidates_read=len(keys),
                candidate_limit_reached=len(keys) == 100,
                metadata_is_authorization=False,
            )
        return value

    def recheck(self, key, expected_checksum):
        result = self.repository.read(key)
        if result.checksum != expected_checksum:
            raise ValueError("memory_storage_integrity")
        return result
