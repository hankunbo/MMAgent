"""Read controlled observations only. This service has no tool execute or graph writer port."""

import json
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

from packages.contracts import Artifact, VersionedReference
from packages.deterministic_pipeline.knowledge_rejection import KnowledgeRejection
from packages.deterministic_pipeline.reporting import accepted_validations
from packages.grounding.models import canonical, digest, seal, validate
from packages.object_store import checksum
from packages.rule_engine import PropagationBudget, snapshot_binding_for
from packages.tool_execution.models import WorkflowResult, validate_workflow_plan
from packages.tool_execution.qc import grayscale_png
from packages.tool_workflow import ExecutionPlan

from .models import (
    Component,
    EngineeringReceipt,
    GraphSnapshot,
    ScoreDefinition,
    ScoreEntry,
    ScoreEnvelope,
    ScoreScope,
    ScoreSource,
    TechnicalFact,
    ValidationFact,
)


@dataclass(frozen=True)
class ToolObservation:
    """Installed by 06/J assembly, never accepted by a result request."""

    key: str
    plan_json: str
    compiler: object
    outputs: tuple = ()  # (output asset type, definition, decoder, field)
    required_dependencies: tuple[str, ...] = ()
    optional_dependencies: tuple[str, ...] = ()
    model_versions: tuple = ()


def normalized_observations(observations):
    """Assembly sets, not precedence lists. The plan retains all 06 step/output scope.

    Within one observation, an output type + score code has exactly one full
    definition (including wire version), decoder and field mapping. Different
    observations may legitimately define the same code for different artifacts.
    """
    result = []
    for o in observations:
        outputs = {}
        for asset_type, definition, decoder, field in o.outputs:
            definition = validate(ScoreDefinition, definition)
            identity = (asset_type, definition.code)
            binding = (asset_type, definition, decoder, field)
            if identity in outputs and outputs[identity] != binding:
                raise ValueError("score_definition_conflict")
            outputs[identity] = binding
        models = {
            canonical(validate(VersionedReference, r)): validate(VersionedReference, r)
            for r in o.model_versions
        }
        result.append(
            replace(
                o,
                outputs=tuple(outputs[k] for k in sorted(outputs)),
                required_dependencies=tuple(sorted(set(o.required_dependencies))),
                optional_dependencies=tuple(sorted(set(o.optional_dependencies))),
                model_versions=tuple(models[k] for k in sorted(models)),
            )
        )
    return tuple(sorted(result, key=lambda o: o.key))


class PipelineGraphReader:
    """07 owner joins saved request/report to CURRENT 04 admission and original 03 evaluation."""

    def __init__(self, pipeline, request_key):
        self.pipeline, self.key = pipeline, request_key

    def read(self, run_id):
        p = self.pipeline
        state = p.runtime.read_request(self.key)
        if state is None or state.run_id != run_id or state.report is None:
            raise ValueError("score_pipeline_request_not_finished")
        report = p._read(state.report)
        config = p.configs.get(state.config_checksum)
        graph = p.runtime.current_graph(run_id)
        if config is None or (
            report["run_id"],
            report["case_code"],
            report["input_checksum"],
            report["config_checksum"],
            report["synthetic"],
        ) != (
            run_id,
            state.case_code,
            state.input_checksum,
            state.config_checksum,
            state.synthetic,
        ):
            raise ValueError("score_report_identity")
        if report["schema_version"] not in {
            "deterministic-case-report/1.0.0",
            "deterministic-case-report/2.0.0",
        }:
            raise ValueError("score_report_version")
        events = [p._read(ref) for ref in state.trace]
        validations = accepted_validations(
            graph,
            events,
            p.audit_reader.list_correlation("SOURCE:DETERMINISTIC_PIPELINE"),
            p.runtime.get_graph_snapshot,
        )
        components = None
        if validations:
            try:
                _, result = p._reviewed_propagation(
                    SimpleNamespace(pack_ref=config.pack_ref),
                    config,
                    state,
                    graph,
                    PropagationBudget(1, 0, 1, True),
                    "score_read",
                )
                components = asdict(result.evaluation)
            except KnowledgeRejection:
                validations = []
        return graph, components, validations, state.report, state.synthetic


def definition(metric):
    layer = "proposition" if metric in {"support", "refutation"} else "graph"
    return ScoreDefinition(
        code="03." + metric,
        layer=layer,
        metric=metric,
        definition="Original 03 EvidenceComponent; counts only, not probability",
        unit="03_counts",
        direction="owner_counts",
    )


def read_object(objects, ref):
    raw = objects.read_bounded(ref.uri, ref.byte_size)
    if len(raw) != ref.byte_size or checksum(raw) != ref.checksum:
        raise ValueError("score_object_checksum")
    return raw


class CaseScoreService:
    def __init__(
        self,
        *,
        runtime,
        executions,
        objects,
        observations,
        synthetic,
        graph_reader=None,
        engineering_stop=None,
        context_version="clinical-run/1.0.0",
    ):
        self.runtime, self.executions, self.objects = runtime, executions, objects
        self.observations = normalized_observations(observations)
        self.synthetic, self.graph_reader = synthetic, graph_reader
        self.engineering_stop = engineering_stop
        if context_version not in {"clinical-run/1.0.0", "research-run/1.0.0"}:
            raise ValueError("score_context_version")
        if context_version == "research-run/1.0.0" and (
            graph_reader is not None or engineering_stop is not None
        ):
            raise ValueError("research_score_clinical_port_forbidden")
        self.context_version = context_version
        keys = [o.key for o in self.observations]
        if len(keys) != len(set(keys)) or len(keys) > 128:
            raise ValueError("score_duplicate_observation")
        for o in self.observations:
            if (
                set(o.required_dependencies) & set(o.optional_dependencies)
                or o.key in (*o.required_dependencies, *o.optional_dependencies)
                or not set((*o.required_dependencies, *o.optional_dependencies)) <= set(keys)
            ):
                raise ValueError("score_dependency_registry")

            # Dependency loops cannot silently erase results or improve coverage.
            def visit(key, seen):
                if key in seen:
                    raise ValueError("score_dependency_cycle")
                entry = next(v for v in self.observations if v.key == key)
                for dep in entry.required_dependencies:
                    visit(dep, seen | {key})

            visit(o.key, set())
            for _, d, decoder, _ in o.outputs:
                d = validate(ScoreDefinition, d)
                if (
                    decoder == "fairclip-output/1.0.0"
                    and self.context_version != "research-run/1.0.0"
                ):
                    raise ValueError("fairclip_research_score_only")
                if decoder == "fairclip-output/1.0.0":
                    from packages.fairclip_research.core import WEIGHT_SHA256

                    if not any(v.checksum == "sha256:" + WEIGHT_SHA256 for v in o.model_versions):
                        raise ValueError("fairclip_fixed_model_version_required")
                if (
                    d.metric in {"logit", "segmentation_confidence", "uncertainty"}
                    and not synthetic
                    and not o.model_versions
                ):
                    raise ValueError("score_model_resource_versions_required")
                allowed = {
                    "png_foreground_pixels/1.0.0": {"measurement"},
                    "synthetic_png_logit/1.0.0": {"logit"},
                    "disc-geometry/1.0.0": {"measurement"},
                    "fairclip-output/1.0.0": {"logit"},
                    "scalar-score-output/1.0.0": {
                        "logit",
                        "quality",
                        "measurement",
                        "segmentation_confidence",
                        "uncertainty",
                    },
                }
                if (
                    decoder not in allowed
                    or d.metric not in allowed[decoder]
                    or decoder.startswith("synthetic_")
                    and not synthetic
                ):
                    raise ValueError("score_decoder_not_qualified")
        self.policy_checksum = digest(self._policy())

    def _policy(self):
        return dict(
            **(
                dict(version="score-assembly/1.1.0")
                if self.context_version == "clinical-run/1.0.0"
                else dict(version="score-assembly/1.2.0", context_version=self.context_version)
            ),
            synthetic=self.synthetic,
            engineering_stop_checksum=self.engineering_stop[1] if self.engineering_stop else None,
            observations=[
                dict(
                    key=o.key,
                    plan=ExecutionPlan.from_bundle_json(o.plan_json).document.checksum,
                    outputs=o.outputs,
                    required=o.required_dependencies,
                    optional=o.optional_dependencies,
                    models=o.model_versions,
                )
                for o in normalized_observations(self.observations)
            ],
        )

    def task_requirements(self, plan, key):
        """Freeze all configured result definitions before the single tool executes."""
        if digest(self._policy()) != self.policy_checksum:
            raise ValueError("score_registry_drift")
        if self.context_version != "research-run/1.0.0" or len(self.observations) != 1:
            raise ValueError("task_completion_score_profile")
        observation = self.observations[0]
        if (
            observation.key != key
            or observation.plan_json != plan.to_bundle_json()
            or observation.required_dependencies
            or observation.optional_dependencies
            or not observation.outputs
        ):
            raise ValueError("task_completion_score_profile")
        import json

        from packages.grounding.models import canonical

        return dict(
            policy=self.policy_checksum,
            outputs=json.loads(canonical(observation.outputs)),
            models=json.loads(canonical(observation.model_versions)),
        )

    def build(self, run_id):
        """The only request input is the existing run ID; no values/checksums/approvals."""
        if digest(self._policy()) != self.policy_checksum:
            raise ValueError("score_registry_drift")
        observations = normalized_observations(self.observations)
        context = None
        if self.context_version == "research-run/1.0.0":
            context = self.runtime.read_research_run(run_id)
            if context.synthetic != self.synthetic:
                raise ValueError("score_synthetic_mismatch")
            graph = None
            case = self.runtime.get_case_bundle(context.request.scope.case_code)
            if (case.encounter_code, case.patient_hash, case.laterality) != (
                context.request.scope.encounter_code,
                context.request.scope.patient_hash,
                context.request.scope.laterality,
            ):
                raise ValueError("score_run_scope")
        else:
            graph = self.runtime.current_graph(run_id)
            case = self.runtime.get_case_bundle(graph.case_code)
        scope = ScoreScope(
            case_code=case.case_code,
            run_id=run_id,
            encounter_code=case.encounter_code,
            patient_hash=case.patient_hash,
            laterality=case.laterality,
        )
        if graph is not None and (
            graph.scope.run_id != run_id or graph.case_code != scope.case_code
        ):
            raise ValueError("score_run_scope")
        status = self.runtime.run_status(run_id)
        artifacts = {
            a.asset_code: validate(Artifact, a)
            for a in self.runtime.artifacts(scope.case_code, run_id)
        }
        observed, technical = {}, []
        for o in observations:
            plan = ExecutionPlan.from_bundle_json(o.plan_json)
            doc = plan.document
            if context is not None:
                context.validate_plan(plan)
            if (
                doc.run_id,
                doc.scope.case_code,
                doc.scope.encounter_code,
                doc.scope.patient_hash,
                doc.scope.laterality,
            ) != (
                run_id,
                scope.case_code,
                scope.encounter_code,
                scope.patient_hash,
                scope.laterality,
            ):
                raise ValueError("score_plan_scope")
            if any(s.deployment.allowlist.synthetic_only != self.synthetic for s in doc.steps):
                raise ValueError("score_synthetic_mismatch")
            # Recompile through 06B current safety, resource, input and version checks.
            o.compiler.validate_plan(plan)
            attempt = self.executions.recover(run_id, o.key)
            workflow = None
            state = "NOT_EXECUTED" if attempt is None else "UNKNOWN_JOB"
            if attempt is not None and attempt["status"] == "CONFIRMED":
                workflow = validate_workflow_plan(
                    validate(WorkflowResult, attempt["result"]), doc, attempt["decision"]
                )
                state = workflow.status
            observed[o.key] = (o, plan, workflow, state)
            technical.append(
                TechnicalFact(
                    key=o.key,
                    status=state,
                    qc_statuses=tuple(s.result.qc_status.value for s in workflow.steps)
                    if workflow
                    else (),
                    required_dependencies=o.required_dependencies,
                    optional_dependencies=o.optional_dependencies,
                    attempt_id=workflow.attempt_id if workflow else None,
                )
            )
        entries, seen = [], {}
        base = dict(scope=scope, synthetic=self.synthetic, policy_checksum=self.policy_checksum)

        def healthy(key):
            o, _, w, _ = observed[key]
            return (
                w is not None
                and w.status == "SUCCEEDED"
                and all(healthy(d) for d in o.required_dependencies)
            )

        for o, plan, workflow, state in observed.values():
            dependency_failed = not all(healthy(k) for k in o.required_dependencies)
            if workflow:
                for step in workflow.steps:
                    for artifact in step.result.artifacts:
                        registered = artifacts.get(artifact.asset_code)
                        if registered != artifact or (
                            artifact.scope.case_code,
                            artifact.scope.run_id,
                            artifact.laterality,
                        ) != (scope.case_code, run_id, scope.laterality):
                            raise ValueError("score_artifact_scope_or_fact")
                        from packages.dataset_adapters.models import ObjectReference

                        raw = self.objects.read_bounded(artifact.uri, 4194304)
                        if checksum(raw) != artifact.checksum:
                            raise ValueError("score_artifact_bytes")
                        roots = self._roots(artifact, artifacts, scope)
                        source = ScoreSource(
                            **base,
                            qualification="research_output",
                            correlation_group=digest(roots),
                            artifact_code=artifact.asset_code,
                            artifact_checksum=artifact.checksum,
                            object_ref=ObjectReference(
                                uri=artifact.uri, checksum=artifact.checksum, byte_size=len(raw)
                            ),
                            attempt_id=workflow.attempt_id,
                            step_id=step.step_id,
                            step_checksum=digest(step),
                            plan_checksum=plan.document.checksum,
                            qc_policy_checksums=tuple(q.policy_checksum for q in step.qc_records),
                            implementation=step.implementation_ref,
                            model_versions=o.model_versions,
                        )
                        for asset_type, d, decoder, field in o.outputs:
                            d = validate(ScoreDefinition, d)
                            if asset_type != artifact.artifact_type_code:
                                continue
                            identity = (artifact.asset_code, d.code)
                            binding = digest((d, decoder, field))
                            attribution = digest((source, dependency_failed))
                            if identity in seen:
                                if seen[identity][0] != binding:
                                    raise ValueError("score_definition_conflict")
                                if seen[identity][1] != attribution:
                                    raise ValueError("score_source_identity_conflict")
                                continue
                            seen[identity] = (binding, attribution)
                            reason = (
                                "required_dependency_failed"
                                if dependency_failed
                                else "tool_failed"
                                if step.result.status.value != "SUCCEEDED"
                                or artifact.qc_status.value != "PASS"
                                else None
                            )
                            if reason is None and decoder in {
                                "scalar-score-output/1.0.0",
                                "disc-geometry/1.0.0",
                                "fairclip-output/1.0.0",
                            }:
                                original = json.loads(raw)
                                if original.get("input_checksum") not in {
                                    artifacts[k].checksum for k in artifact.derived_from
                                }:
                                    raise ValueError("score_output_input_binding")
                            value = None if reason else self._decode(raw, decoder, field, d)
                            entries.append(
                                ScoreEntry(
                                    definition=d,
                                    target=plan.document.scope.case_code,
                                    status="unavailable" if reason else "available",
                                    value=value,
                                    missing_reason=reason,
                                    source=source.model_copy(
                                        update={"output_schema_version": decoder}
                                    ),
                                    calibration="uncalibrated"
                                    if d.metric == "logit"
                                    else "not_applicable",
                                )
                            )
            # Missing outputs remain explicit; successful independent branches survive.
            for asset_type, d, _, _ in o.outputs:
                d = validate(ScoreDefinition, d)
                if workflow and any(
                    (a.asset_code, d.code) in seen
                    for s in workflow.steps
                    for a in s.result.artifacts
                    if a.artifact_type_code == asset_type
                ):
                    continue
                reason = (
                    "required_dependency_failed"
                    if dependency_failed
                    else "not_executed"
                    if state == "NOT_EXECUTED"
                    else "unknown_job"
                    if state == "UNKNOWN_JOB"
                    else "tool_failed"
                )
                entries.append(
                    ScoreEntry(
                        definition=d,
                        target=scope.case_code,
                        status="unavailable",
                        value=None,
                        missing_reason=reason,
                        source=ScoreSource(
                            **base,
                            qualification="technical",
                            correlation_group=digest(
                                dict(key=o.key, plan=plan.document.checksum, output=asset_type)
                            ),
                        ),
                        calibration="uncalibrated" if d.metric == "logit" else "not_applicable",
                    )
                )
        graph_ref, components, validations, report_ref = None, None, [], None
        if self.graph_reader is not None:
            current, components, validations, report_ref, synthetic = self.graph_reader.read(run_id)
            if current != graph or synthetic != self.synthetic:
                raise ValueError("score_graph_changed")
            graph_ref = validate(GraphSnapshot, asdict(snapshot_binding_for(graph)))
            verified_steps = [
                step for _, _, w, _ in observed.values() if w is not None for step in w.steps
            ]
            accepted_count = len(validations)
            validations = [
                v
                for v in validations
                if any(
                    step.result.status.value == "SUCCEEDED"
                    and step.result.qc_status.value == "PASS"
                    and digest(step) in v["item"].provenance.input_checksums
                    and all(
                        any(
                            a.asset_code == asset
                            and a.checksum in v["item"].provenance.input_checksums
                            and a.laterality == scope.laterality
                            for a in step.result.artifacts
                        )
                        for asset in v["item"].evidence_bindings
                    )
                    for step in verified_steps
                )
            ]
            if not validations or len(validations) != accepted_count:
                # Never attribute whole-graph owner counts to only a verified subset.
                components = None
        for metric in ("support", "refutation", "completeness", "conflict"):
            available = components is not None and bool(validations)
            source = ScoreSource(
                **base,
                qualification="accepted_graph" if available else "technical",
                correlation_group=digest(dict(graph=graph.graph_code))
                if graph is not None
                else digest(dict(context=context.checksum)),
                graph=graph_ref,
                accepted_event_ids=tuple(v["reference"]["audit_event_id"] for v in validations),
                validation_checksums=tuple(
                    v["reference"]["validation_checksum"] for v in validations
                ),
            )
            entries.append(
                ScoreEntry(
                    definition=definition(metric),
                    target=graph.goal_node_code if graph is not None else scope.case_code,
                    status="available" if available else "unavailable",
                    value=validate(Component, components[metric]) if available else None,
                    missing_reason=None if available else "graph_not_qualified",
                    source=source,
                )
            )
        technical_source = ScoreSource(
            **base, qualification="technical", correlation_group=digest(dict(run=run_id))
        )
        for metric, layer, unit, bounds, reason in (
            ("probability", "disease", "1", (0.0, 1.0), "not_calibrated"),
            ("calibration_error", "uncertainty", "1", (0.0, None), "not_calibrated"),
            ("segmentation_confidence", "model", "1", (0.0, 1.0), "no_probability_output"),
        ):
            if any(e.definition.metric == metric for e in entries):
                continue
            d = ScoreDefinition(
                code="missing." + metric,
                layer=layer,
                metric=metric,
                definition="No registered qualified source for this quantity",
                unit=unit,
                direction="none",
                minimum=bounds[0],
                maximum=bounds[1],
            )
            entries.append(
                ScoreEntry(
                    definition=d,
                    target=graph.goal_node_code if graph is not None else scope.case_code,
                    status="unavailable",
                    value=None,
                    missing_reason=reason,
                    source=technical_source,
                    calibration="unavailable",
                )
            )
        # Denominator = distinct registered workflow requests, not evidence slots.
        numerator, denominator = sum(healthy(k) for k in observed), len(observed)
        d = ScoreDefinition(
            code="technical.workflow_coverage",
            layer="technical",
            metric="coverage",
            definition=(
                "Confirmed successful workflows with valid required dependencies / "
                "registered distinct workflow requests"
            ),
            unit="1",
            direction="higher",
            minimum=0.0,
            maximum=1.0,
        )
        entries.append(
            ScoreEntry(
                definition=d,
                target=scope.case_code,
                status="available" if denominator else "not_applicable",
                value=numerator / denominator if denominator else None,
                missing_reason=None if denominator else "zero_denominator",
                source=technical_source,
                numerator=numerator,
                denominator=denominator,
            )
        )
        receipt = None
        if self.engineering_stop is not None and report_ref is not None:
            from packages.knowledge_review.engineering_stop import engineering_stop_receipt

            saved_report = json.loads(read_object(self.objects, report_ref))
            trigger = saved_report["stop_reason"]
            reason = "REASON:" + trigger.upper()
            receipt = validate(
                EngineeringReceipt,
                engineering_stop_receipt(
                    self.engineering_stop[0],
                    self.engineering_stop[1],
                    trigger=trigger,
                    reason_code=reason,
                    runtime_config_checksum=saved_report["config_checksum"],
                ),
            )
        # Detect assembly changes during source reads before handing any bytes to stage.
        if digest(self._policy()) != self.policy_checksum:
            raise ValueError("score_registry_drift")
        return seal(
            ScoreEnvelope,
            scope=scope,
            synthetic=self.synthetic,
            policy_checksum=self.policy_checksum,
            entries=entries,
            technical=technical,
            graph=graph_ref,
            graph_status="accepted" if components is not None and validations else "unavailable",
            original_report=report_ref,
            accepted_validations=tuple(
                validate(
                    ValidationFact,
                    {
                        **v["reference"],
                        "target_node_code": v["item"].target_node_code,
                        "clinical_effect": v["item"].clinical_effect.value,
                    },
                )
                for v in validations
            ),
            engineering_stop=receipt,
            run_status=status,
        )

    def _roots(self, artifact, artifacts, scope, visited=frozenset()):
        if artifact.asset_code in visited:
            raise ValueError("score_derivation_cycle")
        if (
            artifact.scope.case_code != scope.case_code
            or artifact.scope.run_id not in (None, scope.run_id)
            or artifact.laterality != scope.laterality
        ):
            raise ValueError("score_derivation_scope")
        if not artifact.derived_from:
            return ((artifact.asset_code, artifact.checksum),)
        result = set()
        for code in artifact.derived_from:
            if code not in artifacts:
                raise ValueError("score_derivation_source_missing")
            result.update(
                self._roots(artifacts[code], artifacts, scope, visited | {artifact.asset_code})
            )
        return tuple(sorted(result))

    def _decode(self, raw, decoder, field, definition):
        if decoder == "fairclip-output/1.0.0":
            from packages.fairclip_research.models import read_output

            value = read_output(raw)
            if field not in {"raw_cosine_0", "raw_cosine_1"} or definition.metric != "logit":
                raise ValueError("fairclip_score_field")
            return value.raw_cosines[int(field[-1])]
        if decoder in {"png_foreground_pixels/1.0.0", "synthetic_png_logit/1.0.0"}:
            _, pixels = grayscale_png(raw, 16777216)
            if decoder == "synthetic_png_logit/1.0.0":
                if not self.synthetic:
                    raise ValueError("score_synthetic_only_decoder")
                return -float(pixels[0])  # Explicit synthetic raw-score algorithm, no classifier.
            if definition.unit != "px2":
                raise ValueError("score_measurement_unit")
            return float(sum(v != 0 for v in pixels))
        if decoder == "scalar-score-output/1.0.0":
            from packages.tool_execution.scalar_output import ScalarOutput

            output = ScalarOutput.model_validate_json(raw)
            if (output.metric, output.unit) != (definition.metric, definition.unit):
                raise ValueError("score_output_semantics")
            return output.value
        from packages.tool_execution.scalar_output import GeometryOutput

        output = GeometryOutput.model_validate_json(raw)
        if field not in {"area_px", "width_px", "height_px"}:
            raise ValueError("score_geometry_field")
        if definition.unit != ("px2" if field == "area_px" else "px"):
            raise ValueError("score_measurement_unit")
        return float(getattr(output, field))

    def save(self, run_id, key):
        return self.runtime.save_score(
            run_id, key, self.policy_checksum, lambda: self.build(run_id), self.objects
        )

    def read(self, run_id, key):
        return self.runtime.read_score(
            run_id, key, self.policy_checksum, lambda: self.build(run_id), self.objects
        )
