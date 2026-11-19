"""No medical thresholds. Only an explicit synthetic assertion interpreter is installed."""

from packages.contracts import GraphDelta, ValidationResult, VersionedReference
from packages.graph_engine import canonical_checksum
from packages.rule_engine import bind_validation
from packages.tool_execution.qc import grayscale_png

from .models import PipelineError
from .tool_evidence import inspect_tool_evidence


class DeterministicValidator:
    def __init__(self, knowledge, objects, published):
        self.knowledge = knowledge
        self.published = published
        self.objects = objects
        self.reference = VersionedReference(
            code="POLICY:DETERMINISTIC_VALIDATION",
            version="1.1.0",
            checksum=canonical_checksum(
                dict(
                    synthetic="RULE:SYNTHETIC_ASSERTION:all_pixels_one",
                    medical_interpreters=[],
                    tool_evidence_binding="09A.1.0.0",
                )
            ),
        )

    def rule(self, admission, requirement_ref, context, query, config):
        requirement = self.knowledge.get_published_evidence(requirement_ref, context=context).record
        result = self.published.retrieve_published_knowledge(
            admission.pack.bindings.release_ref, query, context, config
        )
        records = {h["artifact"].record.reference(): h["artifact"].record for h in result["hits"]}
        rule = records.get(requirement.acceptance_rule_ref)
        if rule is None:
            raise PipelineError("exact_reviewed_validation_rule_missing")
        if rule.reference() not in admission.pack.bindings.rule_refs:
            raise PipelineError("validation_rule_outside_release")
        return requirement, rule

    def eligible(self, input_value, admission, rule, step, plan):
        check = inspect_tool_evidence(step, plan)
        if not check.evidence_candidate:
            return False, check.reason
        if not input_value.synthetic:
            if not step.result.decisive_evidence:
                return False, "nondecisive_research_observation"
            return False, "reviewed_medical_interpreter_missing"
        if not admission.pack.definition.synthetic or not rule.synthetic:
            raise PipelineError("synthetic_knowledge_scope_mismatch")
        if not all(s.deployment.allowlist.synthetic_only for s in plan.document.steps):
            raise PipelineError("synthetic_tool_scope_mismatch")
        if rule.expression_code != "RULE:SYNTHETIC_ASSERTION":
            return False, "reviewed_interpreter_missing"
        return True, "synthetic_assertion_only"

    def evaluate(self, graph, node_code, artifact, rule, step):
        # The public interpreter must not read an arbitrary object supplied beside a step.
        from packages.grounding.models import validate
        from packages.tool_execution.models import StepResult

        step = validate(StepResult, step)
        if artifact not in step.result.artifacts:
            raise PipelineError("evidence_artifact_not_in_tool_result")
        index = step.result.artifacts.index(artifact)
        if (
            step.result.status.value != "SUCCEEDED"
            or step.result.qc_status.value != "PASS"
            or artifact.qc_status.value != "PASS"
            or step.qc_records[index].status.value != "PASS"
            or step.result.validation_status.value != "auto_validated_research"
        ):
            raise PipelineError("evidence_requires_qualified_tool_observation")
        if not rule.synthetic or rule.expression_code != "RULE:SYNTHETIC_ASSERTION":
            raise PipelineError("reviewed_interpreter_missing")
        if (artifact.scope.case_code, artifact.scope.run_id) != (
            graph.case_code,
            graph.scope.run_id,
        ):
            raise PipelineError("evidence_graph_scope_mismatch")
        data = self.objects.read_bounded(artifact.uri, 4194304)
        from packages.object_store import checksum

        if checksum(data) != artifact.checksum:
            raise PipelineError("validation_object_checksum_mismatch")
        shape, pixels = grayscale_png(data, 16777216)
        if shape != artifact.dimensions or artifact.media_type != "image/png":
            raise PipelineError("evidence_bytes_metadata_mismatch")
        supports = bool(pixels) and all(p == 1 for p in pixels)
        node = next(n for n in graph.nodes if n.node_code == node_code)
        if node.status.value != "PENDING" or artifact.asset_code not in node.evidence_bindings:
            raise PipelineError("validation_requires_current_pending_binding")
        ref = rule.reference().to_versioned_reference()
        binding = canonical_checksum(graph)
        token = canonical_checksum(dict(graph=binding, artifact=artifact.checksum, rule=ref))[
            7:55
        ].upper()
        provenance = graph.provenance.model_dump(mode="json") | dict(
            actor="VERIFICATION_AGENT",
            generator_version="09A.synthetic.1.0.0",
            input_checksums=[binding, artifact.checksum, canonical_checksum(step)],
            rule_versions=[ref.model_dump(mode="json")],
            tool_versions=list(step.result.provenance.model_dump(mode="json")["tool_versions"]),
        )
        common = dict(
            schema_version="1.0",
            provenance=provenance,
            scope=graph.scope.model_dump(mode="json"),
            laterality=graph.laterality.value,
            coordinate_space="NOT_APPLICABLE",
            unit=None,
        )
        delta = GraphDelta.model_validate(
            common
            | dict(
                contract_type="GraphDelta",
                graph_code=graph.graph_code,
                base_version=graph.graph_version,
                proposed_by="VERIFICATION_AGENT",
                proposal_kind="GRAPH_DELTA_CANDIDATE",
                idempotency_key="TASK:OID_" + token,
                reason_codes=[ref.code],
                operations=[
                    dict(
                        op="UPDATE_NODE_STATUS",
                        node_code=node_code,
                        from_status="PENDING",
                        to_status="SUPPORTED" if supports else "REFUTED",
                        source="VALIDATED_EVIDENCE",
                        evidence_binding=artifact.asset_code,
                    )
                ],
            )
        )
        value = ValidationResult.model_validate(
            common
            | dict(
                contract_type="ValidationResult",
                validation_code="VALIDATION:OID_" + token,
                validation_scope="EVIDENCE",
                target_node_code=node_code,
                observation="POSITIVE" if supports else "NEGATIVE",
                capability="AVAILABLE",
                tool_execution_status="SUCCEEDED",
                qc_status="PASS",
                validity="VALID",
                clinical_effect="SUPPORTS" if supports else "REFUTES",
                rule_refs=[ref],
                evidence_bindings=[artifact.asset_code],
                graph_delta_candidate=delta,
                reason_codes=[ref.code],
            )
        )
        return bind_validation(graph, value)
