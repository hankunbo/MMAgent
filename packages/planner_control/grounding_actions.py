"""Formal 03 + official 05 adapter. Reads supplied by the trusted application assembly.

read_graph returns the original graph, RulePackage and PropagationContext; this adapter
calls 03 itself. read_request returns the current scope-authorized QueryRequest. Neither
port accepts an agent's legal flags or free-text tool/target selection.
"""

from packages.contracts import PlanAction
from packages.graph_engine import canonical_checksum
from packages.grounding.models import CatalogSnapshot, digest, validate
from packages.grounding.query_models import QueryRequest, QueryResult
from packages.rule_engine import propagate

from .actions import action, build_set
from .models import ActionRejection, Candidate, Reference


def candidate_for(feasible):
    return Candidate(
        id="grounding:" + feasible.identity[7:],
        branch=feasible.candidate.code,
        operation="tool",
        and_ready=True,
        input_refs=(
            Reference(
                role="grounding_candidate",
                code=feasible.candidate.code,
                version="2.0.0",
                checksum=feasible.identity,
            ),
        ),
    )


class GroundingActions:
    def __init__(self, *, grounding, catalogs, read_graph, read_request):
        self.grounding, self.read_graph, self.read_request = grounding, read_graph, read_request
        self.catalogs = catalogs

    def read(self, state, policy):
        if state.graph.kind != "formal_geceg":
            raise ValueError("formal_grounding_cannot_authorize_research_artifact")
        graph, package, context = self.read_graph(state.binding)
        request = validate(QueryRequest, self.read_request(state.binding))
        lock = request.lock
        locks = {r.role: r for r in state.binding.locks}
        if (
            (
                str(graph.case_code),
                str(graph.scope.run_id),
                str(graph.graph_code),
                str(graph.graph_version),
                canonical_checksum(graph),
            )
            != (
                state.binding.case_code,
                state.binding.run_id,
                state.graph.code,
                state.graph.version,
                state.graph.checksum,
            )
            or (lock.run_id, lock.graph_code, str(lock.graph_version), lock.graph_checksum)
            != (state.binding.run_id, state.graph.code, state.graph.version, state.graph.checksum)
            or request.scope.case_code != state.binding.case_code
        ):
            raise ValueError("grounding_graph_scope_mismatch")
        # The visible/query projection and precise lock are registered by the application.
        # This includes patient/encounter/laterality, permitted inputs and all policy versions.
        expected = {
            "grounding_request": digest(request),
            "grounding_lock": lock.checksum,
            "catalog": lock.catalog_ref.checksum,
            "rule": package.package_checksum,
        }
        if any(role not in locks or locks[role].checksum != sha for role, sha in expected.items()):
            raise ValueError("grounding_input_or_policy_lock_mismatch")
        result = validate(QueryResult, self.grounding.query(request))
        result = validate(QueryResult, self.grounding.validate_result(request, result))
        if result.request != request:
            raise ValueError("grounding_request_mismatch")
        catalog = validate(CatalogSnapshot, self.catalogs.read_snapshot(lock.catalog_ref))
        if catalog.reference() != lock.catalog_ref:
            raise ValueError("catalog_lock_mismatch")
        tools = {t.implementation_ref: t for t in catalog.tools}
        propagated = propagate(graph, package, context)
        legal = {
            a.target_node_code
            for a in propagated.legal_actions
            if a.action
            in {PlanAction.SELECT_WORKFLOW, PlanAction.SELECT_EVIDENCE_TASK, PlanAction.SWITCH_PATH}
            and a.graph_version == graph.graph_version
            and a.rule_package_checksum == locks["rule"].checksum
        }
        target = Reference(
            role="node",
            code=request.target_node_code,
            version=state.graph.version,
            checksum=state.graph.checksum,
        )
        choices, rejected = [], []
        if request.target_node_code not in legal:
            rejected.append(
                ActionRejection(
                    target=target.code,
                    category="graph_not_legal",
                    reason="03_did_not_authorize_target",
                )
            )
        else:
            for c in result.candidates:
                required = {
                    code for d in c.dependencies if d.required for code in d.artifact_options
                }
                inputs = tuple(
                    Reference(
                        role="artifact",
                        code=a.asset_code,
                        version=a.artifact_ref.version,
                        checksum=a.checksum,
                    )
                    for a in sorted(request.artifacts, key=lambda a: a.asset_code)
                    if a.asset_code in required
                )
                choices.append(
                    action(
                        kind="select_candidate",
                        target=target,
                        branch=c.candidate.code,
                        candidate=candidate_for(c),
                        candidate_ref=c.identity,
                        required_inputs=inputs,
                        dependencies=c.dependencies,
                        expected_artifact=c.candidate.output.artifact_ref,
                        entrypoint="ToolAdmissionService.admit",
                        preconditions=(
                            "03_legal_target",
                            "05_complete_and",
                            "06_readmission_required",
                        ),
                        reasons=("official_query_feasible",),
                        components=c.components,
                        review_required=result.expert_review_recommended,
                        validation_level=tools[
                            c.candidate.implementation_ref
                        ].validation_status.value,
                        expires_at=state.binding.deadline,
                    )
                )
            # Existing formal artifacts require a dedicated observation/validation owner.
            # Do not route them into the research v5 observer.
            if result.existing_artifacts:
                rejected.append(
                    ActionRejection(
                        target=target.code,
                        category="missing_implementation",
                        reason="formal_observer_not_installed",
                    )
                )
        categories = {
            "patient_data_missing": "missing_data",
            "input_unsatisfied": "and_unsatisfied",
            "unresolved_producer": "unknown_producer",
            "search_incomplete": "search_truncated",
            "implementation_missing": "missing_implementation",
            "safety_restriction": "safety",
            "mapping_missing": "unknown_producer",
            "validation_insufficient": "safety",
        }
        rejected.extend(
            ActionRejection(
                target=target.code,
                category=categories.get(f.category, "safety"),
                reason=f.reason,
                source_checksum=digest(f),
            )
            for f in result.gaps
        )
        return build_set(
            state,
            policy,
            choices=choices,
            safety=result.safety_checksum,
            availability=digest(
                dict(
                    artifacts=request.artifacts,
                    environment=request.environment,
                    source=request.availability_source_ref,
                )
            ),
            sources=(result.checksum, propagated.trace.trace_checksum),
            rejections=rejected,
            enumeration="search_truncated" if result.status == "truncated" else "complete",
        )
