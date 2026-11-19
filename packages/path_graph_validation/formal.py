"""03/07 read-only adapter. Only explicitly installed synthetic mechanism inputs.

The installed reader owns graph replay, Published/template/purpose and current tool
source admission. This adapter never accepts or rebinds a ValidationResult and never
applies its propagation candidate. Production medical assembly remains uninstalled.
"""

from dataclasses import asdict, dataclass

from pydantic_core import to_jsonable_python

from packages.contracts import GECEGInstance
from packages.deterministic_pipeline.reporting import accepted_validations
from packages.graph_engine import canonical_checksum
from packages.rule_engine import (
    PropagationContext,
    RulePackage,
    propagate,
    snapshot_binding_for,
    validate_rule_package,
)

from .research import seal_summary


def payload(value):
    return to_jsonable_python(asdict(value))


@dataclass(frozen=True)
class FormalInputs:
    graph: GECEGInstance
    package: RulePackage
    context: PropagationContext
    events: tuple
    audit_events: tuple


class FormalPathReader:
    def __init__(self, *, read_current, read_snapshot, validate_current_sources, purpose):
        if purpose != "synthetic_mechanism":
            raise ValueError("live_domain_acceptance_not_enabled")
        self.read_current, self.read_snapshot = read_current, read_snapshot
        self.validate_sources = validate_current_sources

    def __call__(self):
        source = self.read_current()
        graph = GECEGInstance.model_validate(source.graph.model_dump(mode="python"))
        context = source.context
        if context.validations:
            raise ValueError("path_graph_unaccepted_candidates_forbidden")
        validated = validate_rule_package(
            graph,
            source.package,
            expected_version=context.expected_package_version,
            expected_checksum=context.expected_package_checksum,
        )
        accepted = accepted_validations(
            graph, source.events, source.audit_events, self.read_snapshot
        )
        # Mandatory installed source owner: exact current Published/purpose/qualification,
        # confirmed Tool results and reset/revocation. Exceptions remain failures.
        self.validate_sources(source, accepted)
        result = propagate(graph, validated.package, context)
        if result.knowledge_gaps:
            raise ValueError("path_graph_invalid_propagation_context")
        rows = {r.node_code: r for r in result.evaluation.node_results}
        paths = []
        for gate in validated.package.gates:
            row = rows[gate.parent_node_code]
            paths.append(
                dict(
                    gate=payload(gate),
                    result=payload(row),
                    evidence=[
                        v["reference"]
                        for v in accepted
                        if v["item"].target_node_code in gate.child_node_codes
                    ],
                )
            )
        evidence = [
            dict(
                target=v["item"].target_node_code,
                effect=v["item"].clinical_effect.value,
                validity=v["item"].validity.value,
                reference=v["reference"],
            )
            for v in accepted
        ]
        unresolved = [payload(b) for b in result.evaluation.blocked_evidence]
        unresolved += [
            dict(
                target=r.node_code,
                reason="unresolved_branch_or_relation",
                nodes=list(r.unresolved_nodes),
            )
            for r in rows.values()
            if r.unresolved_nodes
        ]
        unresolved += [
            dict(
                target=e["target"],
                reason=e["reason"],
                classification="technical_or_uninterpreted_observation",
                clinical_effect="NONE",
            )
            for e in source.events
            if e["kind"] == "observation_classification" and not e["eligible"]
        ]
        goal = rows.get(graph.goal_node_code)
        # Read the existing rule result, not the stale pre-candidate goal status.
        goals = (
            dict(
                target=graph.goal_node_code,
                current_status=next(
                    n.status.value for n in graph.nodes if n.node_code == graph.goal_node_code
                ),
                derived_status=goal.derived_status.value if goal and goal.derived_status else None,
                conclusion="original_03_candidate_only",
            ),
        )
        grouped = {}
        for v in accepted:
            for code in v["item"].evidence_bindings:
                grouped.setdefault(code, []).append(v["item"].target_node_code)
        self.validate_sources(source, accepted)
        latest = self.read_current()
        if canonical_checksum(latest) != canonical_checksum(source):
            raise ValueError("path_graph_snapshot_changed")
        return seal_summary(
            domain="formal_synthetic",
            purpose="synthetic_mechanism",
            case_code=graph.case_code,
            run_id=graph.scope.run_id,
            graph=payload(snapshot_binding_for(graph)),
            binding=dict(
                template=graph.template_ref.model_dump(mode="json"),
                rule_package=payload(validated.package),
                trace=payload(result.trace),
                candidate_checksum=canonical_checksum(result.candidate_delta)
                if result.candidate_delta
                else None,
            ),
            paths=tuple(paths),
            goals=goals,
            evidence=tuple(evidence),
            unresolved=tuple(unresolved),
            relations=tuple(e.model_dump(mode="json") for e in graph.edges),
            duplicate_groups=tuple(
                dict(
                    asset_code=a,
                    targets=sorted(set(targets)),
                    handling="target_bound_no_independent_score",
                )
                for a, targets in sorted(grouped.items())
                if len(targets) > 1
            ),
            coverage=payload(result.evaluation.completeness),
            risk=dict(no_new_weights=True, clinical_probability=None),
            review_facts=dict(
                conflict=bool(result.evaluation.conflict.contributing_nodes),
                uncertain=bool(unresolved),
                high_risk=False,
                required=bool(result.evaluation.conflict.contributing_nodes),
                no_progress=0,
            ),
            legal_candidates=tuple(payload(a) for a in result.legal_actions),
            stop_candidate=payload(result.stop_suggestion) if result.stop_suggestion else None,
        )
