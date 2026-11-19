"""03 structural gate for explicit research proposals; never an AcceptedDeltaEvent."""

from packages.grounding.models import validate
from packages.llm_provider.models import SourceProposal


def validate_research_relation(candidate, *, scope, node_codes):
    candidate = validate(SourceProposal, candidate)
    validate_research_target(candidate.target, scope=scope, node_codes=node_codes)
    # Enum excludes formal SUPPORTS/REFUTES and any executable GraphDelta.
    return candidate


def validate_research_target(target, *, scope, node_codes):
    if scope.graph.kind != "research_artifact" or target not in node_codes:
        raise ValueError("research_relation_scope")
