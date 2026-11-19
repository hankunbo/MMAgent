"""07 translation of known read-only 04 refusals; never a persistence error boundary."""

from db.knowledge.repository import KnowledgeConflictError, KnowledgeNotFoundError

# 03/04 currently expose these binding guards as plain ValueError, not a typed refusal.
# Exact messages only: model validation, rule implementation, permissions, retrieval
# infrastructure/integrity and unknown repository errors must remain distinguishable.
_BINDING_REFUSALS = frozenset(
    {
        "exact DiseasePack version/checksum is not pinned by this run",
        "pack release differs from immutable Runtime lock",
        "release synthetic scope differs from pack",
        "all pack bindings must be exact members of the locked release",
        "reviewed template does not bind this exact pack definition",
        "pack goal differs from reviewed template",
        "pack rules differ from reviewed template rules",
        "synthetic Knowledge cannot authorize a real disease pack",
        "pack evidence slots absent from reviewed template",
        "EvidenceRequirement belongs to another template",
        "EvidenceRequirement must share the filtered template scope",
        "EvidenceRequirement synthetic scope mismatch",
        "EvidenceRequirement lacks the bound reviewed acceptance rule",
        "EvidenceRequirement artifact differs from declaration",
        "incomplete EvidenceRequirement coverage",
        "graph run scope differs from locked run",
        "rule package differs from the reviewed exact pack/template",
        "graph does not originate from the exact reviewed template",
        "rule package references do not match current run locked release",
        "demand_requirement_version_conflict",
        "demand_rule_outside_release",
    }
)


class KnowledgeRejection(Exception):
    """Internal control flow, carrying only sanitized refusal evidence."""

    def __init__(self, phase, reason, *, error_type=None, binding_guard=None, gap=None):
        super().__init__(reason)
        self.phase, self.reason = phase, reason
        self.error_type, self.binding_guard, self.gap = error_type, binding_guard, gap


def knowledge_read(phase, operation, *args, **kwargs):
    """Call only read/admission/build ports, never stage, commit or Runtime mutation."""
    try:
        return operation(*args, **kwargs)
    except KnowledgeNotFoundError as error:
        raise KnowledgeRejection(
            phase, "knowledge_unavailable", error_type=type(error).__name__
        ) from error
    except KnowledgeConflictError as error:
        raise KnowledgeRejection(
            phase, "knowledge_version_binding_rejected", error_type=type(error).__name__
        ) from error
    except ValueError as error:
        if type(error) is not ValueError or str(error) not in _BINDING_REFUSALS:
            raise
        raise KnowledgeRejection(
            phase,
            "knowledge_version_binding_rejected",
            error_type="ValueError",
            binding_guard=str(error),
        ) from error


def require_admitted(admission, phase):
    if admission.knowledge_gap is not None:
        raise KnowledgeRejection(phase, "KNO04C_NOT_ADMITTED", gap=admission.knowledge_gap)
    return admission
