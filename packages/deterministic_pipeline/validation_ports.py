"""Owner-installed validation port; no clinical predicates or authorization defaults."""

from typing import Protocol

from packages.contracts import VersionedReference

from .models import PipelineConfig, PipelineError


class EvidenceValidator(Protocol):
    """Existing pipeline protocol, including exact published-rule reads and 03 binding.

    Implementations own deterministic interpretation, not publication or tool admission.
    Their reference must bind their code/configuration and be locked in every consuming
    PipelineConfig. A checksum is an identity, not a medical approval or a code sandbox.
    """

    reference: VersionedReference

    def rule(self, admission, requirement_ref, context, query, config): ...

    def eligible(self, input_value, admission, rule, step, plan): ...

    def evaluate(self, graph, node_code, artifact, rule, step): ...


def assemble_evidence_validator(factory, knowledge, objects, published, configs):
    """Validate trusted installation before Runtime assembly; never infer an interpreter.

    The factory receives the same read ports as the legacy validator. This is an
    assembly extension only: no interpreter is installed by this module, and the
    legacy real-input rejection remains unchanged. Unknown/missing references do
    not fall back to the legacy validator, current/latest rules, or synthetic code.
    """
    try:
        if not callable(factory):
            raise ValueError
        validator = factory(knowledge, objects, published)
        if any(
            not callable(getattr(validator, name, None))
            for name in ("rule", "eligible", "evaluate")
        ):
            raise ValueError
        ref = validator.reference
        if type(ref) is not VersionedReference:
            raise ValueError
        ref = VersionedReference.model_validate_json(ref.model_dump_json())
        if ref.checksum is None:
            raise ValueError
        parsed = tuple(PipelineConfig.model_validate_json(c.model_dump_json()) for c in configs)
    except Exception:
        raise PipelineError("invalid_evidence_validator_installation") from None
    if not parsed or any(ref not in c.service_refs for c in parsed):
        raise PipelineError("validator_version_not_locked")
    return validator, parsed
