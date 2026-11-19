"""One fresh shared source snapshot per read, not a cache across requests."""

from packages.evidence_details.acceptance_models import DomainState
from packages.grounding.models import digest, validate


def bind_run(seed, run_id):
    raw = seed.model_dump(mode="json")
    material = raw["material"]
    material["scope"]["run_id"] = run_id
    material["scope"].pop("material_checksum")
    material["scope"]["material_checksum"] = digest(material)
    return validate(DomainState, raw)


class FreshAcquisitionProjection:
    """Only GenericEvidence, whose seed differs between phases by run scope.

    Each accepted_observations call creates this object. The current source owner,
    input and tools are read anew; no value survives to the next before/after HTTP
    boundary. Original per-target acquisition projection and receipt checks remain.
    """

    def __init__(self, base, config):
        self.base, self.config = base, config
        self.seed = None

    def __call__(self, phase, run_id):
        if phase.base is not self.base or phase.target is None:
            raise ValueError("generic_acquisition_projection_scope")
        if self.seed is None:
            self.seed, _ = self.base.project(run_id=run_id, config=self.config)
        phase.seed = bind_run(self.seed, run_id)
        phase.initial = phase.project_current()
        return phase.initial, {}
