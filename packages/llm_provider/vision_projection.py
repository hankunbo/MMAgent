"""Versioned wire-only minimization; full owner material stays in the journal."""

from packages.evidence_acquisition.models import TargetedVisionPayload
from packages.grounding.models import validate

PROMPT = "vision-partial-multimodal-1.2.0"


def project(payload):
    value = validate(TargetedVisionPayload, payload)
    task, material = value.vision_task, value.material
    observed = [o for o in material.observations if o.reference == task.observation]
    if len(observed) != 1 or observed[0].image is None:
        raise ValueError("vision_projection_observation")
    allowed = {r.code: r for r in task.allowed_sources}
    sources = {s.reference.code: s for s in material.sources}
    if any(code not in sources or sources[code].reference != ref for code, ref in allowed.items()):
        raise ValueError("vision_projection_sources")
    if set(task.definition.source_refs) != set(task.allowed_sources):
        raise ValueError("vision_projection_definition")
    observation = observed[0]
    return dict(
        schema_version="targeted-vision-wire/1.0.0",
        role="vision",
        task_id=task.assessment_id,
        examination=task.definition.examination,
        instruction=task.definition.interpretation,
        applicability=task.definition.applicability_conditions,
        laterality=task.laterality,
        input_metadata=observation.value,
        image=dict(
            width=observation.image.width,
            height=observation.image.height,
            media_type=observation.image.media_type,
            location=observation.image.location,
        ),
        relevant_sources=[dict(code=code, text=sources[code].text) for code in allowed],
    )
