"""Versioned, patient-free evidence bindings, distinct from historical catalog bytes."""

import copy
import json
from pathlib import Path

from apps.vectorengine_round_materials import RoundCaseEvidence, load_catalog
from packages.grounding.models import digest
from packages.object_store import checksum

PROFILE = "source-complete-view/1.0.0"
NEI = Path("docs/acceptance/13-unified-evidence-v1/nei-risk-blocks.json")
CONFIG = Path("configs/research/unified_evidence_profile_v1.json")


def catalog(profile_path=CONFIG):
    value = copy.deepcopy(load_catalog())
    base_checksum = digest(value)
    nei = json.loads(NEI.read_bytes())
    if nei["id"] != "NEI-GLAUCOMA-RISK" or checksum(nei["text"].encode()) != nei["text_checksum"]:
        raise ValueError("unified_nei_binding")
    value["sources"].append(nei)
    value.update(version=PROFILE, legacy_catalog_checksum=base_checksum)
    profile = json.loads(Path(profile_path).read_bytes())
    if (
        set(profile) != {"schema_version", "profile", "bindings", "merges"}
        or profile["schema_version"] != "evidence-profile/1.0.0"
        or profile["profile"] != PROFILE
    ):
        raise ValueError("unified_profile_version")
    source_ids = {s["id"] for s in value["sources"]}
    for identifier, binding in profile["bindings"].items():
        sources, modalities = binding["selected_source_ids"], binding["input_modalities"]
        if not sources or not set(sources) <= source_ids or not modalities:
            raise ValueError("unified_profile_source_or_modality")
        task = value["tasks"].setdefault(identifier, dict(owner="versioned_evidence_profile"))
        task.update(
            selected_source_ids=sources,
            required_source_ids=sources,
            missing_exact_source_ids=[],
            interpretation_installed=True,
            input_modalities=modalities,
            definition=binding["definition"],
            binding_version=PROFILE,
        )
    value["merges"] = {
        k: dict(target=v, reason="same_clinical_question_keep_provenance_no_extra_vote")
        for k, v in profile["merges"].items()
    }
    return value


class UnifiedSourceReader(RoundCaseEvidence):
    def __init__(self, original, value):
        super().__init__(original, value)
        # No task uses a project-only definition as medical support. Reserve the
        # existing 16-source limit for exact medical excerpts, packet and model.
        self.chunks = [c for c in self.chunks if not c.get("project_definition")]
        self.sources = self.read_sources() + (self.fixed_source,)


def add_context(evidence, state):
    """Age uses its own input artifact, never segmentation Q or image visibility."""
    from apps.vectorengine_case_materials import ref
    from packages.dataset_adapters.public_context import read
    from packages.evaluation_preparation.dynamic_stages import item_for, rebuild
    from packages.grounding.models import canonical, validate

    context = read(
        evidence.runtime, evidence.objects, evidence.visible_value, evidence.context_artifact
    )
    if context != evidence.input_context:
        raise ValueError("unified_context_changed")
    if context["age"]["value"] is None:
        return state
    requirement = next(r for r in state.policy.requirements if r.id == "criterion:AGE")
    resource = ref(
        "resource", evidence.context_artifact.asset_code, evidence.context_artifact.checksum
    )
    body = dict(
        status="obtained",
        method="dataset_public_age",
        unit="year",
        image=None,
        value=canonical(
            dict(
                age=context["age"]["value"],
                projection_policy=context["projection_policy"],
                source_checksum=evidence.context_artifact.checksum,
            )
        ).decode(),
    )
    body["reference"] = ref("observation", "OBS:PUBLIC_CONTEXT:AGE", digest(body)).model_dump(
        mode="json"
    )
    item = item_for(requirement, body, [resource])
    item.update(
        kind="criterion",
        role="risk",
        correlation_group="demographic_age",
        dependence_block="provided_demographics",
        quality=dict(
            value=1.0,
            method="provided_input_no_mask_penalty",
            reason="Source age, not an image estimate.",
            source=resource,
        ),
    )
    policy = state.policy.model_dump(mode="json")
    policy["items"].append(item)
    for row in policy["requirements"]:
        if row["id"] == requirement.id:
            row["missing_reason"] = None
            row["basis_kind"] = "quantitative"
    return rebuild(
        state,
        validate(type(state.policy), policy),
        [*state.material.observations, body],
        resources=[*state.resources, resource],
    )
