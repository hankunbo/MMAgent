"""Read original confirmed FairCLIP facts, never invoke a model from a reader."""

import json

from apps.vectorengine_case_materials import ref
from db.runtime.repository.research_lifecycle import ResearchLifecycleRepository
from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV3
from packages.fairclip_research.core import CONFIG
from packages.fairclip_research.models import read_output
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.models import Material, Source
from packages.object_store import checksum


def read_model(base, target, *, revision=None, configuration=None):
    import re

    if revision is not None and not re.fullmatch(r"[a-z][a-z0-9-]{0,23}", revision):
        raise ValueError("fairclip_execution_revision")
    directory = base.directory / ("fairclip-" + revision if revision else "fairclip-v1")
    if not (directory / "result.json").exists():
        return None
    request = json.loads((directory / "request.json").read_bytes())
    repo = ResearchLifecycleRepository(target)
    context = repo.read_research_run(request["lock"]["run_id"])
    state = repo.lifecycle(context.run_id)
    report = json.loads((directory / "result.json").read_bytes())
    if (
        context.request.model_dump(mode="json") != request
        or context.request.scope.case_code != base.binding["case_code"]
        or not state
        or state["report"] != report
        or state["phase"] != "FINISHED"
    ):
        raise ValueError("dynamic_model_original_lifecycle_binding")
    outputs = report.get("model_outputs", [])
    if not outputs:
        return None
    if len(outputs) != 1:
        raise ValueError("dynamic_model_output_count")
    item = outputs[0]
    source = item["source"]
    asset = base.runtime.read_case_artifact(base.binding["case_code"], source["artifact_code"])
    from pathlib import Path

    from packages.research_runtime.deployment import (
        existing_store,
        load_configuration,
        read_installed_bundle,
    )

    config = load_configuration(Path(configuration or "configs/research/fairclip_single_case.json"))
    if configuration is not None:
        bound = json.loads((directory / "binding.json").read_bytes())
        if (
            bound.get("schema_version") != "new-case-fairclip-binding/1.0.0"
            or bound.get("case_code") != base.binding["case_code"]
            or bound.get("sample_id") != base.binding["sample_id"]
            or bound.get("request_checksum") != digest(request)
            or bound.get("image_checksum") != base.image_artifact.checksum
            or bound.get("result_checksum") != digest(report)
            or bound.get("configuration_checksum") != config["checksum"]
            or bound.get("original_executor") is not True
        ):
            raise ValueError("dynamic_model_selected_configuration_binding")
    installed = read_installed_bundle(config["installed_bundle"])
    # Original H stores model results separately from the visible patient image.
    output_store = existing_store(installed["output_root"], "research_output")
    raw = output_store.read_bounded(asset.uri, 65536)
    output = read_output(raw)
    if (
        checksum(raw) != source["artifact_checksum"]
        or asset.checksum != source["artifact_checksum"]
        or output.model_dump(mode="json") != item["output"]
        or output.input_checksum != base.image_artifact.checksum
        or source["qualification"] != "research_output"
        or source["synthetic"]
    ):
        raise ValueError("dynamic_model_case_output_binding")
    return dict(output=item["output"], source=source, run_id=context.run_id)


def attach_model(state, model):
    if model is None:
        return state
    from .dynamic_stages import rebuild

    output, provenance = model["output"], model["source"]
    resource = ref("resource", provenance["artifact_code"], provenance["artifact_checksum"])
    source_body = dict(configuration=CONFIG, implementation=provenance["implementation"])
    source = Source(
        reference=ref("source", "SOURCE:REGISTERED_FAIRCLIP_METHOD", digest(source_body)),
        locator="Registered FairCLIP method and confirmed 06/H output; not a clinical guideline",
        purpose="research_material",
        state="research_only",
        text=(
            "Registered FairCLIP SLO image/text two-class research classifier. "
            "Classes are non-glaucoma, glaucoma. Original tensor weights, preprocessing and "
            "unit-temperature softmax are fixed. This output is a model-associated signal, "
            "not a calibrated disease probability or an independent clinician assessment. "
            "Current case sensitivity/specificity are unknown. Same SLO as the visual findings; "
            "do not regard them as independent. Method configuration: "
            + canonical(source_body).decode()
        ),
    )
    p0, p1 = output["normalized_model_scores"]
    direction = (p1 - p0) / (p1 + p0)
    body = dict(
        normalized_model_scores=output["normalized_model_scores"],
        raw_cosines=output.get("raw_cosines"),
        model_direction=direction,
        judgment_method="fixed_model_margin",
        mapping="(glaucoma_score - non_glaucoma_score)/(sum of the two scores)",
        source_checksum=digest(provenance),
        input_checksum=output.get("input_checksum"),
        original_run=model["run_id"],
    )
    observation = dict(
        status="obtained",
        method="registered_fairclip",
        unit="normalized_margin",
        value=canonical(body).decode(),
        image=None,
        reference=ref("observation", "OBS:DYNAMIC:FAIRCLIP", digest(body)),
    )
    definition = dict(
        examination="FairCLIP青光眼模型输出",
        applicability_conditions="FairVLMed SLO, original preprocessing/weights; research only",
        interpretation="Copy model_direction as e; method=fixed_model_margin. "
        "Positive supports, negative refutes; neither is diagnosis. "
        "Rate O/L/A anew: uncalibrated output, correlated SLO input.",
        interpretation_status="installed",
        source_refs=[source.reference],
    )
    definition["reference"] = ref(
        "definition", "POLICY:DYNAMIC:FAIRCLIP_MARGIN", digest(definition)
    )
    identifier = "model:FAIRCLIP"
    raw = state.policy.model_dump(mode="json")
    raw["requirements"].append(
        dict(
            id=identifier,
            definition=definition,
            input_modalities=["SLO"],
            basis_kind="model_validation",
            missing_reason=None,
        )
    )
    raw["groups"].append(
        dict(
            id="requirement:" + identifier,
            slots=[[identifier]],
            importance=dict(slo_image_research=1, comprehensive_research=1),
        )
    )
    raw["items"].append(
        dict(
            id=identifier,
            node=identifier,
            kind="model",
            role="model",
            observation_code=observation["reference"].code,
            method=observation["method"],
            unit=observation["unit"],
            laterality="UNKNOWN",
            source_codes=[source.reference.code],
            applicable=True,
            definition=definition,
            model_direction=direction,
            model_observation_ref=observation["reference"],
            model_mapping_version="normalized-class-margin/1.0.0",
            correlation_group="optic_nerve_slo",
            dependence_block="same_slo_image",
            dependency_paths=[[state.resources[0].code, resource.code]],
        )
    )
    material = state.material.model_dump(mode="json")
    material["sources"].append(source.model_dump(mode="json"))
    material["scope"].pop("material_checksum")
    material["scope"]["material_checksum"] = digest(material)
    seed = validate(
        DomainState, state.model_copy(update={"material": validate(Material, material)})
    )
    return rebuild(
        seed,
        validate(TaskPolicyV3, raw),
        [*state.material.observations, observation],
        resources=(*state.resources, resource),
    )
