"""Read exact original case/mask/failure provenance before a partial-rim successor.

This read port creates no observation and grants no dispatch or clinical authority.
The successor must separately check current Tool, Control, definition and budget.
"""

import argparse
import json
from pathlib import Path

from packages.contracts import Artifact
from packages.grounding.models import digest
from packages.object_store import checksum


def read(path):
    return json.loads(Path(path).read_bytes())


def bound_inputs(key):
    """Reconstruct, rather than trust caller-supplied masks or a saved failure flag."""
    from packages.evaluation_preparation.case_execution import owners
    from packages.evaluation_preparation.component_view import select_component
    from packages.evaluation_preparation.generic_input import directory, load, validate_registered
    from packages.evaluation_preparation.generic_ports import read as read_ports

    spec = load(key)
    root = directory(key)
    binding = read(root / "binding.json")
    if binding["input_spec_checksum"] != digest(spec) or binding["execution_namespace"] != key:
        raise ValueError("partial_rim_original_input_identity")
    runtime, objects, _ = owners()
    registered = validate_registered(spec["sample"], (root, binding, runtime, objects))
    ports = read_ports(*registered)
    if ports is None:
        raise ValueError("partial_rim_original_mask_ports_missing")
    assets, raw, _ = ports
    folder = root / "13-full-evidence-tools-v2"
    artifact = Artifact.model_validate(read(folder / "artifact.json"))
    image = Artifact.model_validate(binding["image"])
    if (
        artifact.scope.case_code != binding["case_code"]
        or artifact.derived_from != (image.asset_code,)
        or artifact.provenance.input_checksums != (image.checksum,)
        or artifact.artifact_type_code != "ARTIFACT:RESEARCH_FULL_TOOL_EVIDENCE"
        or runtime.read_case_artifact(binding["case_code"], artifact.asset_code) != artifact
    ):
        raise ValueError("partial_rim_original_tool_artifact_binding")
    stored = objects.read_bounded(artifact.uri, 4194304)
    tools = read(folder / "result.json")
    if (
        checksum(stored) != artifact.checksum
        or json.loads(stored) != tools
        or tools["case_code"] != binding["case_code"]
        or tools["image_checksum"] != image.checksum
    ):
        raise ValueError("partial_rim_original_tool_content_binding")
    masks = {}
    for name in ("disc", "cup"):
        role = name + "_mask"
        component, policy = select_component(raw[role])
        path = folder / (name + "-component.png")
        if path.read_bytes() != component or tools["component_views"][name] != policy:
            raise ValueError("partial_rim_component_not_original_derivation")
        masks[name] = dict(
            path=str(path.resolve()),
            checksum=checksum(component),
            original_asset=assets[role].model_dump(mode="json"),
            component_view=policy,
        )
    failure = tools["results"].get("rim")
    if not failure or failure.get("observation") is not None:
        raise ValueError("partial_rim_original_failed_method_required")
    plan = read(folder / "rim/plan.json")
    claim = read(folder / "rim/process-claim.json")
    receipt = read(folder / "rim/process-result.json")
    manifest = plan["manifest"]
    replacements = {
        "{input.disc_mask}": masks["disc"]["path"],
        "{input.cup_mask}": masks["cup"]["path"],
        "{output.observation}": str((folder / "rim/observation.json").resolve()),
    }
    argv = [replacements.get(v, v) for v in manifest["invocation"]["argv"]]
    if (
        manifest["tool_code"] != "calculate_slo_rim_directional_width"
        or plan["plan"]["tool_code"] != manifest["tool_code"]
        or plan["plan"]["version"] != manifest["version"]
        or plan["plan"]["implementation"] != manifest["implementation_key"]
        or failure["missing_reason"] != "algorithm_execution_failed"
        or failure["execution"] != receipt
        or failure["manifest"] != manifest
        or receipt["argv"] != claim["argv"]
        or receipt["argv"] != plan["plan"]["argv"]
        or receipt["argv"] != argv
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] in (0, 124)
        or checksum((folder / "rim/process.log").read_bytes()) != failure["diagnostic_checksum"]
        or plan["input_checksums"]["disc_mask"] != masks["disc"]["checksum"]
        or plan["cup_checksum"] != masks["cup"]["checksum"]
        or plan["image_checksum"] != image.checksum
    ):
        raise ValueError("partial_rim_original_failure_or_mask_binding")
    # Re-read the same original owners after material reconstruction. A valid hash
    # alone is not authority to consume a changed input or withdrawn artifact.
    validate_registered(spec["sample"], registered)
    if (
        read_ports(*registered) != ports
        or runtime.read_case_artifact(binding["case_code"], artifact.asset_code) != artifact
    ):
        raise ValueError("partial_rim_inputs_changed_during_read")
    proof = dict(
        schema_version="partial-rim-original-input-proof/1.0.0",
        key=key,
        sample=spec["sample"],
        case_code=binding["case_code"],
        input_spec_checksum=digest(spec),
        input_checksum=binding["visible_checksum"],
        image=image.model_dump(mode="json"),
        original_tool_artifact=artifact.model_dump(mode="json"),
        original_failure=failure,
        original_plan_checksum=digest(plan),
        original_claim_checksum=digest(claim),
        masks=masks,
        dispatch_authorized=False,
        definition_accepted=False,
        new_observation_created=False,
    )
    return registered, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    _, proof = bound_inputs(args.key)
    with args.output.open("x") as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(dict(key=args.key, original_input_verified=True, new_requests=0)))


if __name__ == "__main__":
    main()
