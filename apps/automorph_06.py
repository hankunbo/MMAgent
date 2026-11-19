"""New single-file adapter for a separately reviewed 06 deployment.

This file neither installs/approves a tool nor changes the old worker/lock.
Executing it is model work and must be dispatched under the original runner.
"""

import argparse
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    # Direct script execution otherwise makes apps/platform.py shadow the
    # standard-library platform module imported by our dependency stack.
    script_directory = Path(__file__).resolve().parent
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != script_directory]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.object_store import checksum
from packages.tool_execution.fundus_bundle import (
    COMMIT,
    MAX_BYTES,
    TRANSFORM,
    encode_bundle,
    strict_json,
)


def bounded(path, limit=MAX_BYTES):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit:
        raise ValueError("fundus_adapter_file_size")
    return raw


def pack_prediction(directory, *, input_bytes, expected_source_checksum):
    """Preserve saved model bytes; this conversion grants no execution authority."""
    from packages.tool_benchmark.formats import jpeg_dimensions

    directory = Path(directory)
    result = strict_json(bounded(directory / "result.json", 65536), 65536)
    if set(result) != {
        "schema_version",
        "input_checksum",
        "source_checksum",
        "input_shape",
        "analysis_shape",
        "transform",
        "model",
        "device",
        "status",
        "ensemble_std_mean",
        "quality_is_calibrated_accuracy",
        "masks_modified",
        "missing_is_negative",
        "disease_probability",
        "training_overlap",
        "validation_status",
        "physical_scale",
        "artifacts",
        "runtime",
    } or (
        result["schema_version"] != "automorph-fundus-measurement/1.0.0"
        or result["input_checksum"] != checksum(input_bytes)
        or tuple(result["input_shape"]) != jpeg_dimensions(input_bytes)
        or result["source_checksum"] != expected_source_checksum
        or result["analysis_shape"] != [512, 512]
        or result["transform"] != TRANSFORM
        or result["model"] != "AutoMorph-disc-cup-eight-WNet"
        or result["status"] != "predictions_saved"
        or result["masks_modified"] is not False
        or result["quality_is_calibrated_accuracy"] is not False
        or result["missing_is_negative"] is not False
        or result["disease_probability"] is not None
        or result["physical_scale"] is not None
        or result["validation_status"] != "research_smoke_only"
        or result["training_overlap"]
        != "GAMMA_REFUGE_in_upstream_training_config_not_external_test"
    ):
        raise ValueError("fundus_adapter_prediction_binding")
    names = {"labels.png", "disc.png", "cup.png", "rim.png", "overlay.png"}
    if set(result["artifacts"]) != names:
        raise ValueError("fundus_adapter_artifact_inventory")
    payloads = {name: bounded(directory / name) for name in sorted(names)}
    if any(checksum(raw) != result["artifacts"][name] for name, raw in payloads.items()):
        raise ValueError("fundus_adapter_artifact_binding")
    # The decorative RGB overlay is checked but deliberately not part of the
    # new executable measurement object; a viewer can derive it from saved masks.
    return encode_bundle(
        input_checksum=result["input_checksum"],
        input_shape=result["input_shape"],
        source_checksum=result["source_checksum"],
        masks={name: payloads[name + ".png"] for name in ("labels", "disc", "cup", "rim")},
        device=result["device"],
        ensemble_std_mean=result["ensemble_std_mean"],
    )


def run(*, image, output, resource_root, source_checksum):
    from apps import automorph_resources
    from packages.multimodal_research import fundus_worker

    root = Path(resource_root)
    output = Path(output)
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError("fundus_adapter_resource_root")
    if output.exists() or output.is_symlink():
        raise ValueError("fundus_adapter_output_exists")
    raw = bounded(image)
    lock = bounded(root / "ensemble-source-lock.json", 65536)
    if checksum(lock) != source_checksum or strict_json(lock, 65536).get("commit") != COMMIT:
        raise ValueError("fundus_adapter_source_lock")
    # Old files stay byte-identical. This isolated adapter process fixes both
    # import-time aliases explicitly; it never relies on the runner's cwd.
    previous = automorph_resources.ROOT, fundus_worker.ROOT
    try:
        automorph_resources.ROOT = root
        fundus_worker.ROOT = root
        fundus_worker.verify_resources()
        # Keep per-attempt diagnostics on failure; never reuse or overwrite an
        # existing working directory. Only the bundle is an admitted output.
        temporary = Path(tempfile.mkdtemp(prefix="fundus-06-", dir=output.parent))
        prediction = temporary / "prediction"
        fundus_worker.run(image, prediction)
        if bounded(image) != raw or bounded(root / "ensemble-source-lock.json", 65536) != lock:
            raise ValueError("fundus_adapter_input_or_source_changed")
        fundus_worker.verify_resources()
        bundle = pack_prediction(
            prediction, input_bytes=raw, expected_source_checksum=source_checksum
        )
        with output.open("xb") as stream:
            stream.write(bundle)
    finally:
        automorph_resources.ROOT, fundus_worker.ROOT = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resource-root", required=True)
    parser.add_argument("--source-checksum", required=True)
    args = parser.parse_args()
    run(
        image=args.image,
        output=args.output,
        resource_root=args.resource_root,
        source_checksum=args.source_checksum,
    )


if __name__ == "__main__":
    main()
