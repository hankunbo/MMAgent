"""Pinned AutoMorph disc/cup ensemble, model-only bounded research adapter.

No image-quality exclusion, vessel pipeline, clinical thresholds or physical
pixel-size defaults. Original class logits and masks are retained unmodified.
"""

import argparse
import importlib
import io
import json
import sys
from pathlib import Path

from apps.automorph_resources import COMMIT, ROOT
from packages.object_store import checksum


def verify_resources():
    value = json.loads((ROOT / "ensemble-source-lock.json").read_bytes())
    if value["commit"] != COMMIT:
        raise ValueError("automorph_commit_changed")
    from apps.automorph_resources import ENSEMBLE, FILES

    if set(value["files"]) != set(FILES + ENSEMBLE):
        raise ValueError("automorph_resource_inventory")
    for name, expected in value["files"].items():
        if checksum((ROOT / name).read_bytes()) != expected:
            raise ValueError("automorph_resource_changed")
    return value


def run(image, output):
    import numpy as np
    import torch
    from PIL import Image

    resources = verify_resources()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    raw = Path(image).read_bytes()
    if len(raw) > 16777216:
        raise ValueError("fundus_image_size")
    original = Image.open(io.BytesIO(raw))
    if original.mode != "RGB" or original.width * original.height > 16777216:
        raise ValueError("fundus_requires_rgb_image")
    resized = original.resize((512, 512), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float()[None] / 255
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT / "M2_lwnet_disc_cup"))
    get_arch = importlib.import_module("models.get_model").get_arch
    probabilities = []
    with torch.inference_mode():
        for seed in (28, 30, 32, 34, 36, 38, 40, 42):
            model = get_arch("wnet", n_classes=3).to(device)
            path = (
                ROOT
                / "M2_lwnet_disc_cup/experiments/wnet_All_three_1024_disc_cup"
                / str(seed)
                / "model_checkpoint.pth"
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            _, logits = model(tensor.to(device))
            probabilities.append(logits.softmax(1).cpu())
    stack = torch.stack(probabilities)
    mean = stack.mean(0)[0]
    labels = mean.argmax(0).numpy().astype("uint8")
    if not torch.isfinite(stack).all() or stack.shape != (8, 1, 3, 512, 512):
        raise ValueError("fundus_logits_invalid")
    Image.fromarray(labels).save(output / "labels.png")
    for name, mask in (("disc", labels > 0), ("cup", labels == 2), ("rim", labels == 1)):
        Image.fromarray(mask.astype("uint8") * 255).save(output / (name + ".png"))
    overlay = np.asarray(resized).copy()
    colors = np.zeros_like(overlay)
    colors[labels == 1], colors[labels == 2] = (0, 255, 0), (255, 70, 70)
    overlay[labels > 0] = (overlay[labels > 0] * 0.65 + colors[labels > 0] * 0.35).astype("uint8")
    Image.fromarray(overlay).save(output / "overlay.png")
    result = dict(
        schema_version="automorph-fundus-measurement/1.0.0",
        input_checksum=checksum(raw),
        source_checksum=checksum((ROOT / "ensemble-source-lock.json").read_bytes()),
        input_shape=[original.height, original.width],
        analysis_shape=[512, 512],
        transform="RGB_bilinear_512_tensor_0_1_no_normalization",
        model="AutoMorph-disc-cup-eight-WNet",
        device=device,
        status="predictions_saved",
        ensemble_std_mean=float(stack.std(0, unbiased=False).mean()),
        quality_is_calibrated_accuracy=False,
        masks_modified=False,
        missing_is_negative=False,
        disease_probability=None,
        training_overlap="GAMMA_REFUGE_in_upstream_training_config_not_external_test",
        validation_status="research_smoke_only",
        physical_scale=None,
        artifacts={p.name: checksum(p.read_bytes()) for p in sorted(output.glob("*.png"))},
        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda),
    )
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            dict(status=result["status"], device=device, source_files=len(resources["files"]))
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.image, args.output)
