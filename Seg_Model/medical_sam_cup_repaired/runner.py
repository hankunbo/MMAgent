"""Versioned optic-cup component with complete SAM/adapter loading.

Color-fundus research only. GPU serialization and execution lifecycle are owned
by the calling 06 backend. No labels, manual points or medical decisions.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "medical_sam_adapter_medagent_pro"
sys.path.insert(0, str(HERE))
from checkpoint import complete_state  # noqa: E402

VERSION = "cup-complete-checkpoint/1.0.0"
WEIGHTS = {
    "sam_vit_b_01ec64.pth": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    "OpticCup_Fundus_SAM_1024.pth": (
        "adfc7638da6b8336f9b5637c17df33bd7fd8cd88be32e00e81afa05ca69ae231"
    ),
}


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load_model():
    import torch

    for filename, checksum in WEIGHTS.items():
        if digest(LEGACY / "weights" / filename) != checksum:
            raise ValueError("cup_weight_checksum_mismatch")
    spec = importlib.util.spec_from_file_location("cup_legacy_loader", LEGACY / "runner.py")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    backend = legacy._load_backend()
    backend._load_model.cache_clear()
    original_load = torch.load

    def tensor_load(*args, **kwargs):
        kwargs["weights_only"] = True
        return original_load(*args, **kwargs)

    # Legacy construction loads base/adapter non-strictly. Never infer from
    # that intermediate model: compose and strict-load every tensor below.
    torch.load = tensor_load
    try:
        model, device = backend._load_model(str(legacy.CUP_ADAPTER))
        base = tensor_load(legacy.SAM_CHECKPOINT, map_location="cpu")
        adapter = tensor_load(legacy.CUP_ADAPTER, map_location="cpu")
    finally:
        torch.load = original_load
    adapter = adapter.get("state_dict", adapter)
    state, origins = complete_state(model.state_dict(), base, adapter)
    for tensor in state.values():
        if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
            raise ValueError("cup_invalid_checkpoint_tensor")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device, backend._automatic_prompt, origins


def predict(model, device, prompt_function, input_path, output_path):
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as functional

    if input_path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("cup_input_byte_limit")
    with Image.open(input_path) as image:
        width, height = image.size
        if width * height > 32_000_000 or min(width, height) < 32:
            raise ValueError("cup_input_dimensions")
        rgb = np.array(image.convert("RGB").resize((1024, 1024), Image.Resampling.BILINEAR))
    point = prompt_function(rgb)
    # Preserve the deployed wrapper's 0..1 RGB preprocessing. SAM-normalized
    # inputs are a separately evaluated change, not silently coupled to this fix.
    image_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255.0
    image_tensor = image_tensor.to(device)
    coords = torch.as_tensor(point, device=device, dtype=torch.float32)[None]
    labels = torch.ones((1, 1), device=device, dtype=torch.int64)
    with torch.inference_mode():
        embedding = model.image_encoder(image_tensor)
        sparse, dense = model.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        logits, _ = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logits = functional.interpolate(
            logits, (height, width), mode="bilinear", align_corners=False
        )
        if not torch.isfinite(logits).all():
            raise ValueError("cup_nonfinite_output")
        mask = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8)
    pixels = int(mask.sum())
    if pixels == 0 or pixels == width * height:
        raise ValueError("cup_degenerate_output")
    with output_path.open("xb") as stream:
        Image.fromarray(mask * 255).save(stream, format="PNG")
    return {
        "version": VERSION,
        "shape": [height, width],
        "foreground_pixels": pixels,
        "prompt_xy_1024": point.tolist(),
        "mask_sha256": digest(output_path),
        "input_sha256": digest(input_path),
        "research_only": True,
        "clinical_validated": False,
        "slo_validated": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, device, prompt, origins = load_model()
    result = predict(model, device, prompt, args.input, args.output)
    result["loaded_tensor_count"] = len(origins)
    result["renamed_base_count"] = sum(key != value["key"] for key, value in origins.items())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
