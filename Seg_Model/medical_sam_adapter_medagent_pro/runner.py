#!/usr/bin/env python3
"""Stable CLI wrapper for MedAgent-Pro fundus-image segmentation adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT / "source_medagent_pro"
WEIGHT_ROOT = PACKAGE_ROOT / "weights"
SAM_CHECKPOINT = WEIGHT_ROOT / "sam_vit_b_01ec64.pth"
DISC_ADAPTER = WEIGHT_ROOT / "OpticDisc_Fundus_SAM_1024.pth"
CUP_ADAPTER = WEIGHT_ROOT / "OpticCup_Fundus_SAM_1024.pth"
VESSEL_ADAPTER = WEIGHT_ROOT / "Retina%20Vessel_Fundus_sam_1024.pth"

EXPECTED_SHA256 = {
    SAM_CHECKPOINT.name: "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    DISC_ADAPTER.name: "a1e39b25aa17e71fe18a6776bd771fc862b7d28c7562e574646473effdff1af1",
    CUP_ADAPTER.name: "adfc7638da6b8336f9b5637c17df33bd7fd8cd88be32e00e81afa05ca69ae231",
    VESSEL_ADAPTER.name: "301fd199192dc1dd1e433a2458b1b8e070954c0741d20ce67b18bfd4708fbaad",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_files() -> list[Path]:
    return [SAM_CHECKPOINT, DISC_ADAPTER, CUP_ADAPTER, VESSEL_ADAPTER]


def _validate_files(verify_sha256: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"MedAgent-Pro source link is missing: {SOURCE_ROOT}")
    for path in _required_files():
        if not path.is_file():
            raise FileNotFoundError(f"Required weight is missing: {path}")
        actual_sha256 = _sha256(path) if verify_sha256 else None
        if actual_sha256 and actual_sha256 != EXPECTED_SHA256[path.name]:
            raise RuntimeError(f"SHA256 mismatch for {path}")
        results.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": actual_sha256 or EXPECTED_SHA256[path.name],
                "sha256_verified": verify_sha256,
            }
        )
    return results


def _load_backend():
    source = str(SOURCE_ROOT.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from Glaucoma import tools as backend

    # Keep the stable Seg_Model resource paths independent of the upstream layout.
    backend.SAM_CHECKPOINT = SAM_CHECKPOINT
    backend.DISC_ADAPTER = DISC_ADAPTER
    backend.CUP_ADAPTER = CUP_ADAPTER

    # AdapterBlock reads these fields, but the upstream glaucoma wrapper omits them.
    # Keep this compatibility shim local so the source repository remains untouched.
    upstream_model_args = backend._model_args

    def compatible_model_args():
        args = upstream_model_args()
        args.thd = False
        args.chunk = None
        return args

    backend._model_args = compatible_model_args
    return backend


def _automatic_vessel_prompt(rgb_1024):
    """Select a likely vessel pixel from the dark-line response in a fundus image."""
    import cv2
    import numpy as np

    green = rgb_1024[:, :, 1]
    gray = cv2.cvtColor(rgb_1024, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    score = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    score = cv2.GaussianBlur(score, (0, 0), 1.2)

    retinal_region = (gray > 12) & (green > 8)
    margin = int(0.08 * rgb_1024.shape[0])
    retinal_region[:margin, :] = False
    retinal_region[-margin:, :] = False
    retinal_region[:, :margin] = False
    retinal_region[:, -margin:] = False
    score[~retinal_region] = -1

    if np.all(score < 0):
        return np.array([[512.0, 512.0]], dtype=np.float32)
    y, x = np.unravel_index(np.argmax(score), score.shape)
    return np.array([[float(x), float(y)]], dtype=np.float32)


def _scaled_explicit_prompt(input_path: Path, point: tuple[float, float]):
    import numpy as np
    from PIL import Image

    with Image.open(input_path) as image:
        width, height = image.size
    x, y = point
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(
            f"Prompt point ({x}, {y}) is outside input image bounds "
            f"[0, {width}) x [0, {height})"
        )
    return np.array([[x * 1024.0 / width, y * 1024.0 / height]], dtype=np.float32)


def _restore_decoder_mlp_base_weights(backend, adapter_path: Path) -> int:
    """Map official SAM MLP names onto this fork without copying the checkpoint."""
    import torch

    model, _ = backend._load_model(str(adapter_path))
    base_state = torch.load(SAM_CHECKPOINT, map_location="cpu")
    remapped = {}
    for layer_index in range(2):
        source_prefix = f"mask_decoder.transformer.layers.{layer_index}.mlp"
        target_prefix = source_prefix
        key_pairs = (
            (f"{source_prefix}.lin1.weight", f"{target_prefix}.layers.0.0.weight"),
            (f"{source_prefix}.lin1.bias", f"{target_prefix}.layers.0.0.bias"),
            (f"{source_prefix}.lin2.weight", f"{target_prefix}.fc.weight"),
            (f"{source_prefix}.lin2.bias", f"{target_prefix}.fc.bias"),
        )
        model_state = model.state_dict()
        for source_key, target_key in key_pairs:
            if source_key not in base_state or target_key not in model_state:
                raise RuntimeError(f"Cannot map SAM base parameter {source_key} -> {target_key}")
            if base_state[source_key].shape != model_state[target_key].shape:
                raise RuntimeError(f"Shape mismatch for SAM base parameter {source_key}")
            remapped[target_key] = base_state[source_key]

    incompatible = model.load_state_dict(remapped, strict=False)
    unexpected = set(incompatible.unexpected_keys).intersection(remapped)
    if unexpected:
        raise RuntimeError(f"Mapped SAM base parameters were rejected: {sorted(unexpected)}")
    model.eval()
    return len(remapped)


def check(verify_sha256: bool) -> None:
    resources = _validate_files(verify_sha256)
    import torch

    payload = {
        "status": "ready" if torch.cuda.is_available() else "resources_ready_cuda_unavailable",
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "source_root": str(SOURCE_ROOT.resolve()),
        "resources": resources,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def segment(
    target: str,
    input_path: Path,
    output_path: Path,
    point: tuple[float, float] | None = None,
) -> None:
    _validate_files(verify_sha256=False)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = _load_backend()
    if point is not None and target != "retinal_vessel":
        raise ValueError("--point is currently supported only for --target retinal_vessel")

    prompt_mode = "upstream_automatic"
    selected_prompt = None
    base_parameters_restored = 0
    original_prompt_function = backend._automatic_prompt
    try:
        if target == "retinal_vessel":
            prompt_mode = "explicit_positive_point" if point is not None else "automatic_dark_line"

            def vessel_prompt(rgb_1024):
                nonlocal selected_prompt
                selected_prompt = (
                    _scaled_explicit_prompt(input_path, point)
                    if point is not None
                    else _automatic_vessel_prompt(rgb_1024)
                )
                return selected_prompt

            backend._automatic_prompt = vessel_prompt
            base_parameters_restored = _restore_decoder_mlp_base_weights(
                backend, VESSEL_ADAPTER
            )
            mask = backend._predict(
                str(input_path),
                str(output_path.parent),
                output_path.name,
                VESSEL_ADAPTER,
            )
        else:
            function = {
                "optic_disc": backend.segment_optic_disc,
                "optic_cup": backend.segment_optic_cup,
            }[target]
            mask = function(str(input_path), str(output_path.parent), output_path.name)
    finally:
        backend._automatic_prompt = original_prompt_function

    payload = {
        "status": "succeeded",
        "target": target,
        "input_uri": input_path.resolve().as_uri(),
        "output_uri": output_path.resolve().as_uri(),
        "output_sha256": _sha256(output_path),
        "shape": list(mask.shape),
        "foreground_pixels": int(mask.sum()),
        "prompt_mode": prompt_mode,
        "prompt_xy_1024": selected_prompt[0].tolist() if selected_prompt is not None else None,
        "base_parameters_restored": base_parameters_restored,
    }
    print(json.dumps(payload, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Validate source, weights and CUDA")
    check_parser.add_argument("--verify-sha256", action="store_true")

    segment_parser = subparsers.add_parser("segment", help="Run one segmentation tool")
    segment_parser.add_argument(
        "--target",
        choices=("optic_disc", "optic_cup", "retinal_vessel"),
        required=True,
    )
    segment_parser.add_argument("--input", type=Path, required=True)
    segment_parser.add_argument("--output", type=Path, required=True)
    segment_parser.add_argument(
        "--point",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Optional positive vessel point in original-image pixel coordinates",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "check":
        check(args.verify_sha256)
    else:
        segment(
            args.target,
            args.input,
            args.output,
            tuple(args.point) if args.point is not None else None,
        )


if __name__ == "__main__":
    main()
