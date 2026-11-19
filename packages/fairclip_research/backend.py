"""Pinned local model adapter; never calls clip.load or any network/download helper."""

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

from .core import (
    CLASSES,
    CONFIG,
    CONFIG_CHECKSUM,
    TEXTS,
    WEIGHT_BYTES,
    WEIGHT_SHA256,
    scores_from_cosines,
)


def file_digest(path, max_bytes):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("fairclip_resource_path_or_size")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint(path):
    if Path(path).stat().st_size != WEIGHT_BYTES or file_digest(path, 1500000000) != WEIGHT_SHA256:
        raise ValueError("fairclip_weight_binding_mismatch")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if (
            len(entries) > 4096
            or sum(x.file_size for x in entries) > 1500000000
            or any(x.file_size > 536870912 for x in entries)
            or any(
                x.flag_bits & 1 or ".." in Path(x.filename).parts or x.filename.startswith("/")
                for x in entries
            )
            or len({x.filename for x in entries}) != len(entries)
        ):
            raise ValueError("fairclip_archive_bounds")


def load_state(path):
    import torch

    if torch.__version__.split("+")[0] != "2.14.0":
        raise ValueError("fairclip_loader_version_drift")
    verify_checkpoint(path)
    # No safe_globals, JIT, pickle fallback, optimizer restore or downloaded code.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if type(checkpoint) is not dict or "model_state_dict" not in checkpoint:
        raise ValueError("fairclip_checkpoint_structure")
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict) or not 100 <= len(state) <= 512:
        raise ValueError("fairclip_state_structure")
    for name, tensor in state.items():
        if type(name) is not str or type(tensor) is not torch.Tensor:
            raise ValueError("fairclip_tensor_only_state")
        if (
            tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or not torch.isfinite(tensor).all().item()
        ):
            raise ValueError("fairclip_invalid_tensor")
    shapes = {
        "visual.conv1.weight": (768, 3, 16, 16),
        "visual.positional_embedding": (197, 768),
        "visual.proj": (768, 512),
        "token_embedding.weight": (49408, 512),
        "positional_embedding": (77, 512),
        "text_projection": (512, 512),
    }
    if any(k not in state or tuple(state[k].shape) != v for k, v in shapes.items()):
        raise ValueError("fairclip_full_vit_b16_text_image_required")
    return state


def module_from_reviewed_file(path, expected):
    if file_digest(path, 2000000) != expected:
        raise ValueError("fairclip_support_resource_drift")
    spec = importlib.util.spec_from_file_location("fairclip_reviewed_" + Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(state, support):
    module = module_from_reviewed_file(support["model"]["path"], support["model"]["sha256"])
    # Original architecture factory does strict load_state_dict; no newly trained head.
    model = module.build_model(dict(state)).float().eval()
    model.requires_grad_(False)
    return model


def tokenize(support):
    import torch

    item = support["tokenizer"]
    module = module_from_reviewed_file(item["path"], item["sha256"])
    bpe = support["bpe"]
    if file_digest(bpe["path"], 2000000) != bpe["sha256"]:
        raise ValueError("fairclip_bpe_drift")
    tokenizer = module.SimpleTokenizer(bpe["path"])
    result = torch.zeros((2, 77), dtype=torch.int)
    for index, text in enumerate(TEXTS):
        tokens = [tokenizer.encoder["<|startoftext|>"]]
        tokens += tokenizer.encode(text) + [tokenizer.encoder["<|endoftext|>"]]
        if len(tokens) > 77:
            raise ValueError("fairclip_text_length")
        result[index, : len(tokens)] = torch.tensor(tokens)
    return result


def author_gpu_precision(model, support):
    module = module_from_reviewed_file(support["model"]["path"], support["model"]["sha256"])
    module.convert_weights(model)
    return model


def preprocess(raw):
    import numpy as np
    import torch
    from PIL import Image

    if len(raw) > 4194304:
        raise ValueError("fairclip_input_size")
    with Image.open(io.BytesIO(raw)) as source:
        if source.format != "PNG" or source.mode != "L" or source.width * source.height > 16777216:
            raise ValueError("fairclip_visible_grayscale_SLO_required")
        # Author converts SLO to float32 PIL before resize/crop/RGB conversion.
        image = Image.fromarray(np.asarray(source, dtype=np.float32))
    width, height = image.size
    if width <= height:
        size = (224, int(224 * height / width))
    else:
        size = (int(224 * width / height), 224)
    image = image.resize(size, Image.Resampling.BICUBIC)
    left, top = int(round((size[0] - 224) / 2.0)), int(round((size[1] - 224) / 2.0))
    image = image.crop((left, top, left + 224, top + 224)).convert("RGB")
    value = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1)) / 255.0
    mean = torch.tensor(CONFIG["preprocess"]["mean"]).reshape(3, 1, 1)
    std = torch.tensor(CONFIG["preprocess"]["std"]).reshape(3, 1, 1)
    return ((value - mean) / std).unsqueeze(0)


def predict(raw, model, support, *, device):
    import torch

    if device != "cuda:0":
        raise ValueError("fairclip_single_device")
    image = preprocess(raw).to(device)
    tokens = tokenize(support).to(device)
    author_gpu_precision(model, support)
    model = model.to(device)
    with torch.inference_mode():
        image_features = model.encode_image(image)
        text_features = model.encode_text(tokens)
        for features in (image_features, text_features):
            if not torch.isfinite(features).all() or (features.norm(dim=-1) == 0).any():
                raise ValueError("fairclip_invalid_encoded_features")
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.T
        cosines = tuple(float(v) for v in logits[0].cpu())
        scores = tuple(float(v) for v in logits.softmax(dim=1)[0].cpu())
    scores_from_cosines(cosines)  # Validate finite, bounded logits before emitting anything.
    index = 0 if scores[0] >= scores[1] else 1
    return dict(
        schema_version="fairclip-output/1.0.0",
        input_checksum="sha256:" + hashlib.sha256(raw).hexdigest(),
        model_checksum="sha256:" + WEIGHT_SHA256,
        config_checksum=CONFIG_CHECKSUM,
        raw_cosines=cosines,
        normalized_model_scores=scores,
        predicted_index=index,
        predicted_class=CLASSES[index],
        calibration="unverified",
        research_only=True,
        decisive_evidence=False,
        demographics_used=[],
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--support", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda:0",), required=True)
    args = parser.parse_args()
    support = json.loads(Path(args.support).read_text())
    model = build_model(load_state(args.weights), support)
    data = predict(Path(args.input).read_bytes(), model, support, device=args.device)
    with Path(args.output).open("x") as stream:
        json.dump(data, stream, allow_nan=False)


if __name__ == "__main__":
    main()
