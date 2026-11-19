"""Author's two normalized cosine logits and unit-temperature softmax, batch size one."""

import hashlib
import json
import math
import struct

WEIGHT_SHA256 = "1ef73b32d7eab8130c5da3df93415775bba6fffa61ad5828d40c7545c822105e"
WEIGHT_BYTES = 1342885474
SOURCE_COMMIT = "91b05b61acafcb0716c4c4d8c53a747fdf00b8ac"
TEXTS = ("A photo of non-glaucoma", "A photo of glaucoma")
CLASSES = ("non-glaucoma", "glaucoma")
CONFIG = dict(
    version="fairclip-classification/1.0.0",
    source_commit=SOURCE_COMMIT,
    model_sha256=WEIGHT_SHA256,
    class_names=CLASSES,
    texts=TEXTS,
    preprocess=dict(
        input="visible_grayscale8_SLO_PNG",
        resize_input="PIL_F_float32",
        resize_short_edge=224,
        interpolation="PIL_BICUBIC",
        center_crop=224,
        convert="RGB_after_resize_crop",
        tensor_scale=255.0,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
    feature_normalization="L2_image_and_text",
    similarity="dot_product",
    temperature=1.0,
    learned_logit_scale_used=False,
    inference_precision="author_CLIP_convert_weights_mixed_FP16_on_GPU",
    softmax_precision="float16_output_matching_author_GPU_feature_dtype",
    decision="argmax_first_index_on_exact_tie",
    calibration="unverified",
    unused_demographics=("age", "gender", "race", "ethnicity", "language", "maritalstatus"),
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


CONFIG_CHECKSUM = "sha256:" + hashlib.sha256(canonical(CONFIG)).hexdigest()


def scores_from_cosines(values):
    if len(values) != 2 or any(
        isinstance(v, bool) or not math.isfinite(v) or abs(v) > 1.00001 for v in values
    ):
        raise ValueError("fairclip_invalid_cosines")
    exp = [math.exp(float(v) - max(values)) for v in values]
    scores = tuple(struct.unpack("e", struct.pack("e", v / sum(exp)))[0] for v in exp)
    return scores, 0 if scores[0] >= scores[1] else 1


def classify_features(image, texts):
    """Independent arithmetic reference; no label/note/demographic input port."""
    if len(texts) != 2 or not image or any(len(t) != len(image) for t in texts):
        raise ValueError("fairclip_feature_shape")

    def normalize(values):
        if any(isinstance(v, bool) or not math.isfinite(v) for v in values):
            raise ValueError("fairclip_nonfinite_feature")
        norm = math.hypot(*values)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("fairclip_zero_or_invalid_norm")
        return tuple(v / norm for v in values)

    image = normalize(image)
    cosines = tuple(sum(a * b for a, b in zip(image, normalize(t), strict=True)) for t in texts)
    scores, prediction = scores_from_cosines(cosines)
    return cosines, scores, prediction
