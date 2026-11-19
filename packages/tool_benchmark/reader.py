"""Single 06E-owned G2 reader. Only explicit selections; no clinical data access."""

import re
from pathlib import Path

from packages.grounding.models import seal, validate
from packages.object_store import checksum
from packages.tool_execution.process import read_file
from packages.tool_execution.qc import grayscale_png

from .formats import contour_mask, grayscale_bmp, jpeg_dimensions
from .models import (
    BenchmarkError,
    BenchmarkManifest,
    BenchmarkSample,
    FileReference,
    LabelDefinition,
    ModelInput,
    TruthReference,
)


def reference(root, relative_path):
    path = Path(root) / relative_path
    if not path.is_absolute() or path.resolve() != path or not path.is_relative_to(root):
        raise BenchmarkError("dataset_path_rejected")
    data = read_file(path, 16777216)
    return FileReference(
        relative_path=relative_path,
        uri=path.as_uri(),
        checksum=checksum(data),
        byte_size=len(data),
    )


def bound_read(root, ref):
    ref = validate(FileReference, ref)
    actual = reference(root, ref.relative_path)
    if actual != ref:
        raise BenchmarkError("dataset_content_changed")
    data = read_file(Path(root) / ref.relative_path, ref.byte_size)
    if checksum(data) != ref.checksum:
        raise BenchmarkError("dataset_content_changed")
    return data


def papila_sample(root, sample_id, *, split):
    if not re.fullmatch(r"RET[0-9]{3}(OD|OS)", sample_id):
        raise BenchmarkError("papila_sample_id_invalid")
    root = Path(root)
    version = bound_read(root, reference(root, "README.md")).decode().splitlines()[0]
    if version != "# PAPILAv1.1":
        raise BenchmarkError("papila_version_unrecognized")
    # Alternate copies must not silently override one expert; exclude ambiguous samples.
    names = sorted(
        p.name for p in (root / "ExpertsSegmentations/Contours").glob(sample_id + "_*.txt")
    )
    expected = {
        f"{sample_id}_{target}_{expert}.txt"
        for target in ("disc", "cup")
        for expert in ("exp1", "exp2")
    }
    if set(names) != expected:
        raise BenchmarkError("papila_missing_or_ambiguous_expert_contours")
    image = reference(root, f"FundusImages/{sample_id}.jpg")
    shape = jpeg_dimensions(bound_read(root, image))
    truths = tuple(
        TruthReference(
            file=reference(
                root, f"ExpertsSegmentations/Contours/{sample_id}_{target}_{expert}.txt"
            ),
            target="optic_" + target,
            expert=expert,
            encoding="xy_contour",
        )
        for target in ("disc", "cup")
        for expert in ("exp1", "exp2")
    )
    # Reserved references are hashed, not rasterized/evaluated during selection.
    return BenchmarkSample(
        sample_id=sample_id,
        dataset="PAPILA",
        dataset_version="PAPILAv1.1",
        group_id="PAPILA:" + sample_id[:6],
        group_basis="papila_filename_patient",
        split=split,
        official_split="unspecified",
        prior_exposure="development" if split == "development" else "unknown",
        image=image,
        dimensions=shape,
        laterality="right" if sample_id.endswith("OD") else "left",
        truths=truths,
    )


def refuge_sample(root, sample_id, *, official_split, split, labels):
    labels = validate(LabelDefinition, labels)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", sample_id) or official_split not in (
        "train",
        "val",
        "test",
    ):
        raise BenchmarkError("refuge_sample_id_invalid")
    source = bound_read(root, labels.source).decode("utf-8")
    if labels.source_excerpt not in source:
        raise BenchmarkError("label_definition_source_drift")
    if split == "development" and official_split != "train":
        raise BenchmarkError("refuge_development_requires_train")
    image = reference(root, f"{official_split}/images/{sample_id}.jpg")
    candidates = [
        f"{official_split}/mask/{sample_id}{extension}"
        for extension in (".bmp", ".png")
        if (Path(root) / f"{official_split}/mask/{sample_id}{extension}").is_file()
    ]
    if len(candidates) != 1:
        raise BenchmarkError("refuge_missing_or_ambiguous_mask")
    mask = reference(root, candidates[0])
    encoding = "bmp_gray_labels" if candidates[0].endswith(".bmp") else "png_gray_labels"
    shape = jpeg_dimensions(bound_read(root, image))
    return BenchmarkSample(
        sample_id=sample_id,
        dataset="REFUGE2",
        dataset_version="local-copy:" + labels.source.checksum,
        group_id="REFUGE2:image:" + sample_id,
        group_basis="unlinked_image_only",
        split=split,
        official_split=official_split,
        prior_exposure="development" if split == "development" else "unknown",
        image=image,
        dimensions=shape,
        laterality="unknown",
        truths=tuple(
            TruthReference(file=mask, target=target, expert="challenge", encoding=encoding)
            for target in ("optic_disc", "optic_cup")
        ),
    )


def manifest(samples, *, labels=None):
    return seal(BenchmarkManifest, samples=tuple(samples), label_definition=labels)


class BenchmarkReader:
    """Deployment-owned roots + exact manifest. The caller selects identities, not paths.

    Reserved inputs/truth stay locked in this preparation release. A future versioned
    formal evaluator must bind policy, group/exposure evidence and all resource hashes.
    """

    def __init__(self, roots, document):
        self.manifest = validate(BenchmarkManifest, document)
        self.roots = {name: Path(root) for name, root in roots.items()}
        for sample in self.manifest.samples:
            root = self.roots[sample.dataset]
            expected = (
                papila_sample(root, sample.sample_id, split=sample.split)
                if sample.dataset == "PAPILA"
                else refuge_sample(
                    root,
                    sample.sample_id,
                    official_split=sample.official_split,
                    split=sample.split,
                    labels=self.manifest.label_definition,
                )
            )
            if expected != sample:
                raise BenchmarkError("sample_source_binding_mismatch")

    def sample(self, dataset, sample_id):
        # Reparse on every public call: forged model_copy fields are never authority.
        doc = validate(BenchmarkManifest, self.manifest)
        for sample in doc.samples:
            if (sample.dataset, sample.sample_id) == (dataset, sample_id):
                return sample
        raise BenchmarkError("sample_not_registered")

    def model_input(self, dataset, sample_id):
        sample = self.sample(dataset, sample_id)
        if sample.split != "development":
            raise BenchmarkError("formal_evaluation_locked")
        root = self.roots[dataset]
        raw = bound_read(root, sample.image)
        if len(raw) > 4194304:
            raise BenchmarkError("model_input_object_limit")
        if jpeg_dimensions(raw) != sample.dimensions:
            raise BenchmarkError("image_shape_changed")
        return ModelInput(
            benchmark_checksum=self.manifest.checksum,
            sample_id=sample_id,
            image=sample.image,
            dimensions=sample.dimensions,
            laterality=sample.laterality,
        )

    def truth_masks(self, dataset, sample_id, target):
        sample = self.sample(dataset, sample_id)
        if sample.split != "development":
            raise BenchmarkError("formal_evaluation_locked")
        if target not in ("optic_disc", "optic_cup"):
            raise BenchmarkError("truth_target_missing")
        root = self.roots[dataset]
        result = {}
        for truth in sample.truths:
            if truth.target != target:
                continue
            raw = bound_read(root, truth.file)
            if truth.encoding == "xy_contour":
                pixels = contour_mask(raw, sample.dimensions)
            else:
                shape, raw_pixels = (
                    grayscale_bmp(raw)
                    if truth.encoding == "bmp_gray_labels"
                    else grayscale_png(raw, 16777216)
                )
                labels = self.manifest.label_definition
                if shape != sample.dimensions:
                    raise BenchmarkError("truth_shape_mismatch")
                if not set(raw_pixels) <= {labels.background, labels.disc_rim, labels.cup}:
                    raise BenchmarkError("truth_labels_invalid")
                selected = {labels.cup, labels.disc_rim} if target == "optic_disc" else {labels.cup}
                pixels = bytes(int(p in selected) for p in raw_pixels)
            result[truth.expert] = pixels
        return result
