"""Actual PNG bytes and original QC interface; no model/API/clinical assertions."""

import base64
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.grounding.models import canonical, digest, validate
from packages.grounding.synthetic import ref
from packages.object_store import checksum
from packages.tool_execution.fundus_bundle import (
    COMMIT,
    MAX_BYTES,
    TRANSFORM,
    FundusBundlePolicy,
    FundusBundleQCRegistry,
    decode_bundle,
    encode_bundle,
)
from packages.tool_execution.models import ExecutionError
from tests.dataset_adapters.test_multimodal import jpeg
from tests.tool_admission.support import fixture
from tests.tool_execution.test_geometry_qc import mask_png


@pytest.fixture(scope="module")
def material():
    labels = bytes([0, 1, 2, 0] * (512 * 512 // 4))
    masks = {"labels": mask_png((512, 512), labels)}
    for name, selected in (("disc", {1, 2}), ("cup", {2}), ("rim", {1})):
        masks[name] = mask_png((512, 512), bytes(255 if p in selected else 0 for p in labels))
    raw = jpeg()
    blob = encode_bundle(
        input_checksum=checksum(raw),
        input_shape=(2, 3),
        source_checksum=digest("synthetic source lock"),
        masks=masks,
        device="cpu",
        ensemble_std_mean=0.125,
    )
    return SimpleNamespace(masks=masks, image=raw, bundle=blob)


def test_bundle_roundtrip_preserves_actual_masks(material):
    bundle, masks = decode_bundle(material.bundle)
    assert masks == material.masks
    assert bundle.input_shape == (2, 3) and bundle.analysis_shape == (512, 512)
    assert bundle.disease_probability is None and bundle.physical_scale is None


def test_empty_masks_are_valid_technical_output(material):
    png = mask_png((512, 512), bytes(512 * 512))
    raw = encode_bundle(
        input_checksum=checksum(material.image),
        input_shape=(2, 3),
        source_checksum=digest("empty"),
        masks=dict.fromkeys(material.masks, png),
        device="cpu",
        ensemble_std_mean=0.0,
    )
    assert decode_bundle(raw)[1]["disc"] == png


@pytest.mark.parametrize(
    "mode",
    [
        "duplicate",
        "nan",
        "infinity",
        "bool_shape",
        "extra",
        "unknown_version",
        "engine",
        "commit",
        "transform",
        "shape",
        "base64",
        "png_hash",
        "crc",
        "label",
        "class_binding",
        "missing_mask",
        "too_large",
        "truncated",
    ],
)
def test_invalid_bundle_rejected(material, mode):
    value = json.loads(material.bundle)
    if mode == "duplicate":
        data = b'{"device":"cpu",' + material.bundle[1:]
    elif mode in {"nan", "infinity"}:
        value["ensemble_std_mean"] = float("nan" if mode == "nan" else "inf")
        data = json.dumps(value).encode()
    elif mode == "too_large":
        data = b" " * (MAX_BYTES + 1)
    elif mode == "truncated":
        data = material.bundle[:-1]
    else:
        if mode == "bool_shape":
            value["input_shape"][0] = True
        if mode == "extra":
            value["approved"] = True
        if mode == "unknown_version":
            value["schema_version"] = "automorph-fundus-bundle/99.0.0"
        if mode == "engine":
            value["engine"] = "other"
        if mode == "commit":
            value["source_commit"] = "0" * 40
        if mode == "transform":
            value["transform"] = "guessed_resize"
        if mode == "shape":
            value["analysis_shape"] = [2, 3]
        if mode == "base64":
            value["masks"]["disc"]["data_base64"] += "!"
        if mode == "png_hash":
            value["masks"]["disc"]["checksum"] = digest("different")
        if mode in {"crc", "label", "class_binding"}:
            name = "labels" if mode == "label" else "disc"
            raw = (
                (material.masks[name][:-1] + b"!")
                if mode == "crc"
                else mask_png((512, 512), bytes([3 if mode == "label" else 0]) * (512 * 512))
            )
            value["masks"][name] = dict(
                checksum=checksum(raw), data_base64=base64.b64encode(raw).decode()
            )
        if mode == "missing_mask":
            del value["masks"]["cup"]
        data = json.dumps(value).encode()
    with pytest.raises(ExecutionError, match="fundus_bundle_invalid"):
        decode_bundle(data)


def qc_fixture(tmp_path, material):
    f = fixture(tmp_path)
    original = next(iter(f.owners.records.values())).artifact
    uri = f.objects.promote(f.objects.stage(material.image), checksum(material.image))
    original = validate(
        type(original),
        dict(
            original.model_dump(mode="json"),
            uri=uri,
            checksum=checksum(material.image),
            media_type="image/jpeg",
            dimensions=[2, 3],
            coordinate_space="PIXEL_2D",
            unit=None,
        ),
    )
    constraint = f.mapping.candidates[0].output.model_copy(
        update=dict(
            laterality=original.laterality,
            coordinate_space="PIXEL_2D",
            unit=validate(
                type(f.mapping.candidates[0].output.unit), dict(state="not_applicable", code=None)
            ),
        )
    )
    policy = FundusBundlePolicy(
        artifact_ref=constraint.artifact_ref,
        artifact_type_code="ARTIFACT:SYNTHETIC:FUNDUS_BUNDLE",
        input_artifact_type_code=original.artifact_type_code,
        unit_term=constraint.unit,
        source_checksum=digest("synthetic source lock"),
    )
    p = ref("policy", "POLICY:SYNTHETIC:FUNDUS_BUNDLE").model_copy(
        update=dict(checksum=digest(policy))
    )
    output = SimpleNamespace(
        qc_policy_ref=p, qc_port="technical_output_qc/1.0.0", constraint=constraint
    )
    rule = SimpleNamespace(
        media_types=("application/json",), dimensions_min=(1,), dimensions_max=(1,)
    )
    return original, FundusBundleQCRegistry({p: policy}, f.objects), output, rule


@pytest.mark.parametrize(
    "change", [None, "source", "input", "shape", "eye", "media", "count", "rule"]
)
def test_original_qc_binds_installed_policy_and_input(tmp_path, material, change):
    source, qc, output, rule = qc_fixture(tmp_path, material)
    value = json.loads(material.bundle)
    if change == "source":
        value["source_checksum"] = digest("other")
    if change == "input":
        value["input_checksum"] = digest("other")
    if change == "shape":
        value["input_shape"] = [3, 2]
    if change == "eye":
        source = source.model_copy(
            update=dict(laterality="RIGHT" if source.laterality.value != "RIGHT" else "LEFT")
        )
    if change == "media":
        source = source.model_copy(update=dict(media_type="application/octet-stream"))
    if change == "rule":
        rule.dimensions_max = (512, 512)
    data = canonical(value)
    inputs = [] if change == "count" else [source]
    if change:
        with pytest.raises(ExecutionError, match="fundus_bundle_"):
            qc.evaluate(output, rule, data, inputs)
    else:
        observed = qc.evaluate(output, rule, data, inputs)
        assert observed.status.value == "PASS" and observed.dimensions == (1,)
        assert observed.content_checksum == checksum(data)


def prediction(directory, material, source_checksum):
    directory.mkdir()
    artifacts = {name + ".png": raw for name, raw in material.masks.items()}
    artifacts["overlay.png"] = b"synthetic decorative bytes not an admitted output"
    for name, raw in artifacts.items():
        (directory / name).write_bytes(raw)
    result = dict(
        schema_version="automorph-fundus-measurement/1.0.0",
        input_checksum=checksum(material.image),
        source_checksum=source_checksum,
        input_shape=[2, 3],
        analysis_shape=[512, 512],
        transform=TRANSFORM,
        model="AutoMorph-disc-cup-eight-WNet",
        device="cpu",
        status="predictions_saved",
        ensemble_std_mean=0.125,
        quality_is_calibrated_accuracy=False,
        masks_modified=False,
        missing_is_negative=False,
        disease_probability=None,
        training_overlap="GAMMA_REFUGE_in_upstream_training_config_not_external_test",
        validation_status="research_smoke_only",
        physical_scale=None,
        artifacts={n: checksum(v) for n, v in artifacts.items()},
        runtime=dict(torch="synthetic", cuda=None),
    )
    (directory / "result.json").write_bytes(canonical(result))


@pytest.mark.parametrize("failure", [None, "worker", "input", "source", "artifact"])
def test_new_adapter_explicit_resource_root_no_model(tmp_path, material, monkeypatch, failure):
    from apps import automorph_06, automorph_resources
    from packages.multimodal_research import fundus_worker

    resources = tmp_path / "resources"
    resources.mkdir()
    lock = canonical(dict(commit=COMMIT, files={}))
    (resources / "ensemble-source-lock.json").write_bytes(lock)
    image = tmp_path / "input.jpg"
    image.write_bytes(material.image)
    output = tmp_path / "bundle.json"
    previous = (automorph_resources.ROOT, fundus_worker.ROOT)
    calls = []

    def verify():
        assert automorph_resources.ROOT == fundus_worker.ROOT == resources
        calls.append("verify")

    def worker(path, out):
        assert Path(path).read_bytes() == material.image
        calls.append("injected_not_real_model")
        if failure == "worker":
            raise OSError("injected worker failure")
        prediction(out, material, checksum(lock))
        if failure == "input":
            image.write_bytes(b"changed input")
        if failure == "source":
            (resources / "ensemble-source-lock.json").write_bytes(b"changed source")
        if failure == "artifact":
            (out / "cup.png").write_bytes(b"changed artifact")

    monkeypatch.setattr(fundus_worker, "verify_resources", verify)
    monkeypatch.setattr(fundus_worker, "run", worker)
    monkeypatch.chdir(tmp_path)
    arguments = dict(
        image=image, output=output, resource_root=resources, source_checksum=checksum(lock)
    )
    if failure:
        error = OSError if failure == "worker" else ValueError
        message = {
            "worker": "injected worker failure",
            "input": "input_or_source_changed",
            "source": "input_or_source_changed",
            "artifact": "artifact_binding",
        }[failure]
        with pytest.raises(error, match=message):
            automorph_06.run(**arguments)
        assert not output.exists()
        assert (automorph_resources.ROOT, fundus_worker.ROOT) == previous
        assert calls.count("injected_not_real_model") == 1
        return
    automorph_06.run(**arguments)
    assert calls == ["verify", "injected_not_real_model", "verify"]
    assert decode_bundle(output.read_bytes())[1] == material.masks
    assert (automorph_resources.ROOT, fundus_worker.ROOT) == previous
    with pytest.raises(ValueError, match="output_exists"):
        automorph_06.run(
            image=image, output=output, resource_root=resources, source_checksum=checksum(lock)
        )
    assert len(calls) == 3


def test_direct_entrypoint_help_independent_of_cwd(tmp_path):
    entry = Path(__file__).resolve().parents[2] / "apps/automorph_06.py"
    p = subprocess.run(
        [sys.executable, str(entry), "--help"], cwd=tmp_path, capture_output=True, timeout=15
    )
    assert p.returncode == 0, p.stderr
    assert b"--resource-root" in p.stdout and b"--source-checksum" in p.stdout


@pytest.mark.parametrize("failure", ["changed", "missing"])
def test_qc_rereads_original_input_without_hiding_technical_failure(
    tmp_path, material, monkeypatch, failure
):
    source, qc, output, rule = qc_fixture(tmp_path, material)

    def read(*args):
        if failure == "missing":
            raise FileNotFoundError("original input unavailable")
        return b"wrong original content"

    monkeypatch.setattr(qc.objects, "read_bounded", read)
    error = FileNotFoundError if failure == "missing" else ExecutionError
    message = "original input unavailable" if failure == "missing" else "input_changed"
    with pytest.raises(error, match=message):
        qc.evaluate(output, rule, material.bundle, [source])
