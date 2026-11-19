"""Frozen, source-bound reanalysis of existing predictions; no inference or DB writes."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from packages.grounding.models import seal, validate
from packages.tool_execution.qc import grayscale_png

from .development_models import DevelopmentProtocol, DevelopmentSample
from .fairseg import FairSegDevelopmentReader
from .geometry_metrics import compare
from .models import BenchmarkError
from .reader import bound_read, reference

ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT = Path("/opt/data/test/segman-slo-deployment")
DATA = Path("/opt/data/test/download/FairSeg/Dataset")
PREDICTIONS = DEPLOYMENT / "results/fairseg-check-001"
SOURCES = (
    "/opt/data/test/FairSeg/README.md",
    "/opt/data/test/FairSeg/SAMed/utils.py",
    "/opt/data/test/ControlGlaucoma-SegMAN/train/eval_generated_vcdr.py",
    str(DEPLOYMENT / "scripts/segman_slo.py"),
    str(DEPLOYMENT / "weights/segman_b.pth"),
    str(ROOT / "configs/tools/segman-slo-resources-v1.json"),
)


def sha(path):
    with Path(path).open("rb") as f:
        return "sha256:" + hashlib.file_digest(f, "sha256").hexdigest()


def implementation():
    paths = [
        *Path(__file__).parent.glob("*.py"),
        ROOT / "apps/tool_development_evaluate.py",
        ROOT / "packages/dataset_adapters/reading.py",
        ROOT / "packages/tool_execution/qc.py",
        ROOT / "packages/tool_execution/process.py",
        ROOT / "packages/grounding/models.py",
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def prepare():
    old_ref = reference(PREDICTIONS, "manifest.json")
    receipt_ref = reference(PREDICTIONS, "predictions-complete.json")
    old = json.loads(bound_read(PREDICTIONS, old_ref))
    receipts = json.loads(bound_read(PREDICTIONS, receipt_ref))
    if (
        old["selected_count"] != 8
        or receipts["predicted_count"] != 8
        or receipts["ground_truth_read"] is not False
        or receipts["manifest_sha256"] != old_ref.checksum.removeprefix("sha256:")
        or [i["id"] for i in old["files"]] != [i["id"] for i in receipts["files"]]
    ):
        raise BenchmarkError("historical_population_binding")
    samples = []
    for item, saved in zip(old["files"], receipts["files"], strict=True):
        identity = item["id"]
        data = reference(DATA, f"Test/{identity}.npz")
        pred = reference(PREDICTIONS, f"predictions/{identity}/labels.png")
        image = reference(PREDICTIONS, f"inputs/{identity}.png")
        if (
            data.checksum != "sha256:" + item["sha256"]
            or pred.checksum != "sha256:" + saved["labels_sha256"]
            or image.checksum != "sha256:" + saved["extracted_png_sha256"]
            or saved["status"] != "success"
        ):
            raise BenchmarkError("historical_source_drift")
        samples.append(
            DevelopmentSample(
                sample_id=identity,
                data=data,
                prediction=pred,
                input_png=image,
                result=reference(PREDICTIONS, f"predictions/{identity}/result.json"),
            )
        )
    sources = {p: sha(p) for p in SOURCES}
    lock = json.loads(Path(SOURCES[-1]).read_text())
    if sources[SOURCES[-2]] != "sha256:" + lock["checkpoint_sha256"]:
        raise BenchmarkError("checkpoint_binding")
    # These are source-code statements, not anatomy guessed from pixel frequencies.
    code = Path(SOURCES[2]).read_text()
    if (
        "cond[mask == -1] = (0, 0, 255)   # disc rim = blue" not in code
        or "cond[mask == -2] = (255, 0, 0)   # cup = red" not in code
    ):
        raise BenchmarkError("label_source_definition_changed")
    return seal(
        DevelopmentProtocol,
        prepared_at=datetime.now(UTC).isoformat(),
        dataset_root=str(DATA),
        prediction_root=str(PREDICTIONS),
        sources=sources,
        implementation=implementation(),
        historical_manifest=old_ref,
        historical_predictions=receipt_ref,
        samples=tuple(samples),
    )


def verify(protocol):
    protocol = validate(DevelopmentProtocol, protocol)
    if protocol.implementation != implementation():
        raise BenchmarkError("evaluation_implementation_drift")
    if set(protocol.sources) != set(SOURCES) or any(
        sha(p) != v for p, v in protocol.sources.items()
    ):
        raise BenchmarkError("evaluation_source_drift")
    if protocol.dataset_root != str(DATA) or protocol.prediction_root != str(PREDICTIONS):
        raise BenchmarkError("evaluation_root_mismatch")
    old = json.loads(bound_read(PREDICTIONS, protocol.historical_manifest))
    receipts = json.loads(bound_read(PREDICTIONS, protocol.historical_predictions))
    if (
        [s.sample_id for s in protocol.samples] != [i["id"] for i in old["files"]]
        or [s.sample_id for s in protocol.samples] != [i["id"] for i in receipts["files"]]
        or receipts["manifest_sha256"]
        != protocol.historical_manifest.checksum.removeprefix("sha256:")
    ):
        raise BenchmarkError("historical_population_binding")
    for s, item, receipt in zip(protocol.samples, old["files"], receipts["files"], strict=True):
        if (
            s.data.relative_path != f"Test/{s.sample_id}.npz"
            or s.prediction.relative_path != f"predictions/{s.sample_id}/labels.png"
            or s.result.relative_path != f"predictions/{s.sample_id}/result.json"
            or s.input_png.relative_path != f"inputs/{s.sample_id}.png"
            or s.data.checksum != "sha256:" + item["sha256"]
            or s.prediction.checksum != "sha256:" + receipt["labels_sha256"]
            or s.input_png.checksum != "sha256:" + receipt["extracted_png_sha256"]
        ):
            raise BenchmarkError("historical_sample_binding")
    return protocol


def summarize(rows):
    computed = [r for r in rows if r["status"] == "computed"]
    aggregates = {}
    for target in ("disc", "cup", "rim"):
        aggregates[target] = {}
        for metric in (
            "dice",
            "iou",
            "relative_area_error",
            "area_fraction_error",
            "assd_px",
            "hd95_px",
        ):
            values = [
                r["metrics"][target][metric]
                for r in computed
                if r["metrics"][target][metric] is not None
            ]
            aggregates[target][metric] = dict(
                n=len(values),
                mean=sum(values) / len(values) if values else None,
                min=min(values) if values else None,
                max=max(values) if values else None,
                confidence_interval=None,
            )
    errors = {}
    for name in ("vertical_cdr", "area_cdr"):
        values = [r["metrics"]["ratios"][name]["absolute_error"] for r in computed]
        values = [v for v in values if v is not None]
        errors[name] = dict(n=len(values), mae=sum(values) / len(values) if values else None)
    return dict(
        selected=len(rows),
        computed=len(computed),
        technical_failures=len(rows) - len(computed),
        technical_failure_rate=(len(rows) - len(computed)) / len(rows) if rows else None,
        qc_flagged=sum(bool(r["metrics"]["qc_flags"]) for r in computed),
        qc_denominator=len(computed),
        metrics=aggregates,
        ratio_errors=errors,
    )


def evaluate(protocol):
    protocol = verify(protocol)
    reader = FairSegDevelopmentReader(
        DATA, {s.sample_id: s.data for s in protocol.samples}, purpose=protocol.purpose
    )
    lock = json.loads(Path(SOURCES[-1]).read_text())
    started, clock, rows = datetime.now(UTC).isoformat(), time.monotonic(), []
    for sample in protocol.samples:
        row = dict(sample_id=sample.sample_id, status="failed", reason=None)
        try:
            if time.monotonic() - clock > protocol.max_seconds:
                raise BenchmarkError("evaluation_time_budget")
            shape, image = reader.model_input(sample.sample_id)
            if grayscale_png(image, 16777216) != grayscale_png(
                bound_read(PREDICTIONS, sample.input_png), 16777216
            ):
                raise BenchmarkError("prediction_image_alignment")
            pred_shape, pixels = grayscale_png(bound_read(PREDICTIONS, sample.prediction), 16777216)
            result = json.loads(bound_read(PREDICTIONS, sample.result))
            if (
                pred_shape != shape
                or result["preprocessing"]["original_size_hw"] != list(shape)
                or result["status"] != "success"
                or result["ground_truth_used"] is not False
                or result["checkpoint_sha256"] != lock["checkpoint_sha256"]
                or result["source_commit"] != lock["source_commit"]
                or result["input_sha256"] != sample.input_png.checksum.removeprefix("sha256:")
                or result["outputs"]["labels.png"]["sha256"]
                != sample.prediction.checksum.removeprefix("sha256:")
            ):
                raise BenchmarkError("prediction_result_binding")
            truth_shape, truth = reader.truth(sample.sample_id)
            if truth_shape != shape:
                raise BenchmarkError("truth_shape_mismatch")
            row.update(
                status="computed",
                shape=list(shape),
                strata=reader.strata(sample.sample_id),
                metrics=compare(pixels, truth, shape),
            )
        except (BenchmarkError, ValueError, OSError, KeyError) as error:
            row["reason"] = (
                str(error)
                if isinstance(error, BenchmarkError)
                else "technical_read_or_format_error"
            )
        rows.append(row)
    verify(protocol)  # source/code changes during evaluation cannot yield a valid report
    report = summarize(rows)
    report.update(
        schema_version="development-results/1.0.0",
        protocol_checksum=protocol.checksum,
        started_at=started,
        ended_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.monotonic() - clock,
        new_model_calls=0,
        reused_predictions=len(protocol.samples),
        independent_test=False,
        confidence_interval_reason=protocol.confidence_interval,
        tool_qualification_updated=False,
        patient_independence_proven=False,
        unit="image",
        device_strata="unavailable",
        age_strata="deferred_age_unit_not_bound",
    )
    strata = {}
    for field in protocol.strata_fields:
        strata[field] = {}
        for code in (-1, 0, 1, 2) if field == "race" else (-1, 0, 1):
            subset = [r for r in rows if r.get("strata", {}).get(field) == code]
            strata[field][str(code)] = (
                summarize(subset)
                if len(subset) >= protocol.min_public_stratum
                else {"status": "suppressed_small_or_empty_group"}
            )
    report["strata"] = strata
    return rows, report
