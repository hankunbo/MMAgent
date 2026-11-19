"""One explicitly development-only PAPILA check via the original G2 reader."""

import json
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.evaluation_preparation.case_execution import process
from packages.grounding.models import digest
from packages.multimodal_research.fundus_measurements import measure
from packages.tool_benchmark.metrics import overlap
from packages.tool_benchmark.reader import BenchmarkReader, manifest, papila_sample
from packages.tool_execution.qc import grayscale_png


def run():
    root = Path("/opt/data/test/download/PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f")
    output = Path("docs/acceptance/13-active-acquisition-v1/papila-development")
    output.mkdir(parents=True, exist_ok=True)
    sample = papila_sample(root, "RET002OD", split="development")
    reader = BenchmarkReader({"PAPILA": root}, manifest((sample,)))
    value = reader.model_input("PAPILA", "RET002OD")
    prediction = output / "prediction"
    process(
        [
            "/opt/conda/bin/python",
            "-m",
            "packages.multimodal_research.fundus_worker",
            "--image",
            str(root / value.image.relative_path),
            "--output",
            str(prediction),
        ],
        directory=output / "process",
        seconds=180,
    )
    measurements = measure(prediction)
    write_once(output / "measurements.json", measurements)
    metrics = {}
    for name in ("disc", "cup"):
        (ph, pw), mask = grayscale_png((prediction / (name + ".png")).read_bytes(), 1048576)
        h, w = sample.dimensions
        pixels = bytes(
            int(
                mask[
                    min(ph - 1, int((y + 0.5) * ph / h)) * pw + min(pw - 1, int((x + 0.5) * pw / w))
                ]
                > 0
            )
            for y in range(h)
            for x in range(w)
        )
        metrics[name] = {
            expert: overlap(pixels, truth, (h, w))
            for expert, truth in reader.truth_masks("PAPILA", "RET002OD", "optic_" + name).items()
        }
    result = dict(
        status="evaluated",
        sample="RET002OD",
        sample_count=1,
        patient_count=1,
        development_only=True,
        truth_loaded_after_prediction=True,
        benchmark_checksum=reader.manifest.checksum,
        prediction_checksum=digest(json.loads((prediction / "result.json").read_bytes())),
        metrics=metrics,
        clinical_qualified=False,
        independent_performance_estimate=False,
    )
    write_once(output / "evaluation.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    run()
