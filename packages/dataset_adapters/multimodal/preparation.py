"""Offline preparation exports. These are NOT clinical reports or run receipts."""

import json
from pathlib import Path

from packages.dataset_adapters.reading import AdapterError
from packages.object_store import checksum
from packages.tool_execution.process import read_file

from .models import CaseComponents


def persist(sample, output):
    sample = sample.recheck()
    out = Path(output).absolute()
    # Never overwrite an old result or resolve an output symlink to another directory.
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise AdapterError("multimodal_output_symlink")
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    payload_dir = out / "payloads"
    payload_dir.mkdir(mode=0o700)
    files = []
    for id, data in sample.payload_bytes:
        name = checksum(id.encode())[7:] + ".bin"
        path = payload_dir / name
        with path.open("xb") as stream:
            stream.write(data)
        path.chmod(0o600)
        files.append(
            dict(id=id, path="payloads/" + name, checksum=checksum(data), byte_size=len(data))
        )
    envelope = dict(
        schema_version="multimodal-preparation-export/1.0.0",
        manifest=sample.manifest.model_dump(mode="json"),
        payload_files=files,
        model_runs=0,
        llm_requests=0,
        database_writes=0,
        inference_ready=False,
        score=None,
    )
    # Completion marker written last; an interrupted export isn't falsely complete.
    with (out / "preparation.json").open("x") as stream:
        json.dump(envelope, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return out / "preparation.json"


def verify(path):
    p = Path(path).absolute()
    raw = read_file(p, 4194304)
    value = json.loads(raw)
    if type(value) is not dict or set(value) != {
        "schema_version",
        "manifest",
        "payload_files",
        "model_runs",
        "llm_requests",
        "database_writes",
        "inference_ready",
        "score",
    }:
        raise AdapterError("multimodal_export_fields")
    if value.get("schema_version") != "multimodal-preparation-export/1.0.0":
        raise AdapterError("multimodal_export_version")
    if (
        value.get("inference_ready") is not False
        or value.get("score") is not None
        or any(value.get(k) != 0 for k in ("model_runs", "llm_requests", "database_writes"))
    ):
        raise AdapterError("multimodal_export_scope")
    manifest = CaseComponents.model_validate_json(json.dumps(value["manifest"]))
    files = value["payload_files"]
    if len({f["id"] for f in files}) != len(files):
        raise AdapterError("multimodal_export_duplicate_payload")
    expected = {v.id: v for v in manifest.payloads}
    if {f["id"] for f in files} != set(expected):
        raise AdapterError("multimodal_export_payload_set")
    for f in files:
        rel = "payloads/" + checksum(f["id"].encode())[7:] + ".bin"
        owner = expected[f["id"]]
        if f["path"] != rel or f["checksum"] != owner.checksum or f["byte_size"] != owner.byte_size:
            raise AdapterError("multimodal_export_payload_binding")
        data = read_file(p.parent / rel, owner.byte_size)
        if len(data) != owner.byte_size or checksum(data) != owner.checksum:
            raise AdapterError("multimodal_export_payload_changed")
    return manifest


def summary(manifest):
    from collections import Counter

    return dict(
        dataset=manifest.dataset,
        sample=manifest.sample,
        checksum=manifest.checksum,
        task=manifest.task,
        components=len(manifest.components),
        modalities=sorted({c.modality for c in manifest.components}),
        statuses=dict(Counter(c.status for c in manifest.components)),
        payloads=len(manifest.payloads),
        inference_ready=False,
        scope="input_preparation_only_not_case_inference",
        llm_requests=0,
    )
