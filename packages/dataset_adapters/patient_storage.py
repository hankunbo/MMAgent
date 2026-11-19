"""Append-only packet files and bounded verification; no DB/provider imports."""

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from packages.object_store import checksum
from packages.tool_execution.process import read_file

from .multimodal.formats import source
from .patient_models import PatientPacket
from .reading import AdapterError

DOCUMENTS = ("case-input.json", "provenance.json", "restricted-references.json", "validation.json")


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _write(path, raw):
    with path.open("xb") as f:
        f.write(raw)
    path.chmod(0o600)


def persist(prepared, output, *, source_root):
    # Verify sources once more after all eyes/visits have been decoded, before
    # publishing a completion marker. No old plans or input objects are modified.
    for s in prepared.provenance["sources"].values():
        _, binding = source(source_root, s["path"])
        if binding.model_dump(mode="json") != s:
            raise AdapterError("patient_packet_source_changed_before_save")
    out = Path(output).absolute()
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise AdapterError("patient_packet_output_symlink")
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    (out / "payloads").mkdir(mode=0o700)
    objects = dict(
        zip(
            DOCUMENTS,
            (
                prepared.packet.model_dump(mode="json"),
                prepared.provenance,
                prepared.restricted,
                prepared.validation,
            ),
            strict=True,
        )
    )
    bindings = {}
    for name, value in objects.items():
        raw = encoded(value)
        _write(out / name, raw)
        bindings[name] = dict(checksum=checksum(raw), byte_size=len(raw))
    for name, raw in prepared.contents.items():
        _write(out / name, raw)
    # A failed/interrupted writer has no complete.json and is never treated as ready.
    marker = dict(
        schema_version="patient-packet-completion/1.0.0",
        created_at=datetime.now(UTC).isoformat(),
        documents=bindings,
        packet_checksum=prepared.packet.checksum,
    )
    _write(out / "complete.json", encoded(marker))
    verify(out)
    return out


def verify(directory, *, source_root=None):
    directory = Path(directory).absolute()
    marker = json.loads(read_file(directory / "complete.json", 65536))
    if (
        set(marker) != {"schema_version", "created_at", "documents", "packet_checksum"}
        or marker["schema_version"] != "patient-packet-completion/1.0.0"
        or set(marker["documents"]) != set(DOCUMENTS)
    ):
        raise AdapterError("patient_packet_completion_contract")
    values = {}
    for name in DOCUMENTS:
        raw = read_file(directory / name, 16777216)
        if marker["documents"][name] != dict(checksum=checksum(raw), byte_size=len(raw)):
            raise AdapterError("patient_packet_document_changed")
        values[name] = json.loads(raw)
    packet = PatientPacket.model_validate_json(json.dumps(values["case-input.json"]))
    prov = values["provenance.json"]
    if packet.checksum != marker["packet_checksum"] or prov["packet_checksum"] != packet.checksum:
        raise AdapterError("patient_packet_manifest_binding")
    for c in (*packet.components, *packet.background):
        if (
            c.provenance_ref not in prov["records"]
            or not set(c.source_refs) <= prov["sources"].keys()
        ):
            raise AdapterError("patient_packet_provenance_binding")
    for p in packet.payloads:
        raw = read_file(directory / p.path, p.byte_size)
        if len(raw) != p.byte_size or checksum(raw) != p.checksum:
            raise AdapterError("patient_packet_payload_changed")
    if source_root is not None:
        for s in prov["sources"].values():
            _, current = source(source_root, s["path"])
            if current.model_dump(mode="json") != s:
                raise AdapterError("patient_packet_current_source_changed")
    return packet


def summary(packet):
    return dict(
        packet_checksum=packet.checksum,
        identity_basis=packet.identity_basis,
        collection_scope=packet.collection_scope,
        components=len(packet.components),
        background_fields=len(packet.background),
        eye_visit_groups=len(packet.visits),
        eyes=sorted({c.laterality for c in packet.components}),
        statuses=dict(Counter(c.status for c in packet.components)),
        parse_statuses=dict(Counter(c.parse_status for c in packet.components)),
        payloads=len(packet.payloads),
        task_scope_selected=False,
        api_requests=0,
        model_runs=0,
        database_writes=0,
    )
