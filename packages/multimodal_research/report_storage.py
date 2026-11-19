"""Lossless, bounded report chunks in one original Runtime artifact transaction."""

import base64
import json

from packages.grounding.models import canonical, digest
from packages.object_store import checksum

VERSION = "multimodal-report-storage/1.0.0"
CHUNK = 1024 * 1024
MAX_BYTES = 64 * 1024 * 1024


def save(documents, report, page, *, key):
    """Keep existing physical bytes on an exact logical replay, including legacy layouts."""
    desired = (report, dict(html=page, report_checksum=digest(report)))
    with documents.session():
        existing = documents.read(key)
        if existing:
            if unpack(existing, key=key) != desired:
                raise ValueError("multimodal_report_content_conflict")
        else:
            documents.write(key, pack(report, page, key=key))
    if unpack(documents.read(key), key=key) != desired:
        raise ValueError("multimodal_report_readback")


def pack(report, page, *, key):
    values = (report, dict(html=page, report_checksum=digest(report)))
    payloads = [canonical(v) for v in values]
    if sum(map(len, payloads)) > MAX_BYTES:
        raise ValueError("multimodal_report_total_bound")
    # Leave room for the original Runtime envelope, retaining small legacy bytes.
    if all(len(p) <= 3 * 1024 * 1024 for p in payloads):
        return values
    chunks, parts = [], []
    for payload in payloads:
        indices = []
        for offset in range(0, len(payload), CHUNK):
            raw = payload[offset : offset + CHUNK]
            index = len(chunks) + 1
            indices.append(index)
            chunks.append(
                dict(
                    index=index, checksum=checksum(raw), data=base64.b64encode(raw).decode("ascii")
                )
            )
        parts.append(dict(indices=indices, bytes=len(payload), checksum=checksum(payload)))
    return (
        dict(
            schema_version=VERSION,
            multimodal_profile="multimodal-task-report/1.0.0",
            report_key=key,
            parts=parts,
            report_checksum=digest(report),
        ),
        *chunks,
    )


def unpack(values, *, key):
    if not values:
        raise ValueError("multimodal_report_missing")
    head = values[0]
    if head.get("schema_version") != VERSION:
        if len(values) != 2 or values[1].get("report_checksum") != digest(head):
            raise ValueError("multimodal_report_legacy_binding")
        return tuple(values)
    if head.get("report_key") != key or len(head.get("parts", [])) != 2:
        raise ValueError("multimodal_report_storage_binding")
    if len(values) > 131 or sum(p["bytes"] for p in head["parts"]) > MAX_BYTES:
        raise ValueError("multimodal_report_total_bound")
    cursor, result = 1, []
    for part in head["parts"]:
        pieces = []
        if not part["indices"] or part["bytes"] <= 0:
            raise ValueError("multimodal_report_chunk_order")
        for index in part["indices"]:
            if index != cursor or cursor >= len(values):
                raise ValueError("multimodal_report_chunk_order")
            value = values[cursor]
            if value.get("index") != cursor or len(value["data"]) > 4 * ((CHUNK + 2) // 3):
                raise ValueError("multimodal_report_chunk_bound")
            raw = base64.b64decode(value["data"], validate=True)
            if not raw or len(raw) > CHUNK or checksum(raw) != value["checksum"]:
                raise ValueError("multimodal_report_chunk_checksum")
            pieces.append(raw)
            cursor += 1
        payload = b"".join(pieces)
        if len(payload) != part["bytes"] or checksum(payload) != part["checksum"]:
            raise ValueError("multimodal_report_payload_checksum")
        result.append(json.loads(payload))
    if (
        cursor != len(values)
        or result[1].get("report_checksum") != digest(result[0])
        or head["report_checksum"] != digest(result[0])
    ):
        raise ValueError("multimodal_report_storage_checksum")
    return tuple(result)
