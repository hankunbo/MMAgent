"""Managed objects and bounded format headers; no note text or image/model decoding."""

import ast
import re
import struct
import zlib

from packages.grounding.compatibility import artifact_compatibility
from packages.grounding.models import digest, validate
from packages.object_store import checksum

from .models import AdmissionError, InputRecord, VerifiedInput


def image_header(data, limit):
    header = data[:limit]
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(header) < 33 or header[8:16] != b"\x00\x00\x00\rIHDR":
            raise AdmissionError("format_invalid")
        if zlib.crc32(header[12:29]) != struct.unpack(">I", header[29:33])[0]:
            raise AdmissionError("format_invalid")
        width, height, bits, color, compression, filtering, interlace = struct.unpack(
            ">IIBBBBB", header[16:29]
        )
        bit_depths = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
        if (
            not width
            or not height
            or bits not in bit_depths.get(color, ())
            or compression
            or filtering
            or interlace not in (0, 1)
        ):
            raise AdmissionError("format_invalid")
        return "image/png", (height, width)
    if header.startswith(b"\xff\xd8"):
        offset = 2
        while offset + 4 <= len(header):
            if header[offset] != 255:
                raise AdmissionError("format_invalid")
            while offset < len(header) and header[offset] == 255:
                offset += 1
            if offset >= len(header):
                break
            marker = header[offset]
            offset += 1
            if marker in (0xD9, 0xDA):
                break
            size = int.from_bytes(header[offset : offset + 2], "big")
            if size < 2 or offset + size > len(header):
                break
            if marker in {0xC0, 0xC1, 0xC2} and size >= 8:
                height, width = struct.unpack(">HH", header[offset + 3 : offset + 7])
                if height and width:
                    return "image/jpeg", (height, width)
            offset += size
        raise AdmissionError("header_unresolved")
    if header.startswith(b"\x93NUMPY"):
        version = header[6:8]
        if version not in (b"\x01\x00", b"\x02\x00"):
            raise AdmissionError("format_invalid")
        start = 10 if version == b"\x01\x00" else 12
        size = int.from_bytes(header[8:start], "little")
        if start + size > len(header):
            raise AdmissionError("header_unresolved")
        try:
            fields = ast.literal_eval(header[start : start + size].decode("ascii"))
        except (ValueError, SyntaxError, UnicodeError, RecursionError):
            raise AdmissionError("format_invalid") from None
        if type(fields) is not dict or set(fields) != {"descr", "fortran_order", "shape"}:
            raise AdmissionError("format_invalid")
        shape, dtype = fields["shape"], fields["descr"]
        if (
            type(shape) is not tuple
            or not 1 <= len(shape) <= 5
            or any(type(n) is not int or not 1 <= n <= 1000000 for n in shape)
            or type(fields["fortran_order"]) is not bool
            or type(dtype) is not str
            or not re.fullmatch(r"[<>|=][biuf][1248]", dtype)
        ):
            raise AdmissionError("format_invalid")
        count = int(dtype[-1])
        for n in shape:
            count *= n
        if start + size + count != len(data):
            raise AdmissionError("size_mismatch")
        return "application/x-npy", shape
    raise AdmissionError("format_invalid")


def check_input(
    reader,
    objects,
    request,
    code,
    constraint,
    rule,
    policy,
    budget,
    *,
    limited_metadata=False,
    metadata=None,
):
    try:
        raw = reader.read_input(request.scope, request.lock.run_id, code)
    except KeyError:
        raise AdmissionError("patient_data_missing") from None
    record = validate(InputRecord, raw)
    expected = next((a for a in request.artifacts if a.asset_code == code), None)
    if record.metadata != expected:
        raise AdmissionError("input_record_mismatch")
    reasons = artifact_compatibility(
        record.metadata,
        constraint,
        request,
        laterality_required=metadata.laterality_required if metadata else not limited_metadata,
        acquisition_time_required=(
            metadata.acquisition_time_required if metadata else not limited_metadata
        ),
    )
    if reasons:
        raise AdmissionError(reasons[0])
    a = record.artifact
    if record.content_kind not in {"image", "array"}:
        raise AdmissionError("input_content_forbidden")
    if not re.fullmatch(r"object://knowledge/objects/[0-9a-f]{32}", a.uri):
        raise AdmissionError("input_uri_rejected")
    if record.byte_size > policy.max_input_bytes:
        raise AdmissionError("input_byte_limit")
    if not set(rule.anatomical_targets) <= set(record.metadata.scope_terms) & set(
        request.satisfied_terms
    ):
        raise AdmissionError("anatomical_target_mismatch")
    if a.media_type not in rule.media_types:
        raise AdmissionError("format_mismatch")
    budget.consume(record.byte_size)
    data = objects.read_bounded(a.uri, policy.max_input_bytes)
    if len(data) != record.byte_size:
        raise AdmissionError("size_mismatch")
    if checksum(data) != a.checksum:
        raise AdmissionError("checksum_mismatch")
    media, dimensions = image_header(data, policy.header_bytes)
    if (
        media != a.media_type
        or media not in rule.media_types
        or ((media == "application/x-npy") != (record.content_kind == "array"))
    ):
        raise AdmissionError("format_mismatch")
    if (
        dimensions != a.dimensions
        or len(dimensions) != len(rule.dimensions_min)
        or any(
            not low <= n <= high
            for n, low, high in zip(
                dimensions, rule.dimensions_min, rule.dimensions_max, strict=True
            )
        )
    ):
        raise AdmissionError("dimensions_mismatch")
    return VerifiedInput(
        asset_code=code,
        uri=a.uri,
        checksum=a.checksum,
        byte_size=len(data),
        record_checksum=digest(record),
        media_type=media,
        dimensions=dimensions,
    )
