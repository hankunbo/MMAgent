"""Bounded data parsing. No numpy import, pickle execution, archive extraction or logging."""

import ast
import csv
import io
import math
import re
import struct
import zipfile
import zlib
from pathlib import Path

from packages.object_store import checksum
from packages.tool_execution.process import read_file

from .models import FileBinding


class AdapterError(ValueError):
    """Only constant reason codes may cross the data boundary."""


def read_source(root, relative, limit=4194304):
    try:
        p = Path(relative)
        if p.is_absolute() or ".." in p.parts or "\\" in relative:
            raise ValueError
        raw = read_file(Path(root) / p, limit)
        return raw, FileBinding(relative_path=relative, checksum=checksum(raw), byte_size=len(raw))
    except Exception:
        raise AdapterError("source_missing_unsafe_changed_or_oversized") from None


def csv_rows(root, relative, fields, *, required_fields=None):
    raw, binding = read_source(root, relative, 33554432)
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), strict=True)
        names = reader.fieldnames or ()
        if required_fields is None:
            if set(names) != set(fields) or len(names) != len(fields):
                raise ValueError
        elif (
            not set(required_fields) <= set(names)
            or not 1 <= len(names) <= 64
            or len(set(names)) != len(names)
            or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,99}", n) for n in names)
        ):
            raise ValueError
        rows = []
        for row in reader:
            if len(rows) >= 20000 or None in row or any(v is None for v in row.values()):
                raise ValueError
            rows.append(row)
        return rows, binding
    except Exception:
        raise AdapterError("csv_schema_or_bounds_invalid") from None


def basename(value, suffix):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}" + re.escape(suffix), value):
        raise AdapterError("filename_invalid")
    return value


def npy(raw):
    if len(raw) < 10 or raw[:6] != b"\x93NUMPY" or raw[6:8] not in (b"\x01\x00", b"\x02\x00"):
        raise ValueError
    width = 2 if raw[6] == 1 else 4
    n = int.from_bytes(raw[8 : 8 + width], "little")
    if not 1 <= n <= 4096:
        raise ValueError
    h = ast.literal_eval(raw[8 + width : 8 + width + n].decode("latin1"))
    if set(h) != {"descr", "fortran_order", "shape"} or h["fortran_order"] is not False:
        raise ValueError
    shape = h["shape"]
    dtype = h["descr"]
    if (
        type(shape) is not tuple
        or len(shape) > 2
        or any(type(d) is not int or d <= 0 for d in shape)
    ):
        raise ValueError
    match = re.fullmatch(r"([<|])([uiU])([1-9][0-9]{0,2})", dtype)
    if not match:
        raise ValueError  # object/pickle, structured and arbitrary dtypes are never loaded
    size = int(match[3]) * (4 if match[2] == "U" else 1)
    count = math.prod(shape)
    if count * size > 2097152:
        raise ValueError
    data = raw[8 + width + n :]
    if len(data) != count * size:
        raise ValueError
    return shape, dtype, data


NPZ_KEYS = {
    "slo_fundus",
    "maritalstatus",
    "ethnicity",
    "language",
    "gender",
    "race",
    "age",
    "glaucoma",
}


def read_npz(raw, *, runtime_profile=False):
    try:
        if len(raw) > 4194304:
            raise ValueError
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            members = z.infolist()
            names = [i.filename for i in members]
            if runtime_profile:
                if (
                    not 1 <= len(names) <= 64
                    or len(set(names)) != len(names)
                    or "slo_fundus.npy" not in names
                    or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,99}\.npy", n) for n in names)
                ):
                    raise ValueError
            elif len(members) != len(NPZ_KEYS) or set(names) != {k + ".npy" for k in NPZ_KEYS}:
                raise ValueError
            if sum(i.file_size for i in members) > 4194304:
                raise ValueError
            arrays = {}
            metadata = []
            for i in members:
                if i.flag_bits & 1 or i.file_size > 2097152 or i.compress_type not in (0, 8):
                    raise ValueError
                with z.open(i) as f:
                    member = f.read(2097153)
                shape, dtype, data = npy(member)
                key = i.filename[:-4]
                if key == "slo_fundus":
                    if dtype != "|u1" or len(shape) != 2 or math.prod(shape) > 1048576:
                        raise ValueError
                    arrays[key] = (shape, data)
                elif key not in NPZ_KEYS and runtime_profile:
                    # Validate framing, dtype, length, CRC and bounds, but never expose values.
                    pass
                elif shape != ():
                    raise ValueError
                elif key == "age":
                    if not dtype.startswith("<U"):
                        raise ValueError
                    arrays[key] = data.decode("utf-32-le").rstrip("\x00")
                else:
                    if dtype != "<i8":
                        raise ValueError
                    arrays[key] = struct.unpack("<q", data)[0]
                metadata.append(
                    dict(
                        key=key,
                        shape=list(shape),
                        dtype=dtype,
                        compressed_bytes=i.compress_size,
                        expanded_bytes=i.file_size,
                    )
                )
            return arrays, metadata
    except Exception:
        raise AdapterError("npz_corrupt_unsafe_or_schema_invalid") from None


def grayscale_image(shape, pixels):
    """Lossless uint8 PNG projection; no sampling or modality/channel conversion."""
    height, width = shape

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    scan = b"".join(b"\x00" + pixels[y * width : (y + 1) * width] for y in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scan))
        + chunk(b"IEND", b"")
    )
