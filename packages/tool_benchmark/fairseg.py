"""06E-owner extension: bounded FairSeg development truth, never a patient adapter.

The original PAPILA/REFUGE and 07A NPZ contracts remain unchanged. Model projection
materializes only slo_fundus; truth and demographic fields stay on the evaluator side.
"""

import ast
import io
import math
import struct
import zipfile

from packages.dataset_adapters.reading import grayscale_image

from .models import BenchmarkError
from .reader import bound_read

KEYS = {
    "slo_fundus",
    "disc_cup_mask",
    "age",
    "gender",
    "race",
    "ethnicity",
    "language",
    "maritalstatus",
}
LIMIT = 8388608
PIXELS = 1048576


def array(raw):
    """Small numeric NPY subset; objects, pickle and arbitrary strides are forbidden."""
    if raw[:8] not in (b"\x93NUMPY\x01\x00", b"\x93NUMPY\x02\x00"):
        raise BenchmarkError("fairseg_npy_header")
    width = 2 if raw[6] == 1 else 4
    length = int.from_bytes(raw[8 : 8 + width], "little")
    if not 1 <= length <= 4096:
        raise BenchmarkError("fairseg_npy_header")
    try:
        header = ast.literal_eval(raw[8 + width : 8 + width + length].decode("latin1"))
        shape, dtype, order = header["shape"], header["descr"], header["fortran_order"]
        if set(header) != {"shape", "descr", "fortran_order"} or type(order) is not bool:
            raise ValueError
        if type(shape) is not tuple or len(shape) not in (0, 2):
            raise ValueError
        if any(type(n) is not int or n < 1 for n in shape) or math.prod(shape) > PIXELS:
            raise ValueError
        size = {"|u1": 1, "<f8": 8, "<i8": 8}[dtype]
        data = raw[8 + width + length :]
        if len(data) != math.prod(shape) * size:
            raise ValueError
    except (ValueError, KeyError, TypeError, SyntaxError):
        raise BenchmarkError("fairseg_npy_schema") from None
    return shape, dtype, order, data


def members(raw, selected):
    if len(raw) > 16777216 or not set(selected) <= KEYS:
        raise BenchmarkError("fairseg_archive_bounds")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            items = archive.infolist()
            if len(items) != len(KEYS) or {i.filename for i in items} != {k + ".npy" for k in KEYS}:
                raise BenchmarkError("fairseg_archive_members")
            if sum(i.file_size for i in items) > 16777216 or any(
                i.flag_bits & 1 or i.file_size > LIMIT + 4096 or i.compress_type not in (0, 8)
                for i in items
            ):
                raise BenchmarkError("fairseg_archive_bounds")
            result = {}
            for key in selected:
                with archive.open(key + ".npy") as f:
                    result[key] = array(f.read(LIMIT + 4097))
            return result
    except (zipfile.BadZipFile, EOFError, RuntimeError):
        raise BenchmarkError("fairseg_archive_corrupt") from None


def decode_labels(value):
    shape, dtype, order, data = value
    if len(shape) != 2 or dtype != "<f8":
        raise BenchmarkError("fairseg_truth_schema")
    mapped = bytearray()
    for (v,) in struct.iter_unpack("<d", data):
        if v not in (0.0, -1.0, -2.0):
            raise BenchmarkError("fairseg_truth_labels")
        mapped.append(int(-v))
    h, w = shape
    pixels = (
        bytes(mapped[x * h + y] for y in range(h) for x in range(w)) if order else bytes(mapped)
    )
    return shape, pixels


class FairSegDevelopmentReader:
    """Explicit source-bound samples only; reserved/final evaluation is not implemented."""

    def __init__(self, root, samples, *, purpose):
        if purpose != "development_reanalysis":
            raise BenchmarkError("formal_evaluation_locked")
        self.root = root
        self.samples = dict(samples)

    def _read(self, sample_id):
        if sample_id not in self.samples:
            raise BenchmarkError("sample_not_registered")
        return bound_read(self.root, self.samples[sample_id])

    def model_input(self, sample_id):
        shape, dtype, order, data = members(self._read(sample_id), ["slo_fundus"])["slo_fundus"]
        if len(shape) != 2 or dtype != "|u1" or order:
            raise BenchmarkError("fairseg_image_schema")
        return shape, grayscale_image(shape, data)

    def truth(self, sample_id):
        return decode_labels(members(self._read(sample_id), ["disc_cup_mask"])["disc_cup_mask"])

    def strata(self, sample_id):
        values = members(self._read(sample_id), ["gender", "race"])
        result = {}
        for key, (shape, dtype, _, data) in values.items():
            if shape != () or dtype != "<i8":
                raise BenchmarkError("fairseg_stratum_schema")
            code = struct.unpack("<q", data)[0]
            allowed = {"gender": {-1, 0, 1}, "race": {-1, 0, 1, 2}}[key]
            if code not in allowed:
                raise BenchmarkError("fairseg_stratum_code")
            result[key] = code
        return result
