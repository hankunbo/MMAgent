"""Bounded NPZ/XLSX readers. No numpy/pickle, labels-to-prompt or package installation."""

import ast
import errno
import io
import math
import re
import struct
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock

from defusedxml import ElementTree as ET

from packages.dataset_adapters.reading import AdapterError
from packages.object_store import checksum
from packages.tool_execution.process import read_file

from .models import SourceFile

MAX_FILE = 134217728
MAX_MEMBER = 67108864


def source(root, relative):
    # Validate all parents with the original no-symlink bounded reader.
    probe = SourceFile(path=relative, checksum="sha256:" + "0" * 64, byte_size=1)
    try:
        raw = read_file(Path(root).absolute() / probe.path, MAX_FILE)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise AdapterError("multimodal_source_unsafe_path") from None
        if error.errno == errno.ENOENT:
            raise AdapterError("multimodal_source_not_found") from None
        raise AdapterError("multimodal_source_io_error") from None
    return raw, SourceFile(path=relative, checksum=checksum(raw), byte_size=len(raw))


def archive(raw, *, limit=MAX_FILE, count=1024):
    if len(raw) > limit:
        raise AdapterError("multimodal_archive_size")
    z = zipfile.ZipFile(io.BytesIO(raw))
    items = z.infolist()
    if (
        not 1 <= len(items) <= count
        or len({i.filename for i in items}) != len(items)
        or sum(i.file_size for i in items) > limit
        or any(i.flag_bits & 1 or i.compress_type not in (0, 8) for i in items)
        or any(i.file_size > MAX_MEMBER for i in items)
    ):
        z.close()
        raise AdapterError("multimodal_archive_bounds")
    return z


# Pure computation memo: bounded metadata only, never payloads, file paths,
# source qualification, patient identity or execution permissions. Every lookup
# hashes the current immutable payload. npy and audit_component still validate
# current framing, shape, dtype and the input manifest before this method runs.
_NUMERIC_COUNTS_LIMIT = 128
_numeric_counts = OrderedDict()
_numeric_counts_lock = RLock()


def _count_finite(values):
    total = finite = 0
    for value in values:
        total += 1
        if value is not None and math.isfinite(value):
            finite += 1
    return finite, total


@dataclass(frozen=True, repr=False)
class Array:
    raw: bytes
    data: bytes
    shape: tuple[int, ...]
    dtype: str

    def numbers(self):
        codes = {"f4": "f", "f8": "d", "i4": "i", "i8": "q", "u1": "B"}
        if self.dtype[1:] not in codes:
            raise AdapterError("multimodal_numeric_dtype")
        endian = ">" if self.dtype[0] == ">" else "<"
        return (v[0] for v in struct.iter_unpack(endian + codes[self.dtype[1:]], self.data))

    def finite_counts(self):
        """Exact numeric availability; no clinical inference or current-use grant.

        Failures are not memoized. Cache size is bounded independently of input
        size and returned values are immutable; the existing 64 MiB parser limit
        remains in npy. Other Array implementations take the uncached path.
        """
        values = self.numbers()  # Validate supported numeric dtype even on a hit.
        if type(self) is not Array or type(self.data) is not bytes:
            return _count_finite(values)
        key = (self.dtype, len(self.data), checksum(self.data))
        with _numeric_counts_lock:
            cached = _numeric_counts.get(key)
            if cached is not None:
                _numeric_counts.move_to_end(key)
                return cached
        result = _count_finite(values)
        with _numeric_counts_lock:
            _numeric_counts[key] = result
            _numeric_counts.move_to_end(key)
            while len(_numeric_counts) > _NUMERIC_COUNTS_LIMIT:
                _numeric_counts.popitem(last=False)
        return result

    def scalar(self):
        if self.shape != ():
            raise AdapterError("multimodal_not_scalar")
        value = float(next(self.numbers()))
        return value if math.isfinite(value) else None


def npy(raw):
    if len(raw) < 10 or raw[:6] != b"\x93NUMPY" or raw[6:8] not in (b"\x01\x00", b"\x02\x00"):
        raise AdapterError("multimodal_npy_header")
    width = 2 if raw[6] == 1 else 4
    size = int.from_bytes(raw[8 : 8 + width], "little")
    if not 1 <= size <= 4096:
        raise AdapterError("multimodal_npy_header_size")
    header = ast.literal_eval(raw[8 + width : 8 + width + size].decode("latin1"))
    if (
        type(header) is not dict
        or set(header) != {"descr", "fortran_order", "shape"}
        or header["fortran_order"] is not False
        or type(header["shape"]) is not tuple
        or len(header["shape"]) > 4
        or any(type(n) is not int or not 0 <= n <= 100000 for n in header["shape"])
        or type(header["descr"]) is not str
    ):
        raise AdapterError("multimodal_npy_layout")
    match = re.fullmatch(r"([<>|])([fiuUS])([1-9][0-9]{0,2})", header["descr"])
    if not match:
        raise AdapterError("multimodal_npy_dtype")
    kind, width = match[2], int(match[3])
    if (kind in "fi" and width not in (4, 8)) or (kind == "u" and width != 1):
        raise AdapterError("multimodal_npy_dtype")
    length = math.prod(header["shape"]) * width * (4 if kind == "U" else 1)
    payload = raw[8 + (2 if raw[6] == 1 else 4) + size :]
    if length > MAX_MEMBER or len(payload) != length:
        raise AdapterError("multimodal_npy_size")
    return Array(raw, payload, header["shape"], header["descr"])


def npz(raw, visible_keys):
    arrays, names = {}, []
    try:
        with archive(raw, count=64) as z:
            for item in z.infolist():
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,99}\.npy", item.filename):
                    raise AdapterError("multimodal_npz_member")
                name = item.filename[:-4]
                # Framing/dtype/CRC checked even for withheld members; values are never decoded.
                value = npy(z.read(item))
                names.append(name)
                if name in visible_keys:
                    arrays[name] = value
    except (zipfile.BadZipFile, SyntaxError, UnicodeError, EOFError, struct.error):
        raise AdapterError("multimodal_npz_corrupt") from None
    return arrays, tuple(sorted(names))


NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _decode_sheets(raw):
    """Yield numbered rows/cells. Formula/external-reference workbooks are not interpreted."""
    with archive(raw, limit=33554432) as z:
        if any("externalLinks/" in n or "vbaProject" in n for n in z.namelist()):
            raise AdapterError("multimodal_xlsx_external_content")
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = [
                "".join(t.text or "" for t in s.iter(NS + "t"))
                for s in ET.fromstring(z.read("xl/sharedStrings.xml"))
            ]
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        targets = {}
        for rel in rels:
            if rel.attrib.get("TargetMode") == "External":
                raise AdapterError("multimodal_xlsx_external_content")
            target = rel.attrib["Target"]
            path = target.lstrip("/") if target.startswith("/") else "xl/" + target
            if ".." in Path(path).parts or "\\" in path:
                raise AdapterError("multimodal_xlsx_target")
            if rel.attrib["Id"] in targets:
                raise AdapterError("multimodal_xlsx_duplicate_relationship")
            targets[rel.attrib["Id"]] = path
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        sheet_names = set()
        for sheet in workbook.find(NS + "sheets"):
            if sheet.attrib["name"] in sheet_names:
                raise AdapterError("multimodal_xlsx_duplicate_sheet")
            sheet_names.add(sheet.attrib["name"])
            rid = sheet.attrib[
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            ]
            tree = ET.fromstring(z.read(targets[rid]))
            rows = []
            for row in tree.findall(f"{NS}sheetData/{NS}row"):
                number = int(row.attrib["r"])
                if not 1 <= number <= 20000:
                    raise AdapterError("multimodal_xlsx_row_limit")
                cells = {}
                for cell in row:
                    if cell.find(NS + "f") is not None:
                        raise AdapterError("multimodal_xlsx_formula")
                    match = re.fullmatch(r"([A-Z]{1,3})([1-9][0-9]*)", cell.attrib["r"])
                    if not match or int(match[2]) != number or match[1] in cells:
                        raise AdapterError("multimodal_xlsx_cell_binding")
                    value = cell.find(NS + "v")
                    text = value.text or "" if value is not None else ""
                    if cell.attrib.get("t") == "s":
                        index = int(text)
                        if not 0 <= index < len(strings):
                            raise AdapterError("multimodal_xlsx_string_index")
                        text = strings[index]
                    elif cell.attrib.get("t") == "inlineStr":
                        text = "".join(t.text or "" for t in cell.iter(NS + "t"))
                    cells[match[1]] = text
                rows.append((number, cells))
            if len({n for n, _ in rows}) != len(rows):
                raise AdapterError("multimodal_xlsx_duplicate_row")
            yield sheet.attrib["name"], rows


@lru_cache(maxsize=1)
def _sheet_values(raw):
    """Pure bytes-to-values memo: one <=1 MiB workbook, original archive bounds.

    This stores no file path, source status, patient permission or input proof.
    The original source owner still opens and checks current bytes on every read.
    Only immutable tuples are retained; callers receive detached row dictionaries.
    Errors are not cached. Replacing even one byte requires a new full decode.
    """
    return tuple(
        (name, tuple((number, tuple(cells.items())) for number, cells in rows))
        for name, rows in _decode_sheets(raw)
    )


def sheets(raw):
    """Same validated table values; parsing reuse never substitutes source recheck."""
    if type(raw) is not bytes or len(raw) > 1048576:
        yield from _decode_sheets(raw)
        return
    for name, rows in _sheet_values(raw):
        yield name, [(number, dict(cells)) for number, cells in rows]
