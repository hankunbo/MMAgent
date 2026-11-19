"""Deterministic single optic-disc PNG geometry, no clinical interpretation."""

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

VERSION = "disc-geometry/1.0.0"


def decode(data):
    if len(data) > 4194304 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid_png")
    offset = 8
    compressed = bytearray()
    shape = None
    ended = False
    while offset + 12 <= len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        end = offset + 12 + size
        body = data[offset + 8 : end - 4]
        if end > len(data) or zlib.crc32(kind + body) != int.from_bytes(data[end - 4 : end], "big"):
            raise ValueError("invalid_png")
        if kind == b"IHDR" and offset == 8 and size == 13:
            w, h, *fmt = struct.unpack(">IIBBBBB", body)
            if not w or not h or w * h > 16777216 or fmt != [8, 0, 0, 0, 0]:
                raise ValueError("unsupported_png")
            shape = h, w
        elif kind == b"IDAT" and shape:
            compressed.extend(body)
        elif kind == b"IEND" and shape and not body and end == len(data):
            ended = True
        else:
            raise ValueError("unsupported_png")
        offset = end
    if not ended or offset != len(data):
        raise ValueError("invalid_png")
    h, w = shape
    d = zlib.decompressobj()
    raw = d.decompress(compressed, (w + 1) * h + 1)
    if len(raw) != (w + 1) * h or not d.eof or d.unused_data or d.unconsumed_tail:
        raise ValueError("invalid_png")
    prior = bytearray(w)
    count = 0
    x0 = w
    y0 = h
    x1 = y1 = 0
    for y in range(h):
        method = raw[y * (w + 1)]
        row = bytearray(raw[y * (w + 1) + 1 : (y + 1) * (w + 1)])
        if method > 4:
            raise ValueError("invalid_filter")
        for x in range(w):
            a = row[x - 1] if x else 0
            b = prior[x]
            c = prior[x - 1] if x else 0
            if method == 1:
                row[x] = (row[x] + a) % 256
            elif method == 2:
                row[x] = (row[x] + b) % 256
            elif method == 3:
                row[x] = (row[x] + (a + b) // 2) % 256
            elif method == 4:
                p = a + b - c
                row[x] = (row[x] + min((a, b, c), key=lambda v: abs(p - v))) % 256
            if row[x] not in (0, 255):
                raise ValueError("invalid_label")
            if row[x]:
                count += 1
                x0 = min(x0, x)
                y0 = min(y0, y)
                x1 = max(x1, x + 1)
                y1 = max(y1, y + 1)
        prior = row
    if not count:
        raise ValueError("empty_disc")
    return dict(
        schema_version=VERSION,
        input_checksum="sha256:" + hashlib.sha256(data).hexdigest(),
        height=h,
        width=w,
        area_px=count,
        bbox_xyxy=[x0, y0, x1, y1],
        width_px=x1 - x0,
        height_px=y1 - y0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--disc-mask", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with a.disc_mask.open("rb") as f:
        data = f.read(4194305)
    result = decode(data)
    with a.output.open("x") as f:
        json.dump(result, f, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
