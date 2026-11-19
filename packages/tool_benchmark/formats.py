"""Bounded dataset metadata and explicit G2 conversion; not model output QC."""

import math
import struct
import zlib

from .models import BenchmarkError

MAX_PIXELS = 16777216


def jpeg_dimensions(data):
    """Metadata only: this does not certify full JPEG decoding or technical QC."""
    if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise BenchmarkError("jpeg_header_invalid")
    i = 2
    while i + 4 <= min(len(data), 65536):
        if data[i] != 255:
            break
        while i < len(data) and data[i] == 255:
            i += 1
        if i >= len(data):
            break
        marker = data[i]
        i += 1
        if marker in (0xDA, 0xD9):
            break
        size = int.from_bytes(data[i : i + 2], "big")
        if size < 2 or i + size > len(data):
            break
        if marker in (0xC0, 0xC1, 0xC2):
            if size < 8 or data[i + 2] != 8:
                break
            h, w = struct.unpack(">HH", data[i + 3 : i + 7])
            if h and w and h * w <= MAX_PIXELS:
                return (h, w)
            raise BenchmarkError("image_pixel_limit")
        i += size
    raise BenchmarkError("jpeg_header_invalid")


def contour_mask(data, shape):
    """x/y float coordinates; integer pixel centres; even/odd fill, boundary included.

    Close last-to-first, no rounding/resizing/clipping. Reject nonfinite/out-of-image
    coordinates and zero-area contours. This versioned rule is independent of skimage.
    """
    h, w = shape
    if min(h, w) < 1 or h * w > MAX_PIXELS or len(data) > 1048576:
        raise BenchmarkError("contour_size_limit")
    try:
        points = [
            tuple(map(float, line.split()))
            for line in data.decode("ascii").splitlines()
            if line.strip()
        ]
    except (ValueError, UnicodeError):
        raise BenchmarkError("contour_invalid") from None
    if not 3 <= len(points) <= 4096 or any(
        len(p) != 2 or not all(math.isfinite(v) for v in p) for p in points
    ):
        raise BenchmarkError("contour_invalid")
    if any(not (0 <= x <= w - 1 and 0 <= y <= h - 1) for x, y in points):
        raise BenchmarkError("contour_outside_image")
    if points[-1] == points[0]:
        points.pop()
    edges = list(zip(points, points[1:] + points[:1], strict=True))
    area = abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in edges))
    if area == 0:
        raise BenchmarkError("contour_zero_area")
    mask = bytearray(h * w)
    for y in range(math.ceil(min(p[1] for p in points)), math.floor(max(p[1] for p in points)) + 1):
        crossings = []
        for (x1, y1), (x2, y2) in edges:
            if (y1 > y) != (y2 > y):
                crossings.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
            # Explicit boundary inclusion at integer pixel centres.
            if y1 == y2 == y:
                a, b = math.ceil(min(x1, x2)), math.floor(max(x1, x2))
                mask[y * w + a : y * w + b + 1] = b"\x01" * (b - a + 1)
            elif min(y1, y2) <= y <= max(y1, y2) and y1 != y2:
                x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x.is_integer():
                    mask[y * w + int(x)] = 1
        crossings.sort()
        if len(crossings) % 2:
            raise BenchmarkError("contour_odd_crossings")
        for left, right in zip(crossings[::2], crossings[1::2], strict=True):
            a, b = math.ceil(left), math.floor(right)
            mask[y * w + a : y * w + b + 1] = b"\x01" * max(0, b - a + 1)
    if not any(mask):
        raise BenchmarkError("contour_empty")
    return bytes(mask)


def grayscale_bmp(data):
    """Only uncompressed Windows BITMAPINFOHEADER, 8-bit gray palette/24-bit gray."""
    if len(data) < 54 or len(data) > 16777216 or data[:2] != b"BM":
        raise BenchmarkError("bmp_invalid")
    size, offset = struct.unpack_from("<I4xI", data, 2)
    dib, w, signed_h, planes, bits, comp = struct.unpack_from("<IiiHHI", data, 14)
    h = abs(signed_h)
    if (dib, planes, comp) != (40, 1, 0) or bits not in (8, 24):
        raise BenchmarkError("bmp_unsupported")
    if w < 1 or h < 1 or w * h > MAX_PIXELS or size != len(data) or offset < 54:
        raise BenchmarkError("bmp_invalid")
    stride = ((w * bits + 31) // 32) * 4
    if offset + stride * h != len(data):
        raise BenchmarkError("bmp_invalid")
    palette = []
    if bits == 8:
        count = struct.unpack_from("<I", data, 46)[0] or 256
        if not 1 <= count <= 256 or offset != 54 + count * 4:
            raise BenchmarkError("bmp_palette_invalid")
        for i in range(count):
            b, g, r, _ = data[54 + 4 * i : 58 + 4 * i]
            if b != g or g != r:
                raise BenchmarkError("bmp_not_gray")
            palette.append(r)
    elif offset != 54:
        raise BenchmarkError("bmp_unsupported")
    pixels = bytearray()
    for y in range(h):
        row_y = h - y - 1 if signed_h > 0 else y
        row = data[offset + row_y * stride : offset + (row_y + 1) * stride]
        for x in range(w):
            if bits == 8:
                if row[x] >= len(palette):
                    raise BenchmarkError("bmp_palette_invalid")
                pixels.append(palette[row[x]])
            else:
                b, g, r = row[x * 3 : x * 3 + 3]
                if b != g or g != r:
                    raise BenchmarkError("bmp_not_gray")
                pixels.append(r)
    return (h, w), bytes(pixels)


def mask_png(shape, pixels):
    """Export a derived binary reference as PNG; never label it model_prediction."""
    h, w = shape
    if min(h, w) < 1 or h * w > MAX_PIXELS or len(pixels) != h * w or not set(pixels) <= {0, 1}:
        raise BenchmarkError("mask_invalid")

    def chunk(kind, body):
        return (
            struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        )

    raw = b"".join(b"\x00" + pixels[y * w : (y + 1) * w] for y in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
