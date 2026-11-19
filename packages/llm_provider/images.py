"""Bounded header/dimension binding, not image QC or clinical interpretation."""

import struct

from packages.grounding.models import validate
from packages.object_store import checksum

from .models import ImageBinding


def validate_image(binding, raw):
    b = validate(ImageBinding, binding)
    if (
        type(raw) is not bytes
        or not 1 <= len(raw) <= 8_388_608
        or checksum(raw) != b.reference.checksum
    ):
        raise ValueError("image_bytes_binding")
    size = None
    if b.media_type == "image/png":
        if len(raw) >= 33 and raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR":
            size = struct.unpack(">II", raw[16:24])
    elif raw[:2] == b"\xff\xd8":
        n = 2
        while n + 4 <= len(raw):
            if raw[n] != 255:
                break
            marker = raw[n + 1]
            n += 2
            if marker in {0xD8, 0xD9}:
                continue
            length = int.from_bytes(raw[n : n + 2], "big")
            if length < 2 or n + length > len(raw):
                break
            if marker in {0xC0, 0xC1, 0xC2} and length >= 8:
                size = (
                    int.from_bytes(raw[n + 5 : n + 7], "big"),
                    int.from_bytes(raw[n + 3 : n + 5], "big"),
                )
                break
            n += length
    if size != (b.width, b.height):
        raise ValueError("image_dimensions_binding")
    return b
