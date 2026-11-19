"""Deterministic grayscale display normalization; no synthesized image contents."""

from packages.dataset_adapters.reading import grayscale_image
from packages.object_store import checksum
from packages.tool_execution.qc import grayscale_png


def contrast_view(raw):
    shape, pixels = grayscale_png(raw, 1048576)
    counts = [0] * 256
    for p in pixels:
        counts[p] += 1
    nonzero = len(pixels) - counts[0]
    if nonzero == 0:
        return raw, dict(
            method="identity_constant_image",
            source=checksum(raw),
            output=checksum(raw),
            pixel_geometry="unchanged",
        )
    n = 0
    high = 255
    for i in range(1, 256):
        n += counts[i]
        if n >= nonzero * 0.995:
            high = i
            break
    result = grayscale_image(shape, bytes(min(255, round(p * 255 / high)) for p in pixels))
    return result, dict(
        method="linear_0_to_nonzero_p99_5",
        source=checksum(raw),
        output=checksum(result),
        high=high,
        pixel_geometry="unchanged",
        clipped_fraction=sum(counts[high + 1 :]) / len(pixels),
        synthetic_content=False,
        independent_evidence=False,
    )
