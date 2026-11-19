"""Bounded byte adaptation into the unchanged registered geometry owner."""

import os
import tempfile
from pathlib import Path

from packages.multimodal_research.fundus_measurements import measure


def measure_masks(raw):
    """Private ephemeral files satisfy the original path port; never copy formulas."""
    if set(raw) != {"disc", "cup", "rim", "labels"}:
        raise ValueError("fundus_mask_set")
    if any(not isinstance(value, bytes) or len(value) > 4 * 1024 * 1024 for value in raw.values()):
        raise ValueError("fundus_mask_bytes_bound")
    with tempfile.TemporaryDirectory(prefix="mmagent-mask-measure-") as name:
        directory = Path(name)
        for kind, value in raw.items():
            with (directory / (kind + ".png")).open("xb") as stream:
                os.chmod(stream.name, 0o600)
                stream.write(value)
        return measure(directory)
