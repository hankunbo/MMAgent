"""Bounded, pinned upstream subset download; no installation or patient execution."""

import json
import urllib.request
from pathlib import Path

from packages.object_store import checksum

COMMIT = "9a953e5edfa419b454e9fb7b06235eb413828d05"
ROOT = Path(".local/automorph-disc-cup-" + COMMIT[:12]).absolute()
FILES = (
    "LICENSE",
    "M2_lwnet_disc_cup/models/get_model.py",
    "M2_lwnet_disc_cup/models/res_unet_adrian.py",
    "M2_lwnet_disc_cup/utils/get_loaders.py",
    "M2_lwnet_disc_cup/utils/paired_transforms_tv04.py",
    "M2_lwnet_disc_cup/utils/model_saving_loading.py",
    "M2_lwnet_disc_cup/generate_av_results.py",
    "M2_lwnet_disc_cup/experiments/wnet_All_three_1024_disc_cup/30/config.cfg",
    "M2_lwnet_disc_cup/experiments/wnet_All_three_1024_disc_cup/30/model_checkpoint.pth",
)
ENSEMBLE = tuple(
    "M2_lwnet_disc_cup/experiments/wnet_All_three_1024_disc_cup/"
    + str(seed)
    + "/model_checkpoint.pth"
    for seed in (28, 32, 34, 36, 38, 40, 42)
)


def fetch():
    from apps.vectorengine_single_case import write_once

    resources = {}
    for name in FILES + ENSEMBLE:
        path = ROOT / name
        url = "https://raw.githubusercontent.com/rmaphoh/AutoMorph/" + COMMIT + "/" + name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=40) as response:
                raw = response.read(4_194_305)
            if len(raw) > 4_194_304:
                raise ValueError("automorph_resource_size")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
        resources[name] = checksum(path.read_bytes())
    value = dict(source="https://github.com/rmaphoh/AutoMorph", commit=COMMIT, files=resources)
    write_once(ROOT / "ensemble-source-lock.json", value)
    return value


if __name__ == "__main__":
    print(json.dumps(fetch(), indent=2))
