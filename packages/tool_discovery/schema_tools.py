"""Additive 11B schema snapshots; never modify contracts v1."""

import argparse
import json
from pathlib import Path

from . import models_v11
from .models import DiscoveryPlan, DiscoveryPolicy, DiscoveryQuery, DiscoveryResult, Provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    legacy = (DiscoveryPlan, DiscoveryPolicy, DiscoveryQuery, DiscoveryResult, Provenance)
    current = tuple(getattr(models_v11, m.__name__) for m in legacy if m is not DiscoveryPolicy)
    current += (models_v11.Semantics,)
    for model in legacy + current:
        folder = Path(__file__).parent / "schemas"
        if model in current:
            folder /= "v1.1"
        path = folder / (model.__name__ + ".schema.json")
        expected = json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n"
        if args.write:
            path.parent.mkdir(exist_ok=True)
            path.write_text(expected)
        elif path.read_text() != expected:
            raise SystemExit(model.__name__ + " schema drift")
        print(model.__name__ + " schema verified")


if __name__ == "__main__":
    main()
