"""Deterministic 07A schemas; contracts v1 stays byte-identical."""

import argparse
import json
from pathlib import Path

from .golden import BoundGoldenCandidate, GoldenCandidate
from .models import BoundRegistration, Projection, Registration, RestrictedManifest, VisibleManifest
from .runtime_models import RuntimeInputProfile, RuntimeRegistration, RuntimeVisibleManifest

MODELS = (
    VisibleManifest,
    RestrictedManifest,
    Projection,
    Registration,
    BoundRegistration,
    GoldenCandidate,
    BoundGoldenCandidate,
    RuntimeInputProfile,
    RuntimeVisibleManifest,
    RuntimeRegistration,
)


def schemas(*, write=False):
    root = Path(__file__).parent / "schemas"
    for model in MODELS:
        target = root / (model.__name__ + ".schema.json")
        expected = (
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True, ensure_ascii=False)
            + "\n"
        )
        if write:
            target.write_text(expected)
        elif target.read_text() != expected:
            raise ValueError("dataset_schema_drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    schemas(write=args.write)
