"""Read deployment-owned policy declarations with locally verifiable provenance."""

import hashlib
import json
from pathlib import Path

from .models import AdapterError, ProjectionPolicy, parse


def load_projection_policy(path: Path, repository_root: Path):
    if path.stat().st_size > 262144:
        raise AdapterError("policy_byte_limit")
    policy = parse(ProjectionPolicy, json.loads(path.read_text()))
    root = repository_root.resolve(strict=True)
    for declaration in policy.declarations:
        source = (root / declaration.source_path).resolve(strict=True)
        if (
            not source.is_relative_to(root)
            or not source.is_file()
            or source.stat().st_size > 1048576
        ):
            raise AdapterError("declaration_source_outside_bounds")
        raw = source.read_bytes()
        if "sha256:" + hashlib.sha256(raw).hexdigest() != declaration.source_checksum:
            raise AdapterError("declaration_source_drift")
        if declaration.excerpt not in raw.decode("utf-8"):
            raise AdapterError("declaration_excerpt_missing")
    return policy
