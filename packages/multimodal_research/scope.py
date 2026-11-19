"""Patient-level planning scopes composed from immutable prepared inputs.

The scope is a planning envelope only.  It does not merge raw payloads, alter
the original preparations, run tools, or grant clinical/execution authority.
"""

from __future__ import annotations

import json
from pathlib import Path

from packages.grounding.models import digest


def create_baseline_bilateral(inputs, output):
    from packages.dataset_adapters.multimodal.preparation import verify

    if len(inputs) != 2:
        raise ValueError("planner_bilateral_scope_requires_two_inputs")
    manifests = [verify(p) for p in inputs]
    if any(m.dataset != "grape" for m in manifests):
        raise ValueError("planner_bilateral_scope_dataset")
    samples = [m.sample for m in manifests]
    subjects = {s.split(":", 1)[0] for s in samples}
    eyes = {s.split(":", 1)[1] for s in samples if ":" in s}
    if len(subjects) != 1 or eyes != {"OD", "OS"}:
        raise ValueError("planner_bilateral_scope_eye_pair")
    if any(m.cutoff_visit != 0 for m in manifests):
        raise ValueError("planner_bilateral_scope_baseline_only")
    components = []
    for path, manifest in zip(inputs, manifests):
        components.append({
            "input": str(Path(path).absolute()),
            "checksum": manifest.checksum,
            "sample": manifest.sample,
            "eye": manifest.sample.split(":", 1)[1],
            "component_ids": [c.id for c in manifest.components],
        })
    scope = {
        "schema_version": "multimodal-planning-scope/1.0.0",
        "scope": "baseline_bilateral",
        "dataset": "grape",
        "subject": next(iter(subjects)),
        "visit": 0,
        "eyes": ["OD", "OS"],
        "inputs": components,
        "execution_authorized": False,
        "api_requests": 0,
        "model_runs": 0,
        "database_writes": 0,
    }
    scope["checksum"] = digest(scope)
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("planner_scope_output_must_be_new")
    out.mkdir(parents=True, mode=0o700)
    with (out / "scope.json").open("x") as stream:
        json.dump(scope, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"output": str(out), "checksum": scope["checksum"], "scope": scope["scope"],
            "eyes": scope["eyes"], "visit": scope["visit"],
            "execution_authorized": False, "api_requests": 0, "database_writes": 0}


def read(path):
    value = json.loads(Path(path).absolute().read_bytes())
    if value.get("schema_version") != "multimodal-planning-scope/1.0.0":
        raise ValueError("planner_scope_schema")
    if value.get("scope") != "baseline_bilateral" or value.get("eyes") != ["OD", "OS"]:
        raise ValueError("planner_scope_not_baseline_bilateral")
    if value.get("visit") != 0 or value.get("execution_authorized") is not False:
        raise ValueError("planner_scope_authorization")
    if digest({k: v for k, v in value.items() if k != "checksum"}) != value.get("checksum"):
        raise ValueError("planner_scope_checksum")
    return value
