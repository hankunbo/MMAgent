"""Explicit operator software handoff for an already started pre-LLM run.

Original input/start/receipts never change. This grants no tool/source authority;
all original owners and the original deadline are still checked during recovery.
"""

import json
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.grounding.models import digest
from packages.object_store import checksum

FIELDS = ("upstream_implementation", "report_implementation")
VERSION = "generic-implementation-repair/1.0.0"
COMPATIBILITY_REGISTRY = Path("configs/research/generic_implementation_compatibility_v1.json")
COMPATIBILITY_SCOPE = "frozen_input_software_compatibility_only"


def _read_proof(reference):
    path = Path(reference["path"]).resolve()
    if not path.is_relative_to(Path("docs/acceptance").resolve()):
        raise ValueError("generic_repair_proof_scope")
    raw = path.read_bytes()
    if checksum(raw) != reference["checksum"]:
        raise ValueError("generic_repair_proof_changed")
    return json.loads(raw)


def _proof(reference, *, scope="software_only_acquisition_definition_repair"):
    proof = _read_proof(reference)
    if proof.get("scope") != scope or proof.get("passed") is not True:
        raise ValueError("generic_repair_not_verified")
    freeze = Path(proof["freeze"]["path"])
    if checksum(freeze.read_bytes()) != proof["freeze"]["checksum"]:
        raise ValueError("generic_repair_freeze_changed")
    files = json.loads(freeze.read_bytes())["files"]
    for name, value in proof["after"].items():
        if name not in FIELDS:
            raise ValueError("generic_repair_unknown_field")
        for filename, expected in value.items():
            p = Path(filename)
            key = str(p.relative_to(Path.cwd()) if p.is_absolute() else p)
            if files.get(key) != expected or checksum(p.read_bytes()) != expected:
                raise ValueError("generic_repair_uncovered_implementation")
    for command in proof["commands"]:
        folder = Path(command)
        record = json.loads((folder / "command.json").read_bytes())
        if record["exit_code"] != 0 or (folder / "command.exit").read_text().strip() != "0":
            raise ValueError("generic_repair_failed_verification")
        if checksum((folder / "command.log").read_bytes()) != record["log_checksum"]:
            raise ValueError("generic_repair_verification_changed")
    return proof


def compatibility_reference(spec, field, current):
    """Exact configured software handoff, not run/current source authorization.

    Unlike the old pre-LLM repair this can explain a completed historical input.
    It writes nothing, never edits that input, and does not change any control,
    source or provider guard. New execution still needs its original permission.
    Both frozen maps must match one proof; no hash bypass or path-only allowlist.
    """
    if field not in FIELDS or not COMPATIBILITY_REGISTRY.exists():
        return None
    registry = json.loads(COMPATIBILITY_REGISTRY.read_bytes())
    references = registry.get("proofs")
    if (
        set(registry) != {"schema_version", "proofs"}
        or registry.get("schema_version") != "generic-software-compatibility-registry/1.0.0"
        or not isinstance(references, list)
        or not 1 <= len(references) <= 16
        or any(not isinstance(r, dict) or set(r) != {"path", "checksum"} for r in references)
    ):
        raise ValueError("generic_compatibility_registry")
    selected = None
    for reference in references:
        proof = _read_proof(reference)
        if (
            proof.get("schema_version") != "generic-software-compatibility/1.0.0"
            or proof.get("scope") != COMPATIBILITY_SCOPE
            or set(proof.get("before", {})) != set(FIELDS)
            or set(proof.get("after", {})) != set(FIELDS)
            or not isinstance(proof.get("commands"), list)
            or not proof["commands"]
        ):
            raise ValueError("generic_compatibility_proof")
        if (
            any(proof["before"][name] != spec.get(name) for name in FIELDS)
            or proof["after"][field] != current
        ):
            continue
        _proof(reference, scope=COMPATIBILITY_SCOPE)
        if selected is not None:
            raise ValueError("generic_compatibility_ambiguous")
        selected = reference
    return selected


def implementation_matches(spec, field, current):
    if spec.get(field) == current:
        return True
    if compatibility_reference(spec, field, current) is not None:
        return True
    from .generic_control import guard
    from .generic_input import directory

    path = directory(spec["key"]) / "implementation-repair.json"
    if field not in FIELDS or not path.exists():
        return False
    record = json.loads(path.read_bytes())
    started = json.loads((path.parent / "started.json").read_bytes())
    if (
        record.get("schema_version") != VERSION
        or record.get("key") != spec["key"]
        or record.get("input_checksum") != digest(spec)
        or record.get("start_checksum") != digest(started)
        or started["input_checksum"] != digest(spec)
    ):
        raise ValueError("generic_repair_scope_or_start_changed")
    proof = _proof(record["proof"])
    if proof["before"].get(field) != spec.get(field) or proof["after"].get(field) != current:
        return False
    guard(spec["key"])
    return True


def authorize(key, proof_path):
    """Operator calls only after a reviewed, passing increment; no runtime/secret access."""
    import fcntl

    from .generic_control import guard
    from .generic_input import directory

    out = directory(key)
    with (out / "operation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = json.loads((out / "input.json").read_bytes())
        started = json.loads((out / "started.json").read_bytes())
        guard(key)
        if spec["key"] != key or started["input_checksum"] != digest(spec):
            raise ValueError("generic_repair_input_changed")
        if (out / "export-reference.json").exists() or list(
            (out / "evidence").glob("**/request.json")
        ):
            raise ValueError("generic_repair_only_pre_llm")
        if (out / "fairclip-generic-v1/request.json").exists():
            raise ValueError("generic_repair_model_attempt_requires_owner_recovery")
        for ledger in (out / "tools").glob("*.json"):
            value = json.loads(ledger.read_bytes())
            if value.get("tool_step") != "segman" or value.get("status") != "owner_returned":
                raise ValueError("generic_repair_unconfirmed_tool")
        reference = dict(
            path=str(Path(proof_path)), checksum=checksum(Path(proof_path).read_bytes())
        )
        proof = _proof(reference)
        if set(proof["before"]) != set(FIELDS) or set(proof["after"]) != set(FIELDS):
            raise ValueError("generic_repair_field_set")
        if any(proof["before"][field] != spec.get(field) for field in FIELDS):
            raise ValueError("generic_repair_wrong_baseline")
        record = dict(
            schema_version=VERSION,
            key=key,
            input_checksum=digest(spec),
            start_checksum=digest(started),
            proof=reference,
            semantics="explicit_software_handoff_same_key_no_new_grant_budget_or_deadline",
            tools_reexecuted=0,
            api_requests=0,
        )
        write_once(out / "implementation-repair.json", record)
        return record
