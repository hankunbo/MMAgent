"""A reviewed code transition is not permission to run or mutate old input."""

import hashlib
import json
from copy import deepcopy

import pytest

from apps import segman_resource_lock as resource
from packages.evaluation_preparation import generic_execution_lock as owner


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def ref(root, path):
    return dict(
        path=str(path.relative_to(root)),
        checksum="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    )


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    root = tmp_path
    for name in resource.LOCK_FILES:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("current: " + name)
    current = resource.implementation(root)
    before = {
        name: "sha256:" + hashlib.sha256(("old: " + name).encode()).hexdigest() for name in current
    }
    spec = dict(
        schema_version="fairvlmed-run-input/1.10.0",
        key="example",
        execution_lock_profile=resource.PROFILE,
        execution_lock_implementation=before,
    )
    case = root / "case"
    case.mkdir()
    save(case / "input.json", spec)
    binding = dict(
        schema_version="generic-case-binding/1.2.0",
        execution_namespace="case",
        input_spec_checksum=resource.digest(spec),
    )
    spec["key"] = "case"
    save(case / "input.json", spec)
    binding["input_spec_checksum"] = resource.digest(spec)
    d = root / "docs/acceptance/checked"
    d.mkdir(parents=True)
    log = d / "command.log"
    log.write_text("passed\n")
    code = d / "command.exit"
    code.write_text("0\n")
    command = save(d / "command.json", dict(direct_exit=0, log_checksum=ref(root, log)["checksum"]))
    freeze = save(d / "freeze.json", dict(files=current))
    proof = dict(
        schema_version="generic-execution-software-handoff/1.0.0",
        scope="software_compatibility_only_no_dispatch_authority",
        passed=True,
        execution_authorized=False,
        before=before,
        after=current,
        freeze=ref(root, freeze),
        commands=[dict(receipt=ref(root, command), exit=ref(root, code), log=ref(root, log))],
    )
    path = save(d / "proof.json", proof)
    registry = root / "configs/research/generic_execution_compatibility_v1.json"
    save(
        registry,
        dict(
            schema_version="generic-execution-compatibility-registry/1.0.0",
            proofs=[ref(root, path)],
        ),
    )
    monkeypatch.setattr(owner, "ROOT", root)
    return root, spec, current, proof, path, registry, case, binding


def test_exact_approved_software_reads_preserve_original_input(handoff):
    root, spec, current, _, _, _, case, binding = handoff
    before = (case / "input.json").read_bytes()
    assert resource.execution_compatibility(root, spec, current)
    assert owner.validate_profile(spec)
    assert resource.verify_input(root, case, binding) == spec
    assert (case / "input.json").read_bytes() == before


@pytest.mark.parametrize(
    "fault",
    [
        "input",
        "current",
        "scope",
        "authority",
        "collection",
        "freeze",
        "failed_exit",
        "log",
        "duplicate",
        "outside",
    ],
)
def test_unverified_transition_is_rejected(handoff, fault):
    root, spec, current, proof, path, registry, case, binding = handoff
    if fault in {"input", "current"}:
        target = spec["execution_lock_implementation"] if fault == "input" else current
        target[next(iter(target))] = "sha256:" + "0" * 64
        assert not resource.execution_compatibility(root, spec, current)
        return
    if fault == "scope":
        proof["scope"] = "execute"
    elif fault == "authority":
        proof["execution_authorized"] = True
    elif fault == "collection":
        proof["before"].pop(next(iter(proof["before"])))
    elif fault == "freeze":
        frozen = save(path.parent / "other-freeze.json", dict(files={}))
        proof["freeze"] = ref(root, frozen)
    elif fault == "failed_exit":
        code = path.parent / "command.exit"
        code.write_text("1\n")
        proof["commands"][0]["exit"] = ref(root, code)
    elif fault == "log":
        log = path.parent / "command.log"
        log.write_text("changed\n")
        proof["commands"][0]["log"] = ref(root, log)
    elif fault == "outside":
        f = save(root / "outside.json", dict(files=current))
        proof["freeze"] = ref(root, f)
    save(path, proof)
    refs = [ref(root, path)] * (2 if fault == "duplicate" else 1)
    save(
        registry, dict(schema_version="generic-execution-compatibility-registry/1.0.0", proofs=refs)
    )
    with pytest.raises(ValueError, match="generic_execution_compatibility_"):
        resource.execution_compatibility(root, spec, current)


def test_unmatched_current_code_cannot_pass_runtime_or_tool_gate(handoff):
    root, spec, current, _, _, _, case, binding = handoff
    (root / resource.LOCK_FILES[0]).write_text("unreviewed change")
    with pytest.raises(ValueError, match="generic_execution_lock_configuration_changed"):
        owner.validate_profile(spec)
    with pytest.raises(ValueError, match="generic_execution_lock_binding"):
        resource.verify_input(root, case, binding)


def test_exact_code_without_registry_keeps_old_contract(handoff):
    root, spec, current, _, _, registry, case, binding = handoff
    registry.unlink()
    spec = deepcopy(spec)
    spec["execution_lock_implementation"] = current
    assert owner.validate_profile(spec)
    spec["execution_lock_profile"] = "wrong"
    with pytest.raises(ValueError, match="generic_execution_lock_configuration_changed"):
        owner.validate_profile(spec)
