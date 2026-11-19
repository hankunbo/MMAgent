"""A reviewed code handoff cannot reset a running case's input, budget or deadline."""

import json
from pathlib import Path

import pytest

from packages.evaluation_preparation.generic_control import cancel, start
from packages.evaluation_preparation.generic_input import directory
from packages.evaluation_preparation.generic_repair import authorize, implementation_matches
from packages.grounding.models import digest
from packages.llm_provider.provider import ProviderError
from packages.object_store import checksum


@pytest.fixture
def repair(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    current = Path("component.py")
    current.write_text("corrected software")
    after = {str(current): checksum(current.read_bytes())}
    before = {str(current): digest("old implementation")}
    spec = dict(key="repair-case", upstream_implementation=before, report_implementation=before)
    out = directory(spec["key"])
    out.mkdir(parents=True)
    (out / "input.json").write_text(json.dumps(spec))
    started = start(spec)
    evidence = Path("docs/acceptance/repair")
    evidence.mkdir(parents=True)
    frozen = evidence / "freeze.json"
    frozen.write_text(json.dumps(dict(files=after)))
    command = evidence / "test"
    command.mkdir()
    (command / "command.log").write_text("1 passed")
    (command / "command.exit").write_text("0\n")
    (command / "command.json").write_text(
        json.dumps(dict(exit_code=0, log_checksum=checksum((command / "command.log").read_bytes())))
    )
    proof = dict(
        scope="software_only_acquisition_definition_repair",
        passed=True,
        freeze=dict(path=str(frozen), checksum=checksum(frozen.read_bytes())),
        commands=[str(command)],
        before={k: before for k in ("upstream_implementation", "report_implementation")},
        after={k: after for k in ("upstream_implementation", "report_implementation")},
    )
    path = evidence / "verification.json"
    path.write_text(json.dumps(proof))
    return spec, out, started, path, after


def test_explicit_handoff_same_key_immutable_input_start_and_retry(repair):
    spec, out, started, path, after = repair
    original = (out / "input.json").read_bytes()
    assert not implementation_matches(spec, "upstream_implementation", after)
    first = authorize(spec["key"], path)
    assert authorize(spec["key"], path) == first
    for field in ("upstream_implementation", "report_implementation"):
        assert implementation_matches(spec, field, after)
    assert (out / "input.json").read_bytes() == original
    assert json.loads((out / "started.json").read_bytes()) == started
    assert start(spec) == started


@pytest.mark.parametrize("fault", ["input", "start", "proof", "code", "test_exit", "cancel"])
def test_handoff_cannot_cover_unreviewed_or_cancelled_state(repair, fault):
    spec, out, _, path, after = repair
    authorize(spec["key"], path)
    if fault == "input":
        spec = dict(spec, changed=True)
    elif fault == "start":
        value = json.loads((out / "started.json").read_bytes())
        value["deadline"] = "2999-01-01T00:00:00+00:00"
        (out / "started.json").write_text(json.dumps(value))
    elif fault == "proof":
        path.write_text("{}")
    elif fault == "code":
        Path("component.py").write_text("unreviewed")
    elif fault == "test_exit":
        (path.parent / "test/command.exit").write_text("1")
    else:
        cancel(spec["key"])
    with pytest.raises((ValueError, ProviderError)):
        implementation_matches(spec, "upstream_implementation", after)


def test_repair_cannot_reauthorize_existing_llm_phase(repair):
    spec, out, _, path, _ = repair
    phase = out / "evidence/review"
    phase.mkdir(parents=True)
    (phase / "request.json").write_text("{}")
    with pytest.raises(ValueError, match="only_pre_llm"):
        authorize(spec["key"], path)


def compatibility_registry(path, *, commands=True):
    proof = json.loads(path.read_bytes())
    proof.update(
        schema_version="generic-software-compatibility/1.0.0",
        scope="frozen_input_software_compatibility_only",
    )
    if not commands:
        proof["commands"] = []
    path.write_text(json.dumps(proof))
    reference = dict(path=str(path), checksum=checksum(path.read_bytes()))
    registry = Path("configs/research/generic_implementation_compatibility_v1.json")
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            dict(schema_version="generic-software-compatibility-registry/1.0.0", proofs=[reference])
        )
    )
    return registry


def test_software_compatibility_can_read_stopped_history_but_cannot_authorize_execution(repair):
    from packages.evaluation_preparation.generic_control import guard

    spec, out, _, path, after = repair
    compatibility_registry(path)
    cancel(spec["key"])
    before = {str(p): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    for field in ("upstream_implementation", "report_implementation"):
        assert implementation_matches(spec, field, after)
    with pytest.raises(ProviderError):
        guard(spec["key"])
    assert not (out / "implementation-repair.json").exists()
    assert before == {str(p): p.read_bytes() for p in out.rglob("*") if p.is_file()}


@pytest.mark.parametrize("fault", ["proof", "code", "exit", "log", "empty", "version", "ambiguous"])
def test_software_compatibility_requires_exact_verified_proof(repair, fault):
    spec, _, _, path, after = repair
    registry = compatibility_registry(path, commands=fault != "empty")
    if fault == "proof":
        path.write_text("{}")
    elif fault == "code":
        Path("component.py").write_text("unreviewed")
    elif fault == "exit":
        (path.parent / "test/command.exit").write_text("1")
    elif fault == "log":
        (path.parent / "test/command.log").write_text("edited")
    elif fault in ("version", "ambiguous"):
        raw = json.loads(registry.read_bytes())
        if fault == "version":
            raw["schema_version"] = "unsupported"
        else:
            raw["proofs"] *= 2
        registry.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="generic_.*"):
        implementation_matches(spec, "upstream_implementation", after)


def test_software_compatibility_does_not_cover_other_baselines_or_extra_files(repair):
    spec, _, _, path, after = repair
    compatibility_registry(path)
    assert not implementation_matches(
        dict(spec, report_implementation={}), "upstream_implementation", after
    )
    assert not implementation_matches(
        spec, "upstream_implementation", dict(after, extra=digest("new"))
    )
