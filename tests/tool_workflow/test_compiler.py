import builtins
import dataclasses
import hashlib
import io
import json
import os
import subprocess
from datetime import timedelta

import pytest

from packages.grounding.models import canonical, digest, validate
from packages.tool_workflow import (
    ExecutionPlan,
    FormalContext,
    ResearchContext,
    ToolWorkflowCompiler,
)
from tests.tool_admission.support import reseal
from tests.tool_admission.test_research import research
from tests.tool_workflow.support import choose, commands, prepared


def compile_plan(f):
    return f.compiler.compile(f.context, f.choice)


def test_multistep_topology_and_shared_upstream(tmp_path):
    f = prepared(tmp_path, chain=True)
    f.choice = choose(f.context.admission, all_roots=True)
    plan = compile_plan(f)
    steps = plan.document.steps
    assert len(steps) == 3
    assert len({s.step_id for s in steps}) == 3
    assert steps[1].dependencies == steps[2].dependencies == (steps[0].step_id,)
    assert not steps[0].dependencies
    assert all(
        o.bytes_checksum is None and not o.exists and o.initial_qc_status == "PENDING"
        for s in steps
        for o in s.outputs
    )
    assert all(any(i.qc_before_consume for i in s.inputs) for s in steps[1:])
    assert all(
        s.pending_dynamic_checks == f.context.admission.retained[0].pending_dynamic_checks
        for s in steps
    )
    assert not plan.document.execution_authorized
    assert f.compiler.validate_plan(plan) == plan


def test_single_root_does_not_execute_all_retained_roots(tmp_path):
    f = prepared(tmp_path, chain=True)
    assert len(f.context.admission.retained_roots) == 2
    assert len(compile_plan(f).document.steps) == 2


def test_order_invariance_and_no_input_mutation(tmp_path):
    f = prepared(tmp_path, chain=True)
    original = canonical((f.context, f.choice))
    first = compile_plan(f)
    data = f.choice.model_dump(mode="json")
    data["roots"].reverse()
    data["nodes"].reverse()
    for n in data["nodes"]:
        n["inputs"].reverse()
        for p in n["inputs"]:
            p["sources"].reverse()
    selection = f.context.selection.model_dump(mode="json")
    selection["candidate_refs"].reverse()
    selection["invocations"].reverse()
    ctx = validate(FormalContext, {**f.context.model_dump(mode="json"), "selection": selection})
    assert f.compiler.compile(ctx, data) == first
    assert canonical((f.context, f.choice)) == original
    detached = first.dry_run()
    detached["steps"][0]["argv"][0] = "evil"
    assert compile_plan(f) == first
    with pytest.raises(dataclasses.FrozenInstanceError):
        first._document_json = "{}"


@pytest.mark.parametrize(
    "change",
    [
        "AND_missing",
        "OR_missing",
        "cardinality",
        "wrong_port",
        "unknown_root",
        "duplicate_step",
        "duplicate_source",
        "or_all_branches",
    ],
)
def test_invalid_choices(tmp_path, change):
    f = prepared(tmp_path, chain=True)
    data = f.choice.model_dump(mode="json")
    root = next(n for n in data["nodes"] if n["candidate_ref"] == data["roots"][0])
    if change == "AND_missing":
        root["inputs"].pop()
    elif change == "OR_missing":
        data["nodes"] = [root]
    elif change == "cardinality":
        next(p for p in root["inputs"] if p["sources"])["sources"] = []
    elif change == "wrong_port":
        root["inputs"][0]["port"]["tool_ref"]["version"] = "999.0.0"
    elif change == "unknown_root":
        data["roots"] = [digest("absent")]
    elif change == "duplicate_step":
        data["nodes"].append(data["nodes"][0])
    elif change == "duplicate_source":
        p = next(p for p in root["inputs"] if p["sources"])
        p["sources"].append(p["sources"][0])
    else:
        p = next(p for p in root["inputs"] if p["sources"])
        p["sources"].append(dict(kind="artifact", asset_code="ASSET:OID_MISSING01"))
    with pytest.raises(ValueError):
        f.compiler.compile(f.context, data)


@pytest.mark.parametrize(
    "change", ["argv", "scope", "purpose", "config", "output", "checks", "checksum"]
)
def test_modified_plan_even_resealed_rejected(tmp_path, change):
    f = prepared(tmp_path)
    plan = compile_plan(f)
    data = plan.dry_run()
    if change == "argv":
        data["steps"][0]["argv"].append("-c")
    elif change == "scope":
        data["scope"]["patient_hash"] = digest("different-patient")
    elif change == "purpose":
        data["purpose"] = "shadow"
    elif change == "config":
        data["config"] = reseal(f.config, version="2.0.0").model_dump(mode="json")
    elif change == "output":
        data["steps"][0]["outputs"][0]["logical_uri"] = "file:///tmp/forged.png"
    elif change == "checks":
        data["steps"][0]["pending_dynamic_checks"] = []
    data.pop("checksum")
    data["checksum"] = digest(data) if change != "checksum" else digest("wrong")
    forged = ExecutionPlan(canonical(data).decode(), plan._context_json)
    with pytest.raises(ValueError):
        f.compiler.validate_plan(forged)


@pytest.mark.parametrize(
    "change",
    ["expired", "revoked", "input_bytes", "resource_bytes", "profile", "invocation", "approval"],
)
def test_controlled_readmission_required(tmp_path, change):
    f = prepared(tmp_path)
    plan = compile_plan(f)
    if change == "expired":
        f.f.service.clock = lambda: f.context.request.snapshot_at + timedelta(days=3)
    elif change == "revoked":
        f.f.catalogs.disable(plan.document.steps[0].implementation_ref)
    elif change == "input_bytes":
        record = next(iter(f.f.owners.records.values()))
        f.f.objects._path(record.artifact.uri).write_bytes(b"changed")
    elif change == "resource_bytes":
        (tmp_path / "synthetic-entrypoint.txt").write_bytes(b"changed")
    elif change == "profile":
        manifest = next(iter(f.f.owners.manifests.values()))
        p = f.f.owners.profiles[manifest.checksum]
        f.f.owners.profiles[manifest.checksum] = reseal(p, version="2.0.0")
    else:
        data = json.loads(plan._context_json)
        if change == "invocation":
            data["selection"]["invocations"][0]["parameters"] = {"threshold": 0.4}
        else:
            data["admission"]["execution_authorized"] = True
        plan = dataclasses.replace(plan, _context_json=canonical(data).decode())
    with pytest.raises((ValueError, OSError)):
        f.compiler.validate_plan(plan)


def snapshot(root):
    return {
        str(p.relative_to(root)): (
            "dir" if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
        )
        for p in root.rglob("*")
    }


def test_dry_run_no_process_or_filesystem_writes(tmp_path, monkeypatch):
    f = prepared(tmp_path, chain=True)
    before = snapshot(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run attempted a process or filesystem mutation")

    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    for name in (
        "system",
        "fork",
        "posix_spawn",
        "posix_spawnp",
        "mkdir",
        "makedirs",
        "rename",
        "replace",
        "unlink",
        "remove",
    ):
        monkeypatch.setattr(os, name, forbidden)
    original_open, original_io, original_os_open = builtins.open, io.open, os.open

    def safe_open(original):
        def read_only(file, mode="r", *args, **kwargs):
            assert not any(m in mode for m in "wax+"), mode
            return original(file, mode, *args, **kwargs)

        return read_only

    def safe_os_open(path, flags, *args, **kwargs):
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        return original_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", safe_open(original_open))
    monkeypatch.setattr(io, "open", safe_open(original_io))
    monkeypatch.setattr(os, "open", safe_os_open)
    plan = compile_plan(f)
    assert json.loads(plan.dry_run_json()) == plan.dry_run()
    f.compiler.validate_plan(plan)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("purpose", ["debug", "shadow"])
def test_research_port_real_readmission(tmp_path, purpose):
    f = prepared(tmp_path)
    req = research(f.f, tmp_path, purpose=purpose)
    reader, invocations = commands(f.f, tmp_path, f.config, catalog_ref=req.lock.catalog_ref)
    invocation = next(
        i
        for i in invocations
        if i.implementation_ref == req.candidate.implementation_ref
        and f.f.owners.manifests[
            (req.lock.catalog_ref, req.candidate.tool_ref, i.implementation_ref)
        ].checksum
        == i.manifest_checksum
    )
    req = reseal(req, invocation=invocation)
    f.f.owners.requests[(req.code, req.version, req.checksum)] = req
    admission = f.f.service.admit_research(req)
    assert admission.status == "preparation_only"
    context = ResearchContext(request=req, admission=admission)
    compiler = ToolWorkflowCompiler(f.f.service, reader, f.config)
    plan = compiler.compile(context, choose(admission))
    assert plan.document.purpose == purpose
    assert not plan.document.decisive_evidence
    assert compiler.validate_plan(plan) == plan
    del f.f.owners.requests[(req.code, req.version, req.checksum)]
    with pytest.raises(KeyError):
        compiler.validate_plan(plan)


def test_serialized_bundle_retains_actual_revalidation_context(tmp_path):
    f = prepared(tmp_path)
    plan = compile_plan(f)
    recovered = ExecutionPlan.from_bundle_json(plan.to_bundle_json())
    assert f.compiler.validate_plan(recovered) == plan
    bundle = json.loads(plan.to_bundle_json())
    assert set(bundle["context"]) == {"kind", "request", "query_result", "selection", "admission"}
    bundle["context"]["request"]["scope"]["patient_hash"] = digest("other")
    with pytest.raises(ValueError, match="context_binding_mismatch|cross patient"):
        ExecutionPlan.from_bundle_json(canonical(bundle).decode())
