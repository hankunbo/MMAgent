"""Explicit new deployment evidence, never edit old inventories or model semantics."""

import hashlib
import importlib.util
import json
import marshal
import subprocess
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.grounding.models import digest, validate
from packages.research_runtime.deployment import ROOT, load_configuration, read_installed_bundle
from packages.research_runtime.readiness import UnboundSingleCasePolicy


def pin(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("fairclip_inventory_file")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return dict(
        path=str(path), byte_size=path.stat().st_size, checksum="sha256:" + hasher.hexdigest()
    )


def prepare(output, *, selected_gpu=None, generic_training=False):
    output = Path(output).resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    configuration = load_configuration(ROOT / "configs/research/fairclip_single_case.json")
    old = read_installed_bundle(configuration["installed_bundle"])
    bundle = json.loads(json.dumps(old))
    review = bundle["execution_policy"]["reviews"][0]
    before = {v["path"]: v for v in review["files"]}
    actual = {}
    for directory in review["inventory_roots"]:
        for path in sorted(Path(directory).rglob("*")):
            if path.is_symlink():
                raise ValueError("fairclip_inventory_symlink")
            if path.is_file():
                actual[str(path)] = pin(path)
    missing = sorted(before.keys() - actual.keys())
    added = sorted(actual.keys() - before.keys())
    changed = sorted(k for k in before.keys() & actual.keys() if before[k] != actual[k])
    write_once(
        output / "observed-inventory.json",
        dict(files=actual, added=added, changed=changed, missing=missing),
    )
    reviewed_caches = set()
    for path in added:
        cache = Path(path)
        if cache.suffix == ".pyc" and not cache.is_relative_to(ROOT / "apps"):
            source = Path(importlib.util.source_from_cache(path))
            if str(source) not in before or actual[str(source)] != before[str(source)]:
                raise ValueError("fairclip_generated_cache_source_not_reviewed")
            data = cache.read_bytes()
            if data[:4] != importlib.util.MAGIC_NUMBER or marshal.loads(data[16:]) != compile(
                source.read_bytes(), str(source), "exec", dont_inherit=True, optimize=0
            ):
                raise ValueError("fairclip_generated_cache_not_equal_to_reviewed_source")
            reviewed_caches.add(str(cache))
    if missing or any(
        not Path(p).is_relative_to(ROOT / "apps") and p not in reviewed_caches
        for p in added + changed
    ):
        raise ValueError("fairclip_non_application_inventory_change_requires_review")
    # Model-facing entrypoint and the isolated environment/model/support closure
    # must remain byte-identical. New unrelated app files are not new model code.
    if (
        actual[review["entrypoint"]["path"]] != review["entrypoint"]
        or actual[review["executable"]["path"]] != review["executable"]
    ):
        raise ValueError("fairclip_executed_entrypoint_changed")
    inspected = """# FairCLIP current application inventory review

Scope: the two approved new FairVLMed SLO research cases, unchanged original 06
execution and shared-GPU policy. No installation, download or qualification upgrade.
The executed apps/fairclip_tool.py only adds the repository root and calls the
fixed packages.fairclip_research.backend. That backend loads only tensor weights
with weights_only=True and the three digest-pinned model/tokenizer/BPE resources;
it does not discover or import sibling apps or dataset labels. The model-facing
entrypoint, environment, fairclip_research sources, weights and support resources were
hashed and required to match the old review. The difference JSON enumerates all
new/changed sibling application files; they are not used by this model entry.
Added CPython caches, including environment imports, are accepted only after their code
objects equal compilation of the unchanged reviewed sources; they are pinned too.
This updates the broad application-directory inventory, not the medical validation
level. Original read-only catalog admission, exact files, capability limits,
minimal environment, project GPU lock and UNKNOWN recovery remain mandatory.
The current environment proof is separately measured; old dates are not extended.
This review does not promise process confinement or clinical accuracy.
"""
    if generic_training:
        inspected = inspected.replace(
            "the two approved new FairVLMed SLO research cases",
            "user-authorized selected Training FairVLMed SLO research inputs under the "
            "generic selected-input/1.0.0 adapter; exact input, run and budgets bound separately",
        )
    record = output / "review.md"
    if record.exists():
        if record.read_text() != inspected:
            raise ValueError("fairclip_review_conflict")
    else:
        record.write_text(inspected)
        record.chmod(0o600)
    write_once(
        output / "inventory-delta.json",
        dict(
            added=added,
            changed=changed,
            missing=missing,
            old_bundle=configuration["installed_bundle"],
            model_closure_unchanged=True,
            source_equivalent_generated_caches=sorted(reviewed_caches),
        ),
    )
    review["files"] = [actual[p] for p in sorted(actual)]
    review["review_record"] = pin(record)
    policy = bundle["execution_policy"]
    if selected_gpu:
        # Explicit new host proof, not mutation or expiry extension of old evidence.
        result = subprocess.run(
            [
                policy["nvidia_smi"]["path"],
                "--query-gpu=uuid,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            env={},
        )
        devices = [line.split(", ") for line in result.stdout.strip().splitlines()]
        chosen = [d for d in devices if d[0] == selected_gpu]
        if len(chosen) != 1 or float(chosen[0][3]) < policy["vram_budget_mib"]:
            raise ValueError("fairclip_current_host_device_unavailable")
        policy["nvidia_smi"] = pin(policy["nvidia_smi"]["path"])
        policy["gpu_pool"], policy["selected_gpu"] = [selected_gpu], selected_gpu
        write_once(
            output / "host-observation.json",
            dict(
                devices=devices,
                selected_gpu=selected_gpu,
                nvidia_smi=policy["nvidia_smi"],
                original_gpu=old["execution_policy"]["selected_gpu"],
                inference_or_weights_changed=False,
                reason="container host GPU/driver changed",
            ),
        )
    policy["reference"] = dict(
        policy["reference"],
        version="1.3.0" if selected_gpu else "1.1.0",
        checksum=digest({k: v for k, v in policy.items() if k != "reference"}),
    )
    bundle["execution_policy"] = validate(UnboundSingleCasePolicy, policy).model_dump(mode="json")
    write_once(output / "installed-bundle.json", bundle)
    configuration["installed_bundle"] = pin(output / "installed-bundle.json")
    configuration["checksum"] = digest({k: v for k, v in configuration.items() if k != "checksum"})
    write_once(output / "deployment-config.json", configuration)
    return str(output / "deployment-config.json")
