"""Stdlib-only resource exclusion for the explicitly versioned generic runner.

The parent owns the original Runtime resource lease; this host lock additionally
protects local runner processes. When NFS locking works we also hold the old lock,
so old and new runners remain mutually exclusive. This is not execution authority.
"""

import errno
import fcntl
import hashlib
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path

PROFILE = "runtime-case-host-resource/1.0.0"
INPUT_VERSION = "fairvlmed-run-input/1.9.0"
INPUT_VERSIONS = (INPUT_VERSION, "fairvlmed-run-input/1.10.0", "fairvlmed-run-input/1.11.0")
LOCK_FILES = (
    "apps/segman_resource_lock.py",
    "apps/segman_generic_runner.py",
    "apps/glaucoma_case_evidence.py",
    "apps/fairvlmed_case.py",
    "db/runtime/repository/case_pipeline.py",
    "packages/evaluation_preparation/generic_execution_lock.py",
    "packages/evaluation_preparation/generic_input.py",
    "packages/evaluation_preparation/generic_run.py",
    "packages/evaluation_preparation/generic_control.py",
    "packages/evaluation_preparation/generic_campaign.py",
    "packages/evaluation_preparation/case_execution.py",
    "packages/evaluation_preparation/comprehensive_tools.py",
)


def digest(value):
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode()
        ).hexdigest()
    )


def implementation(root):
    return {
        name: "sha256:" + hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in LOCK_FILES
    }


def execution_compatibility(root, spec, current):
    """Exact reviewed software handoff only; never an execution/lease grant.

    Frozen inputs and their original hashes remain unchanged. Both implementations
    and successful verification artifacts must match the configured proof. Input,
    Runtime, cancellation, budget and tool qualification gates still run normally.
    The helper stays stdlib-only for the original tool subprocess environment.
    """
    root = Path(root).resolve()
    registry = root / "configs/research/generic_execution_compatibility_v1.json"
    if not registry.exists():
        return False
    value = json.loads(registry.read_bytes())
    refs = value.get("proofs")
    if (
        set(value) != {"schema_version", "proofs"}
        or value["schema_version"] != "generic-execution-compatibility-registry/1.0.0"
        or not isinstance(refs, list)
        or not 1 <= len(refs) <= 16
    ):
        raise ValueError("generic_execution_compatibility_registry")

    def checked(reference):
        if not isinstance(reference, dict) or set(reference) != {"path", "checksum"}:
            raise ValueError("generic_execution_compatibility_reference")
        path = (root / reference["path"]).resolve()
        if not path.is_relative_to(root / "docs/acceptance"):
            raise ValueError("generic_execution_compatibility_evidence_scope")
        raw = path.read_bytes()
        if "sha256:" + hashlib.sha256(raw).hexdigest() != reference["checksum"]:
            raise ValueError("generic_execution_compatibility_evidence_changed")
        return raw

    matched = False
    for reference in refs:
        proof = json.loads(checked(reference))
        if (
            proof.get("schema_version") != "generic-execution-software-handoff/1.0.0"
            or proof.get("scope") != "software_compatibility_only_no_dispatch_authority"
            or proof.get("passed") is not True
            or proof.get("execution_authorized") is not False
            or set(proof.get("before", {})) != set(LOCK_FILES)
            or set(proof.get("after", {})) != set(LOCK_FILES)
            or not isinstance(proof.get("commands"), list)
            or not proof["commands"]
        ):
            raise ValueError("generic_execution_compatibility_proof")
        if (
            proof["before"] != spec.get("execution_lock_implementation")
            or proof["after"] != current
        ):
            continue
        frozen = json.loads(checked(proof["freeze"]))["files"]
        if any(frozen.get(name) != sha for name, sha in current.items()):
            raise ValueError("generic_execution_compatibility_uncovered_code")
        for command in proof["commands"]:
            receipt = json.loads(checked(command["receipt"]))
            code = checked(command["exit"]).decode().strip()
            log = checked(command["log"])
            if (
                receipt.get("direct_exit") != 0
                or code != "0"
                or receipt.get("log_checksum") != "sha256:" + hashlib.sha256(log).hexdigest()
            ):
                raise ValueError("generic_execution_compatibility_failed_checks")
        if matched:
            raise ValueError("generic_execution_compatibility_ambiguous")
        matched = True
    return matched


def verify_input(root, case, binding):
    spec = json.loads((case / "input.json").read_bytes())
    if (
        binding.get("schema_version") != "generic-case-binding/1.2.0"
        or spec.get("schema_version") not in INPUT_VERSIONS
        or spec.get("execution_lock_profile") != PROFILE
        or spec.get("key") != case.name
        or binding.get("execution_namespace") != case.name
        or binding.get("input_spec_checksum") != digest(spec)
        or (
            spec.get("execution_lock_implementation") != implementation(root)
            and not execution_compatibility(root, spec, implementation(root))
        )
    ):
        raise ValueError("generic_execution_lock_binding")
    return spec


@contextmanager
def resource_lock(root, *, private_root=None):
    # The optional path is an internal fixture port, never a runner CLI argument.
    local = private_root or Path("/tmp") / ("mmagent-segman-locks-" + str(os.getuid()))
    local.mkdir(mode=0o700, exist_ok=True)
    st = local.lstat()
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise ValueError("generic_resource_lock_directory")
    directory = os.open(local, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open("cuda0.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(fd, "a") as host:
        st = os.fstat(host.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
            raise ValueError("generic_resource_lock_file")
        fcntl.flock(host, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root / ".local/segman-component.lock").open("a") as legacy:
            try:
                fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno not in (errno.ENOLCK, errno.ENOSYS, errno.EOPNOTSUPP):
                    raise
                # Explicit profile uses Runtime + host locks, not an implicit unlocked retry.
                legacy_available = False
            else:
                legacy_available = True
            yield dict(host_lock=True, legacy_lock=legacy_available, profile=PROFILE)
