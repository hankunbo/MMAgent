"""Explicit refreshed local environment proof, not a timestamp edit of an old probe."""

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from packages.grounding.models import digest, seal, validate
from packages.tool_admission.models import EnvironmentProbe
from packages.tool_execution.trusted_process import NvidiaObservation, TrustedLocalProcessRunner
from packages.tool_workflow.compiler import local_path

from .deployment import InstalledProfiles
from .readiness import UnboundSingleCasePolicy


class CurrentFairCLIPProfiles(InstalledProfiles):
    """One persisted proof per new case; retries reuse it, never reset expiry."""

    def __init__(self, record, catalogs, *, proof_path):
        super().__init__(record, catalogs)
        self.proof_path = proof_path

    def read_probe(self, manifest, profile):
        from apps.vectorengine_single_case import write_once

        installed = self.current()
        if installed.get("startup_probe") not in {None, "fairclip-absolute-entry/1.0.0"}:
            raise ValueError("fairclip_startup_probe_version")
        if self.proof_path.exists():
            proof = json.loads(self.proof_path.read_bytes())
        else:
            policy = validate(UnboundSingleCasePolicy, installed["execution_policy"])
            reader = SimpleNamespace(policy=policy, observation=NvidiaObservation())
            TrustedLocalProcessRunner.check_capabilities(reader)
            for review in policy.reviews:
                TrustedLocalProcessRunner.verify_review(reader, review)
            TrustedLocalProcessRunner.gpu_ready(reader)
            script = (
                "import importlib.metadata as m,json,sys; import torch,numpy,PIL; "
                "print(json.dumps({'versions':{n:m.version(n) for n in "
                "('torch','numpy','pillow')},'executable':sys.executable}))"
            )
            result = subprocess.run(
                (
                    [
                        local_path(profile.runtime.interpreter_uri),
                        "-B",
                        policy.reviews[0].entrypoint.path,
                        "--check-runtime",
                    ]
                    if installed.get("startup_probe") == "fairclip-absolute-entry/1.0.0"
                    else [local_path(profile.runtime.interpreter_uri), "-I", "-B", "-c", script]
                ),
                capture_output=True,
                timeout=60,
                check=False,
                env={
                    "PATH": os.defpath,
                    "CUDA_VISIBLE_DEVICES": "",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            if result.returncode != 0 or len(result.stdout) > 8192:
                raise ValueError("fairclip_current_dependency_probe_failed")
            observed = json.loads(result.stdout)
            if observed["versions"] != profile.runtime.dependencies:
                raise ValueError("fairclip_current_dependency_versions")
            now = datetime.now(UTC)
            facts = dict(
                schema_version="fairclip-environment-observation/1.0.0",
                observed_at=now.isoformat(),
                installed_checksum=digest(installed),
                manifest_checksum=manifest.checksum,
                profile_checksum=profile.checksum,
                observed=observed,
                inventory_verified=True,
                gpu_observed=True,
                tool_forward_count=0,
            )
            source = installed["probe"]["source_ref"] | {
                "code": "SOURCE:FAIRCLIP:ENV:" + digest(facts)[7:23].upper(),
                "checksum": digest(facts),
            }
            probe = seal(
                EnvironmentProbe,
                version="1.0.0",
                manifest_checksum=manifest.checksum,
                profile_checksum=profile.checksum,
                runtime_checksum=digest(profile.runtime),
                source_ref=source,
                observed_at=now,
                expires_at=now + timedelta(minutes=45),
                interpreter="available",
                dependencies="available",
                gpu="available",
                resources="available",
                preconditions="satisfied",
            )
            proof = dict(facts=facts, probe=probe.model_dump(mode="json"))
            write_once(self.proof_path, proof)
        probe = validate(EnvironmentProbe, proof["probe"])
        if (
            proof["facts"]["installed_checksum"] != digest(installed)
            or probe.source_ref.checksum != digest(proof["facts"])
            or probe.manifest_checksum != manifest.checksum
            or probe.profile_checksum != profile.checksum
            or probe.runtime_checksum != digest(profile.runtime)
        ):
            raise ValueError("fairclip_current_probe_binding")
        # Original admission enforces its expiry. No automatic renewal on retry.
        return probe
