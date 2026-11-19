"""Exact deployment-installed research records. Caller checksums are not grants."""

import time
from pathlib import Path

from packages.grounding.models import validate
from packages.tool_admission.models import (
    DeploymentProfile,
    EnvironmentProbe,
    InputRecord,
    ResearchRequest,
)
from packages.tool_execution.trusted_process import verify_file
from packages.tool_workflow.models import CommandAllowlist


class InstalledRecords:
    def __init__(self, *, profiles=(), probes=(), inputs=(), requests=(), allowlists=()):
        self._files = dict(
            profiles=profiles,
            probes=probes,
            inputs=inputs,
            requests=requests,
            allowlists=allowlists,
        )

    def _read(self, kind, model):
        result = []
        for f in self._files[kind]:
            verify_file(f, deadline=time.monotonic() + 5, limit=16777216)
            result.append(validate(model, model.model_validate_json(Path(f.path).read_text())))
        return result

    @staticmethod
    def _one(values):
        if len(values) != 1:
            raise KeyError("exact_installed_record_required")
        return values[0]

    def read_profile(self, manifest):
        return self._one(
            [
                v
                for v in self._read("profiles", DeploymentProfile)
                if v.manifest_checksum == manifest.checksum
            ]
        )

    def read_probe(self, manifest, profile):
        return self._one(
            [
                v
                for v in self._read("probes", EnvironmentProbe)
                if (v.manifest_checksum, v.profile_checksum)
                == (manifest.checksum, profile.checksum)
            ]
        )

    def read_input(self, scope, run_id, code):
        return self._one(
            [
                v
                for v in self._read("inputs", InputRecord)
                if v.artifact.asset_code == code
                and v.metadata.case_code == scope.case_code
                and v.metadata.patient_hash == scope.patient_hash
                and v.metadata.encounter_code == scope.encounter_code
                and v.metadata.run_id == run_id
            ]
        )

    def read_request(self, code, version, checksum):
        return self._one(
            [
                v
                for v in self._read("requests", ResearchRequest)
                if (v.code, v.version, v.checksum) == (code, version, checksum)
            ]
        )

    def read_allowlist(self, manifest, profile, config):
        return self._one(
            [
                v
                for v in self._read("allowlists", CommandAllowlist)
                if (v.manifest_checksum, v.profile_checksum, v.compiler_config_checksum)
                == (manifest.checksum, profile.checksum, config.checksum)
            ]
        )
