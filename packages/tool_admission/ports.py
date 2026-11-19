"""Service assembly ports. Callers cannot supply storage roots, DSNs or probes."""

from typing import Protocol

from packages.grounding.models import Ref
from packages.grounding.query_models import CaseScope
from packages.integrations.medgraph import DeploymentManifest

from .models import DeploymentProfile, EnvironmentProbe, InputRecord, ResearchRequest


class InputReader(Protocol):
    def read_input(self, scope: CaseScope, run_id: str, asset_code: str) -> InputRecord:
        """Exact ingestion/Runtime projection; KeyError means absent, I/O errors propagate."""
        ...


class ProfileReader(Protocol):
    def read_profile(self, manifest: DeploymentManifest) -> DeploymentProfile:
        """Deployment owner supplies versioned rules and sourced output target bindings.

        Bindings explicitly type the anatomical subset of exact output constraints;
        neither caller-provided targets nor a checksum establish those semantics.
        """
        ...


class ProbeReader(Protocol):
    def read_probe(
        self, manifest: DeploymentManifest, profile: DeploymentProfile
    ) -> EnvironmentProbe:
        """Current bounded probe facts, no model or subprocess in admission."""
        ...


class ResearchReader(Protocol):
    def read_request(self, code: str, version: str, checksum: str) -> ResearchRequest:
        """Controlled registration bound to existing scope, catalog, exact inputs and purpose."""
        ...


class ManifestPort(Protocol):
    def read_manifest(
        self, catalog_ref: Ref, tool_ref: Ref, implementation_ref: Ref
    ) -> DeploymentManifest: ...
