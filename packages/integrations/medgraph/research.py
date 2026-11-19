"""Deployment-owned, exact research overlays. Legacy projection is never mutated.

The assembly installs a pinned record, not a request-supplied approval. Audit files
are declarations by the deployment reviewer, never invented native database facts.
Current native safety and file integrity are checked again at each public read.
"""

import json
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import (
    ArtifactConstraint,
    CatalogSnapshot,
    CatalogTool,
    Ref,
    Value,
    check_seal,
    digest,
    seal,
    validate,
)
from packages.grounding.ports import ToolSafety
from packages.tool_execution.trusted_models import ReviewedFile
from packages.tool_execution.trusted_process import verify_file

from .models import AdapterError, NativeManifest, ReviewedDeploymentManifest
from .projection import adaptation_version
from .resources import LocalResourcePolicy, observe_manifest_resources


class GapResolution(Value):
    gap: str
    basis: str = Field(min_length=20, max_length=4096)
    evidence: ReviewedFile
    disposition: Literal["verified_for_shadow", "formal_gate_deferred_shadow_only"]


class ResearchReview(Value):
    version: Literal["medgraph-research-review/1.0.0"] = "medgraph-research-review/1.0.0"
    checksum: str
    owner: Literal["06E_DEPLOYMENT_REVIEWER"] = "06E_DEPLOYMENT_REVIEWER"
    origin: Literal["deployment_declaration_not_native"] = "deployment_declaration_not_native"
    source_catalog_ref: Ref
    source_implementation_ref: Ref
    source_manifest_checksum: str
    config_checksum: str
    research_only: Literal[True] = True
    decisive_evidence: Literal[False] = False
    purposes: tuple[Literal["debug", "shadow"], ...] = ("debug", "shadow")
    resolutions: tuple[GapResolution, ...] = Field(min_length=1, max_length=32)
    input_constraints: tuple[ArtifactConstraint, ...] = Field(min_length=1, max_length=64)
    output_constraints: tuple[ArtifactConstraint, ...] = Field(min_length=1, max_length=64)
    deployment_declaration: NativeManifest
    parameter_schema: dict
    preprocessing_policy: dict
    postprocessing_policy: dict
    interpreter: ReviewedFile
    # All native resources are verified separately, including missing native hashes.
    native_resource_evidence: ReviewedFile

    @model_validator(mode="after")
    def coherent(self):
        check_seal(self)
        if any(
            f.byte_size == 0
            for f in (
                self.interpreter,
                self.native_resource_evidence,
                *(g.evidence for g in self.resolutions),
            )
        ):
            raise ValueError("research_review_evidence_must_not_be_empty")
        if len({g.gap for g in self.resolutions}) != len(self.resolutions):
            raise ValueError("duplicate_gap_resolution")
        for g in self.resolutions:
            if (g.gap == "validation_gates_missing") != (
                g.disposition == "formal_gate_deferred_shadow_only"
            ):
                raise ValueError("formal_gate_is_not_closed_by_engineering_review")
        return self


def reviewed_ref(review):
    return Ref(
        kind="source_version",
        code="SOURCE:06E:RESEARCH_REVIEW",
        version="1.0.0",
        checksum=review.checksum,
    )


class ReviewedCatalog:
    """Install only in trusted application assembly; request APIs cannot install it."""

    def __init__(self, legacy, record: ReviewedFile, resources: LocalResourcePolicy):
        self.legacy, self.record, self.resources = legacy, record, resources
        self.review = self._load()
        review = self.review
        declaration = review.preprocessing_policy.get("metadata_requirements")
        if declaration is not None:
            from packages.tool_admission.method_metadata import VERSION, validate_review_metadata
            from packages.tool_admission.research_metadata import FAIRCLIP_METADATA

            if isinstance(declaration, dict) and declaration.get("schema_version") == VERSION:
                validate_review_metadata(review)
            elif declaration != FAIRCLIP_METADATA or not set(declaration["purposes"]) <= set(
                review.purposes
            ):
                raise AdapterError("research_metadata_review_purpose_mismatch")
        old = legacy.read_snapshot(review.source_catalog_ref)
        if old.synthetic:
            raise AdapterError("research_native_catalog_required")
        tool = next(
            (t for t in old.tools if t.implementation_ref == review.source_implementation_ref), None
        )
        if tool is None:
            raise AdapterError("review_source_identity_missing")
        manifest = legacy.read_manifest(old.reference(), tool.tool_ref, tool.implementation_ref)
        if (
            manifest.checksum != review.source_manifest_checksum
            or manifest.config_checksum != review.config_checksum
        ):
            raise AdapterError("review_source_version_mismatch")
        if (
            manifest.native_manifest is None
            or manifest.native_manifest.validation_status != "smoke_tested"
        ):
            raise AdapterError("review_requires_native_smoke_tested")
        if set(manifest.binding.gaps) != {g.gap for g in review.resolutions}:
            raise AdapterError("review_gap_evidence_incomplete")
        for constraints, original in (
            (review.input_constraints, tuple(i.constraint for i in tool.inputs)),
            (review.output_constraints, tool.outputs),
        ):
            if len(constraints) != len(original) or any(
                a.artifact_ref != b.artifact_ref for a, b in zip(constraints, original, strict=True)
            ):
                raise AdapterError("review_native_port_type_changed")
        reference = tool.implementation_ref.model_copy(
            update={
                "version": adaptation_version(review.checksum),
                "checksum": digest(dict(native=tool.implementation_ref, review=review.checksum)),
            }
        )
        self.original_manifest = manifest
        self.tool = validate(
            CatalogTool,
            {
                **tool.model_dump(mode="json"),
                "implementation_ref": reference,
                "enabled": True,
                "catalog_status": "active",
                "validation_status": "smoke_tested",
                "inputs": [
                    i.model_copy(update={"constraint": c})
                    for i, c in zip(tool.inputs, review.input_constraints, strict=True)
                ],
                "outputs": review.output_constraints,
            },
        )
        self.catalog = seal(
            CatalogSnapshot,
            code=old.code + ":RESEARCH",
            version=adaptation_version(review.checksum),
            synthetic=False,
            manifest_checksum=digest(dict(native=old.reference(), review=review.checksum)),
            artifacts=old.artifacts,
            tools=(self.tool,),
            estimates=(),
        )
        self.manifest = seal(
            ReviewedDeploymentManifest,
            **{
                k: v
                for k, v in manifest.model_dump(mode="json").items()
                if k
                not in {
                    "schema_version",
                    "checksum",
                    "catalog_ref",
                    "implementation_ref",
                    "binding",
                    "invocation",
                    "parameter_schema",
                    "preprocessing_policy",
                    "postprocessing_policy",
                    "policy_checksum",
                }
            },
            catalog_ref=self.catalog.reference(),
            implementation_ref=reference,
            binding=manifest.binding.model_copy(
                update={"implementation_ref": reference, "gaps": ()}
            ),
            invocation=review.deployment_declaration.invocation,
            parameter_schema=review.parameter_schema,
            preprocessing_policy=review.preprocessing_policy,
            postprocessing_policy=review.postprocessing_policy,
            policy_checksum=review.checksum,
            review_ref=reviewed_ref(review),
            original_manifest_checksum=manifest.checksum,
            deployment_declaration=review.deployment_declaration,
        )
        self._verify_resources()
        self.current_safety(self.catalog.reference())

    def _load(self):
        verify_file(self.record, deadline=time.monotonic() + 10, limit=1048576)
        return ResearchReview.model_validate_json(Path(self.record.path).read_text())

    def _verify_resources(self):
        review = self.review
        from packages.tool_admission.method_metadata import VERSION, validate_review_metadata

        declaration = review.preprocessing_policy.get("metadata_requirements")
        if isinstance(declaration, dict) and declaration.get("schema_version") == VERSION:
            metadata = validate_review_metadata(review)
            verify_file(metadata.evidence, deadline=time.monotonic() + 10, limit=1048576)
        for f in (
            review.interpreter,
            review.native_resource_evidence,
            *(g.evidence for g in review.resolutions),
        ):
            verify_file(
                f,
                deadline=time.monotonic() + 10,
                limit=1048576 if f != review.interpreter else 67108864,
            )
        observed = observe_manifest_resources(self.original_manifest, self.resources)
        expected = json.loads(Path(review.native_resource_evidence.path).read_text())
        if observed.model_dump(mode="json") != expected:
            raise AdapterError("review_native_resource_drift")
        # Known native hashes cannot be superseded by local review measurements.
        if any(
            o.declared_checksum_matches is False or o.declared_size_matches is False
            for o in observed.observations
        ):
            raise AdapterError("review_native_resource_mismatch")
        deployed = observe_manifest_resources(self.manifest, self.resources)
        if any(
            o.declared_checksum_matches is not True or o.declared_size_matches is not True
            for o in deployed.observations
        ):
            raise AdapterError("review_deployment_resource_mismatch")

    def read_snapshot(self, reference):
        if reference != self.catalog.reference() or self._load() != self.review:
            raise AdapterError("uninstalled_review_or_catalog")
        self._verify_resources()
        return self.catalog

    def read_manifest(self, catalog_ref, tool_ref, implementation_ref):
        self.read_snapshot(catalog_ref)
        if (tool_ref, implementation_ref) != (self.tool.tool_ref, self.tool.implementation_ref):
            raise AdapterError("review_exact_manifest_required")
        return self.manifest

    def current_safety(self, reference):
        if reference != self.catalog.reference() or self._load() != self.review:
            raise AdapterError("uninstalled_review_or_catalog")
        native = self.legacy.current_safety(self.review.source_catalog_ref)
        disabled = self.review.source_implementation_ref in native.disabled_implementations
        return ToolSafety(
            catalog_ref=reference,
            disabled_implementations=(self.tool.implementation_ref,) if disabled else (),
            revision=native.revision,
        )


class ReviewedCatalogSet:
    """A new frozen catalog of installed reviews; never edit an existing run lock."""

    def __init__(self, components):
        self.components = tuple(components)
        components = self.components
        if not self.components or len({c.review.source_catalog_ref for c in components}) != 1:
            raise AdapterError("one_native_snapshot_required")
        tools = tuple(c.tool for c in components)
        if len({t.implementation_ref for t in tools}) != len(tools):
            raise AdapterError("duplicate_installed_implementation")
        versions = tuple(c.review.checksum for c in components)
        self.catalog = seal(
            CatalogSnapshot,
            code="CATALOG:06E:INSTALLED_RESEARCH",
            version=adaptation_version(digest(versions)),
            synthetic=False,
            manifest_checksum=digest(versions),
            artifacts=components[0].catalog.artifacts,
            tools=tools,
            estimates=(),
        )

    def read_snapshot(self, reference):
        if reference != self.catalog.reference():
            raise AdapterError("uninstalled_catalog_set")
        for c in self.components:
            c.read_snapshot(c.catalog.reference())
        return self.catalog

    def read_manifest(self, catalog_ref, tool_ref, implementation_ref):
        self.read_snapshot(catalog_ref)
        matches = [
            c
            for c in self.components
            if (c.tool.tool_ref, c.tool.implementation_ref) == (tool_ref, implementation_ref)
        ]
        if len(matches) != 1:
            raise AdapterError("review_exact_manifest_required")
        m = matches[0].manifest
        return seal(
            ReviewedDeploymentManifest,
            **{
                **{k: v for k, v in m.model_dump(mode="json").items() if k != "checksum"},
                "catalog_ref": catalog_ref,
            },
        )

    def current_safety(self, reference):
        if reference != self.catalog.reference():
            raise AdapterError("uninstalled_catalog_set")
        observations = [c.current_safety(c.catalog.reference()) for c in self.components]
        return ToolSafety(
            catalog_ref=reference,
            disabled_implementations=tuple(
                i for o in observations for i in o.disabled_implementations
            ),
            revision=max(o.revision for o in observations),
        )
