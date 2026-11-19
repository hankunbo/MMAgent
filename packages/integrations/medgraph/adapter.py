"""Immutable derived storage and the existing ToolCatalogReader implementation."""

import os
import tempfile
from pathlib import Path
from typing import Protocol

from packages.grounding.models import Ref, canonical, digest, seal
from packages.grounding.ports import ToolSafety

from .models import (
    AdapterError,
    DeploymentManifest,
    NativeSnapshot,
    ProjectionPolicy,
    SnapshotEnvelope,
    parse,
)
from .projection import implementation_content, normalized, project
from .retirement import RETIREMENT_VERSION, retired


class NativeReader(Protocol):
    def capture(self) -> NativeSnapshot: ...


class ManifestReader(Protocol):
    def read_manifest(
        self, catalog_ref: Ref, tool_ref: Ref, implementation_ref: Ref
    ) -> DeploymentManifest: ...


class SnapshotStore:
    """Append-only content-addressed files, published atomically without overwrite.

    The directory is a trusted deployment store, never a caller-selected resource URI.
    Re-derive on every read: there is no separately editable tool master/projection.
    """

    def __init__(self, root: Path, *, max_bytes=16777216):
        self.root = root.resolve(strict=True)
        self.max_bytes = max_bytes
        if not self.root.is_dir():
            raise AdapterError("snapshot_directory_required")

    def _path(self, reference):
        reference = parse(Ref, reference)
        if reference.kind != "catalog":
            raise AdapterError("catalog_kind_required")
        return self.root / (reference.checksum[7:] + ".json")

    def save(self, envelope):
        envelope = parse(SnapshotEnvelope, envelope)
        self._verify(envelope)
        raw = canonical(envelope)
        if len(raw) > self.max_bytes:
            raise AdapterError("snapshot_byte_limit")
        destination = self._path(envelope.catalog.reference())
        fd, name = tempfile.mkstemp(prefix=".derived-", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fchmod(stream.fileno(), 0o444)
                os.fsync(stream.fileno())
            try:
                os.link(name, destination)
            except FileExistsError:
                if self.read(envelope.catalog.reference()) != envelope:
                    raise AdapterError("immutable_snapshot_conflict") from None
            directory = os.open(self.root, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.unlink(name)
        return envelope.catalog.reference()

    def _verify(self, envelope):
        expected = project(
            envelope.native, envelope.config, envelope.policy, synthetic=envelope.catalog.synthetic
        )
        if expected != envelope:
            raise AdapterError("derived_projection_mismatch")

    def read(self, reference):
        path = self._path(reference)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as stream:
                raw = stream.read(self.max_bytes + 1)
        except OSError:
            raise AdapterError("exact_catalog_not_found") from None
        if len(raw) > self.max_bytes:
            raise AdapterError("snapshot_byte_limit")
        import json

        try:
            envelope = parse(SnapshotEnvelope, json.loads(raw))
        except ValueError:
            raise AdapterError("invalid_snapshot") from None
        if envelope.catalog.reference() != reference:
            raise AdapterError("exact_catalog_mismatch")
        self._verify(envelope)
        return envelope


class MedgraphCatalogReader:
    def __init__(self, source: NativeReader, store: SnapshotStore, config, policy):
        from .models import AdapterConfig

        self.source = source
        self.store = store
        self.config = parse(AdapterConfig, config)
        self.policy = parse(ProjectionPolicy, policy)

    def capture_snapshot(self, *, synthetic=False):
        envelope = project(self.source.capture(), self.config, self.policy, synthetic=synthetic)
        self.store.save(envelope)
        return envelope.catalog

    def read_snapshot(self, reference):
        return self.store.read(reference).catalog

    def read_native(self, reference):
        """Bounded audit/deployment port, not an Agent context (contains local IDs)."""
        return self.store.read(reference).native

    def list_types(self, reference):
        return self.read_snapshot(reference).artifacts

    def list_tools(self, reference, *, capability_ref=None, tool_ref=None):
        catalog = self.read_snapshot(reference)
        if capability_ref is not None:
            capability_ref = parse(Ref, capability_ref)
        if tool_ref is not None:
            tool_ref = parse(Ref, tool_ref)
        return tuple(
            t
            for t in catalog.tools
            if (capability_ref is None or capability_ref in t.capabilities)
            and (tool_ref is None or tool_ref == t.tool_ref)
        )

    def list_ports(self, reference, tool_ref):
        tools = self.list_tools(reference, tool_ref=tool_ref)
        if not tools:
            raise AdapterError("exact_tool_not_found")
        return tools[0].inputs, tuple(zip(tools[0].output_ports, tools[0].outputs, strict=True))

    def read_workflow(self, reference, workflow_ref):
        workflow_ref = parse(Ref, workflow_ref)
        envelope = self.store.read(reference)
        from .projection import workflow_reference

        for workflow in envelope.native.workflows:
            if (
                workflow_reference(envelope.native, workflow, envelope.config.source_code)
                == workflow_ref
            ):
                return (
                    workflow,
                    tuple(s for s in envelope.native.steps if s.workflow_id == workflow.id),
                    tuple(e for e in envelope.native.edges if e.workflow_id == workflow.id),
                )
        raise AdapterError("exact_workflow_not_found")

    def read_manifest(self, catalog_ref, tool_ref, implementation_ref):
        envelope = self.store.read(catalog_ref)
        tool_ref, implementation_ref = parse(Ref, tool_ref), parse(Ref, implementation_ref)
        bindings = [
            b
            for b in envelope.bindings
            if b.tool_ref == tool_ref and b.implementation_ref == implementation_ref
        ]
        if len(bindings) != 1:
            raise AdapterError("exact_implementation_not_found")
        binding = bindings[0]
        if retired(binding.tool_code, binding.native_version, binding.implementation_key):
            raise AdapterError("implementation_withdrawn_by_project")
        manifest = next(
            (
                m
                for m in envelope.native.manifests
                if (m.tool_code, m.version, m.implementation_key)
                == (binding.tool_code, binding.native_version, binding.implementation_key)
            ),
            None,
        )
        version = next(v for v in envelope.native.versions if v.id == binding.tool_version_id)
        return seal(
            DeploymentManifest,
            catalog_ref=catalog_ref,
            tool_ref=tool_ref,
            implementation_ref=implementation_ref,
            binding=binding,
            native_manifest=manifest,
            invocation=manifest.invocation if manifest else None,
            parameter_schema=version.metadata.get("parameter_schema"),
            preprocessing_policy=version.metadata.get("preprocessing_policy"),
            postprocessing_policy=version.metadata.get("postprocessing_policy"),
            native_manifest_state="visible" if manifest else "disabled_in_native_function",
            ports=tuple(
                p for p in envelope.native.ports if p.tool_version_id == binding.tool_version_id
            ),
            config_checksum=digest(envelope.config),
            policy_checksum=digest(envelope.policy),
        )

    def current_safety(self, reference):
        envelope = self.store.read(reference)
        # A failed/partial query raises, never an empty revocation set.
        current = normalized(self.source.capture())
        disabled = []
        identities = {i.id: i for i in current.implementations}
        for binding in envelope.bindings:
            impl = identities.get(binding.implementation_id)
            unsafe = (
                envelope.config != self.config
                or envelope.policy != self.policy
                or impl is None
                or retired(binding.tool_code, binding.native_version, binding.implementation_key)
            )
            if impl is not None:
                content = implementation_content(current, impl)
                unsafe = unsafe or (
                    digest(content) != binding.content_checksum
                    or not impl.enabled
                    or not content["version"].enabled
                    or impl.validation_status not in {"smoke_tested", "validated"}
                )
            if unsafe:
                disabled.append(binding.implementation_ref)
        # Native has no monotone safety revision. This is a deterministic content revision,
        # not an invented native sequence or authorization token.
        revision = int(
            digest(
                {"native": current.model_dump(mode="json"), "retirement_policy": RETIREMENT_VERSION}
            )[7:],
            16,
        )
        return ToolSafety(
            catalog_ref=reference, disabled_implementations=tuple(disabled), revision=revision
        )
