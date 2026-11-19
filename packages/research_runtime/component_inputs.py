"""Original visible component → CASE input Artifact → registered 06 request.

This is a recoverable projection-first registration, not tool qualification or
permission to launch a child run. The campaign execution bridge must separately
bind current control/deadline/budget. No source dataset, label store or conversion
is read here; old single-SLO registration remains unchanged.
"""

import math
from typing import Literal

from pydantic import Field, model_validator

from db.runtime.repository.case_pipeline import identity
from db.runtime.repository.ports import IdempotencyConflictError
from packages.contracts import Artifact
from packages.dataset_adapters.multimodal.models import Component, Payload
from packages.dataset_adapters.multimodal.storage import read_input
from packages.grounding.models import (
    Candidate,
    Checksum,
    Code,
    Policy,
    Ref,
    Term,
    Value,
    check_seal,
    digest,
    unique,
    validate,
)
from packages.grounding.query_models import ArtifactMetadata, CaseScope
from packages.object_store import checksum
from packages.tool_admission.inputs import image_header
from packages.tool_admission.models import ArtifactTypeBinding, InputRecord, InvocationInput

from .registration import VisibleResearchRequests
from .task_context import TaskRunContext, check_current_inputs


class ComponentInputPolicy(Value):
    """Installed mapping, not a caller-supplied authorization flag."""

    schema_version: Literal["research-component-input-policy/1.0.0"] = (
        "research-component-input-policy/1.0.0"
    )
    checksum: Checksum
    source_kind: Literal["image", "map", "volume", "vector"]
    source_concept: str = Field(min_length=1, max_length=128)
    source_modality: str = Field(min_length=1, max_length=128)
    source_dtype: str = Field(min_length=1, max_length=64)
    input_binding: ArtifactTypeBinding
    modality: Term
    scope_terms: tuple[Code, ...] = Field(max_length=128)

    @model_validator(mode="after")
    def coherent(self):
        unique(self.scope_terms)
        if (
            self.input_binding.contract_unit is None
            and self.input_binding.unit_term.state == "known"
        ):
            raise ValueError("component_policy_unit")
        check_seal(self)
        return self


def check_component(component, payload, raw, policy):
    """Exact bounded header/value checks, not image decoding or medical QC."""
    c = validate(Component, component)
    p = validate(Payload, payload)
    policy = validate(ComponentInputPolicy, policy)
    if c.status != "obtained":
        raise ValueError("component_input_unavailable")
    if (c.kind, c.concept, c.modality, c.dtype) != (
        policy.source_kind,
        policy.source_concept,
        policy.source_modality,
        policy.source_dtype,
    ):
        raise ValueError("component_input_mapping")
    if c.payload_ids != (p.id,):
        raise ValueError("component_input_payload")
    if c.unit != policy.input_binding.contract_unit:
        raise ValueError("component_input_unit")
    if p.byte_size > 4_194_304:
        raise ValueError("component_input_size")
    expected_media = "image/jpeg" if c.kind == "image" else "application/x-npy"
    if p.media_type != expected_media:
        raise ValueError("component_input_media")
    if len(raw) != p.byte_size or checksum(raw) != p.checksum:
        raise ValueError("component_input_bytes")
    media, shape = image_header(raw, 65536)
    if media != expected_media or shape != c.shape:
        raise ValueError("component_input_shape")
    if c.kind == "image":
        from packages.tool_benchmark.formats import jpeg_dimensions

        if jpeg_dimensions(raw) != c.shape:
            raise ValueError("component_input_shape")
        return "image", "PIXEL_2D"
    from packages.dataset_adapters.multimodal.formats import npy

    array = npy(raw)
    rank, coordinate = {
        "map": (2, "PIXEL_2D"),
        "volume": (3, "VOXEL_3D"),
        "vector": (1, "NOT_APPLICABLE"),
    }[c.kind]
    if array.dtype != c.dtype or len(shape) != rank:
        raise ValueError("component_input_shape_dtype")
    if not all(math.isfinite(x) for x in array.numbers()):
        raise ValueError("component_input_nonfinite")
    return "array", coordinate


class ComponentResearchRequests(VisibleResearchRequests):
    """One installed component/method binding, immutable parent and request key.

    Projection + binding receipt + request use original, separate transactions.
    An interruption can leave a reusable input projection but cannot make it an
    executed tool. Current reads rebuild through the original input owner.
    """

    def __init__(
        self,
        *,
        runtime,
        inputs,
        evidence,
        documents,
        parent_run_id,
        component_id,
        request_key,
        component_policy,
        candidate,
        invocation,
        catalog_ref,
        policy,
    ):
        if type(request_key) is not str or not 1 <= len(request_key) <= 128:
            raise ValueError("research_key_invalid")
        if type(component_id) is not str or not 1 <= len(component_id) <= 128:
            raise ValueError("component_selector_invalid")
        candidate = validate(Candidate, candidate)
        invocation = validate(InvocationInput, invocation)
        catalog_ref = validate(Ref, catalog_ref)
        policy = validate(Policy, policy)
        self.inputs, self.evidence, self.documents = inputs, evidence, documents
        self.parent = validate(TaskRunContext, runtime.read_research_run(parent_run_id))
        self.component_id, self.request_key = component_id, request_key
        self.component_policy = validate(ComponentInputPolicy, component_policy)
        if catalog_ref != self.parent.request.lock.catalog_ref:
            raise ValueError("component_parent_catalog")
        if len(candidate.inputs) != 1:
            raise ValueError("component_single_port_required")
        if invocation.implementation_ref != candidate.implementation_ref:
            raise ValueError("component_implementation_mismatch")
        self.registration_binding = dict(
            version="research-component-registration/1.0.0",
            request_key_checksum=digest(request_key),
            parent=self.parent.checksum,
            component_id=component_id,
            component_policy=self.component_policy,
            candidate=candidate,
            invocation=invocation,
            catalog=catalog_ref,
            policy=policy,
        )
        super().__init__(
            runtime=runtime,
            visible_reader=None,
            objects=inputs.visible_store,
            candidate=candidate,
            invocation=invocation,
            catalog_ref=catalog_ref,
            policy=policy,
            input_binding=self.component_policy.input_binding,
            modality=self.component_policy.modality,
            scope_terms=self.component_policy.scope_terms,
            config_checksum=digest(self.registration_binding),
        )
        # Assembly is not admission. A confirmed attempt must remain recoverable
        # if its original input bytes later disappear. Keep static owner identity
        # here; every registration/current-input path still calls _current().
        if self.component_id not in self.parent.request.selected_components:
            raise ValueError("component_not_selected")
        if self.documents.run_id != self.parent.run_id:
            raise ValueError("component_document_scope")
        root = self.documents.root
        if root.scope.case_code != self.parent.request.scope.case_code or not any(
            (a.asset_code, a.checksum, a.artifact_type_code, a.media_type)
            == (root.asset_code, root.checksum, root.artifact_type_code, root.media_type)
            for a in self.parent.request.artifacts
        ):
            raise ValueError("component_document_scope")

    @staticmethod
    def request_code(key):
        return "RESEARCH:COMPONENT:" + digest(key)[7:39].upper()

    def _current(self, *, include_components=False):
        current = validate(TaskRunContext, self.runtime.read_research_run(self.parent.run_id))
        if current != self.parent:
            raise ValueError("component_parent_changed")
        checked = check_current_inputs(current, self.inputs, include_components=include_components)
        visible = checked[0] if include_components else checked
        if (
            self.documents.run_id != current.run_id
            or self.documents.root != visible.case.artifacts[0]
        ):
            raise ValueError("component_document_scope")
        if self.component_id not in current.request.selected_components:
            raise ValueError("component_not_selected")
        return checked

    def _project(self, visible, *, stored=None):
        c = next(c for c in visible.components.components if c.id == self.component_id)
        if c.status != "obtained" or len(c.payload_ids) != 1:
            raise ValueError("component_input_unavailable_or_series")
        root = visible.case.artifacts[0]
        if c.laterality != root.laterality or c.laterality != visible.case.laterality:
            raise ValueError("component_eye_mismatch")
        payload = next(p for p in visible.components.payloads if p.id == c.payload_ids[0])
        # Internal callers can use the original owner's just-completed full
        # read. Nothing is cached on this object or reused across public calls.
        if stored is None:
            stored = read_input(visible, self.objects)
        if stored.components != visible.components:
            raise ValueError("component_current_storage_binding")
        blob = next(b for b in stored.blobs if b.id == payload.id)
        if blob.byte_size > 4_194_304 or len(blob.chunks) != 1:
            raise ValueError("component_input_size")
        chunk = blob.chunks[0]
        raw = self.objects.read_bounded(chunk.uri, 4_194_304)
        kind, coordinate = check_component(c, payload, raw, self.component_policy)
        binding = digest(
            dict(visible=visible.checksum, component=c, registration=self.registration_binding)
        )
        artifact = validate(
            Artifact,
            dict(
                root.model_dump(mode="json"),
                asset_code=identity("ASSET", "component-input:" + binding),
                artifact_type_code=self.input_binding.contract_type_code,
                uri=chunk.uri,
                checksum=payload.checksum,
                media_type=payload.media_type,
                dimensions=c.shape,
                coordinate_space=coordinate,
                unit=self.input_binding.contract_unit,
                qc_status="UNASSESSED",
                derived_from=(root.asset_code,),
                provenance=dict(
                    root.provenance.model_dump(mode="json"),
                    generator_version="component-input.1.0.0",
                    input_checksums=(root.checksum, binding),
                ),
            ),
        )
        metadata = ArtifactMetadata(
            asset_code=artifact.asset_code,
            checksum=artifact.checksum,
            case_code=visible.case.case_code,
            patient_hash=visible.case.patient_hash,
            encounter_code=visible.case.encounter_code,
            run_id=None,
            artifact_ref=self.input_binding.artifact_ref,
            modality=self.modality,
            laterality=artifact.laterality,
            coordinate_space=artifact.coordinate_space,
            unit=self.input_binding.unit_term,
            scope_terms=self.scope_terms,
            qc_status=artifact.qc_status,
            acquired_at=None,
        )
        record = InputRecord(
            artifact=artifact,
            type_binding=self.input_binding,
            metadata=metadata,
            byte_size=len(raw),
            content_kind=kind,
            source_ref=self.input_binding.source_ref,
        )
        receipt = dict(
            schema_version="research-component-registration/1.0.0",
            binding=self.registration_binding,
            visible_checksum=visible.checksum,
            input_checksum=visible.components.checksum,
            component=c,
            payload=payload,
            input_record=record,
            request_code=self.request_code(self.request_key),
            tool_qualified=False,
            execution_authorized=False,
        )
        return record, receipt

    def _receipt_key(self):
        return "component-input:" + self.request_code(self.request_key)

    def _records(self, visible):
        current, stored = self._current(include_components=True)
        if visible != current:
            raise ValueError("component_visible_changed")
        return self._projected_records(visible, stored=stored)

    def _projected_records(self, visible, *, stored=None):
        """Private projection after this owner has just read current inputs.

        No cached qualification or caller-owned validated flag. All chunks are
        checked by the input owner; persistence and receipt bindings remain here.
        """
        record, receipt = self._project(visible, stored=stored)
        if (
            self.evidence.read_case_artifact(visible.case.case_code, record.artifact.asset_code)
            != record.artifact
        ):
            raise ValueError("component_projection_changed")
        import json

        from packages.grounding.models import canonical

        if self.documents.read(self._receipt_key()) != (json.loads(canonical(receipt)),):
            raise ValueError("component_receipt_changed")
        return (record,)

    def register(self, visible, key):
        if key != self.request_key:
            raise ValueError("component_registration_identity")
        current, stored = self._current(include_components=True)
        if visible != current:
            raise ValueError("component_registration_identity")
        record, receipt = self._project(visible, stored=stored)
        # Detect an existing conflicting request before adding a new projection.
        try:
            old = super().for_key(key)
        except ValueError as error:
            if str(error) != "research_registration_pending":
                raise
        else:
            if (
                old.artifacts != (record.metadata,)
                or old.candidate != self.candidate
                or old.invocation != self.invocation
            ):
                raise IdempotencyConflictError("research_registration_conflict")
            return self.read_request(old.code, old.version, old.checksum)
        self.evidence.register_observation_bundle(
            (record.artifact,),
            # First committed projection owns the global request key, including
            # the window before receipt/request registration. Its original owner
            # rejects any changed parent/mapping/content without a partial write.
            identity("TASK", "component-projection:" + self.request_code(key)),
        )
        with self.documents.session():
            self.documents.write(self._receipt_key(), (receipt,))
        request = super().register(visible, key)
        return self.read_request(request.code, request.version, request.checksum)

    def read_request(self, code, version, checksum):
        return self._read_with_record(code, version, checksum)[0]

    def _read_with_record(self, code, version, checksum):
        if code != self.request_code(self.request_key):
            raise ValueError("component_registration_identity")
        request = super().read_request(code, version, checksum)
        visible, stored = self._current(include_components=True)
        record = self._projected_records(visible, stored=stored)[0]
        expected_run = identity("RUN", "research:" + code)
        if (
            request.artifacts != (record.metadata,)
            or request.candidate != self.candidate
            or request.invocation != self.invocation
            or request.lock.catalog_ref != self.catalog_ref
            or request.compatibility_policy != self.policy
            or request.lock.run_id != expected_run
            or request.scope != self.parent.request.scope
            or request.purpose != "shadow"
            or request.satisfied_terms != self.scope_terms
            or len(request.inputs) != 1
            or request.inputs[0].port != self.candidate.inputs[0].port
            or request.inputs[0].asset_codes != (record.artifact.asset_code,)
        ):
            raise ValueError("component_request_changed")
        return request, record

    def for_key(self, key):
        if key != self.request_key:
            raise ValueError("component_registration_identity")
        value = super().for_key(key)
        return self.read_request(value.code, value.version, value.checksum)

    def read_input(self, scope, run_id, asset_code):
        saved = super().for_key(self.request_key)
        request, record = self._read_with_record(saved.code, saved.version, saved.checksum)
        if validate(CaseScope, scope) != request.scope or run_id != request.lock.run_id:
            raise ValueError("component_input_scope")
        if record.artifact.asset_code != asset_code:
            raise ValueError("component_input_identity")
        return record
