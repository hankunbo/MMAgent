"""Multimodal inputs in the ORIGINAL case/Artifact/object/idempotency repositories.

Large arrays are chunked under the existing 4 MiB object boundary. Registration
is input persistence, not clinical interpretation or tool/model authorization.
"""

import json
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

import psycopg
from pydantic import model_validator

from db.runtime.repository.ports import IdempotencyConflictError
from packages.contracts import Artifact, CaseBundle, Provenance, ScopeReference
from packages.dataset_adapters.models import ObjectReference, Value
from packages.dataset_adapters.prepared import UNKNOWN_PATIENT
from packages.dataset_adapters.reading import AdapterError
from packages.disease_packs.models import SealedDeclaration, seal
from packages.object_store import checksum

from .models import CaseComponents


class StoredPayload(Value):
    id: str
    checksum: str
    byte_size: int
    chunks: tuple[ObjectReference, ...]

    @model_validator(mode="after")
    def length(self):
        if not self.chunks or sum(c.byte_size for c in self.chunks) != self.byte_size:
            raise ValueError("multimodal_blob_chunks")
        return self


class StoredInput(SealedDeclaration):
    schema_version: Literal["multimodal-input-object/1.0.0"] = "multimodal-input-object/1.0.0"
    components: CaseComponents
    blobs: tuple[StoredPayload, ...]

    @model_validator(mode="after")
    def bindings(self):
        expected = {p.id: (p.checksum, p.byte_size) for p in self.components.payloads}
        actual = {p.id: (p.checksum, p.byte_size) for p in self.blobs}
        if expected != actual or len(actual) != len(self.blobs):
            raise ValueError("multimodal_stored_payload_binding")
        return self


class MultimodalVisible(SealedDeclaration):
    schema_version: Literal["visible-input/3.0.0"] = "visible-input/3.0.0"
    adapter_version: Literal[
        "multimodal-adapters/1.0.0",
        "multimodal-adapters/1.1.0",
        "multimodal-adapters/1.2.0",
        "multimodal-adapters/1.3.0",
    ] = "multimodal-adapters/1.0.0"
    case: CaseBundle
    components: CaseComponents
    input_object: ObjectReference
    inference_ready: Literal[False] = False

    @model_validator(mode="after")
    def case_binding(self):
        if self.adapter_version != self.components.adapter_version:
            raise ValueError("multimodal_visible_adapter_binding")
        c = self.case
        if len(c.artifacts) != 1 or c.observations or c.report_uri or c.extensions:
            raise ValueError("multimodal_visible_case_binding")
        asset = c.artifacts[0]
        if (
            asset.artifact_type_code != "ARTIFACT:MULTIMODAL_INPUT_INDEX"
            or asset.uri != self.input_object.uri
            or asset.checksum != self.input_object.checksum
            or asset.scope != c.scope
            or c.patient_hash != (self.components.patient_group or UNKNOWN_PATIENT)
            or len(c.goals) != 1
            or c.goals[0].goal_code != "GOAL:" + self.components.task.upper()
            or c.goals[0].goal_level != "evidence_planning"
        ):
            raise ValueError("multimodal_visible_case_binding")
        return self


class MultimodalRegistration(SealedDeclaration):
    schema_version: Literal["dataset-registration/4.0.0"] = "dataset-registration/4.0.0"
    preparation_version: Literal["multimodal-prepared/1.0.0"] = "multimodal-prepared/1.0.0"
    visible: MultimodalVisible
    # No full original NPZ/workbook is stored in the visible object store.
    restricted_manifest: ObjectReference


def register(service, sample, *, key):
    from .adapters import Prepared

    if type(sample) is not Prepared:
        raise AdapterError("multimodal_original_adapter_required")
    sample = sample.recheck()
    owned = []

    def put(store, data):
        uri = store.stage(data)
        owned.append((store, uri))
        ref = store.promote(uri, checksum(data))
        owned[-1] = (store, ref)
        return ObjectReference(uri=ref, checksum=checksum(data), byte_size=len(data))

    def build():
        blobs = []
        for id, raw in sample.payload_bytes:
            chunks = tuple(
                put(service.visible_store, raw[i : i + 4194304])
                for i in range(0, len(raw), 4194304)
            )
            blobs.append(
                StoredPayload(id=id, checksum=checksum(raw), byte_size=len(raw), chunks=chunks)
            )
        stored = seal(StoredInput, components=sample.manifest, blobs=tuple(blobs))
        ref = put(service.visible_store, stored.model_dump_json().encode())
        token = uuid4().hex.upper()
        scope = ScopeReference(scope_type="CASE", case_code="CASE:ANON_" + token)
        provenance = Provenance(
            actor="PLANNER",
            created_at=datetime.now(UTC),
            generator_version="1.0.0",
            source_codes=("SOURCE:DATASET:" + sample.manifest.dataset.upper(),),
        )
        eyes = {c.laterality for c in sample.manifest.components}
        eye = next(iter(eyes)) if len(eyes) == 1 else "UNKNOWN"
        asset = Artifact(
            schema_version="1.0",
            contract_type="Artifact",
            provenance=provenance,
            scope=scope,
            laterality=eye,
            coordinate_space="NOT_APPLICABLE",
            unit=None,
            asset_code="ASSET:ANON_" + token,
            artifact_type_code="ARTIFACT:MULTIMODAL_INPUT_INDEX",
            uri=ref.uri,
            checksum=ref.checksum,
            media_type="application/json",
            qc_status="UNASSESSED",
            research_only=True,
        )
        case = CaseBundle(
            schema_version="1.0",
            contract_type="CaseBundle",
            provenance=provenance,
            scope=scope,
            laterality=eye,
            coordinate_space="NOT_APPLICABLE",
            unit=None,
            case_code=scope.case_code,
            patient_hash=sample.manifest.patient_group or UNKNOWN_PATIENT,
            encounter_code="ENCOUNTER:ANON_" + token,
            goals=(
                {
                    "goal_code": "GOAL:" + sample.manifest.task.upper(),
                    "goal_level": "evidence_planning",
                },
            ),
            artifacts=(asset,),
        )
        visible = seal(
            MultimodalVisible,
            case=case,
            components=sample.manifest,
            input_object=ref,
            adapter_version=sample.manifest.adapter_version,
        )
        hidden = dict(
            schema_version="multimodal-restricted-locators/1.0.0",
            case_code=case.case_code,
            dataset=sample.manifest.dataset,
            input_checksum=sample.manifest.checksum,
            excluded_fields=sample.manifest.excluded_fields,
            sources=[s.model_dump(mode="json") for s in sample.manifest.sources],
            label_values_copied=False,
            evaluation_ready=False,
        )
        restricted = put(service.evaluator_store, json.dumps(hidden).encode())
        return seal(MultimodalRegistration, visible=visible, restricted_manifest=restricted)

    try:
        return service.repository.register(key, sample.manifest.checksum, build)
    except (psycopg.OperationalError, psycopg.InterfaceError):
        raise AdapterError("dataset_commit_unknown_retry_same_key") from None
    except Exception as error:
        for store, uri in reversed(owned):
            store.discard(uri)
        if isinstance(error, IdempotencyConflictError):
            raise AdapterError("dataset_input_conflict") from None
        raise AdapterError("multimodal_registration_failed") from None


def read_input(visible, store):
    visible = MultimodalVisible.model_validate_json(visible.model_dump_json())
    ref = visible.input_object
    raw = store.read_bounded(ref.uri, ref.byte_size)
    if checksum(raw) != ref.checksum or len(raw) != ref.byte_size:
        raise AdapterError("multimodal_input_object_changed")
    stored = StoredInput.model_validate_json(raw)
    if stored.components != visible.components:
        raise AdapterError("multimodal_component_object_mismatch")
    for blob in stored.blobs:
        import hashlib

        h = hashlib.sha256()
        for chunk in blob.chunks:
            data = store.read_bounded(chunk.uri, chunk.byte_size)
            if len(data) != chunk.byte_size or checksum(data) != chunk.checksum:
                raise AdapterError("multimodal_chunk_changed")
            h.update(data)
        if "sha256:" + h.hexdigest() != blob.checksum:
            raise AdapterError("multimodal_blob_changed")
    return stored
