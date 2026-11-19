"""Registration orchestration; separate visible/evaluator stores and sanitized failures."""

from datetime import UTC, datetime
from uuid import uuid4

import psycopg

from db.runtime.repository.dataset_input import LegacyRegistrationError
from db.runtime.repository.ports import IdempotencyConflictError
from packages.contracts import Artifact, CaseBundle, Provenance, ScopeReference
from packages.disease_packs.catalog import validate_prepared_case_contract
from packages.disease_packs.models import seal
from packages.object_store import checksum

from .adapters import projection, wire
from .models import BoundRegistration, ObjectReference, RestrictedManifest, VisibleManifest
from .prepared import capture
from .reading import AdapterError
from .runtime_models import (
    RuntimeInputProfile,
    RuntimeRegistration,
    RuntimeVisibleManifest,
    read_visible,
)


class DatasetPreparationService:
    def __init__(self, repository, visible_store, evaluator_store, *, catalog=None):
        self.catalog = catalog
        if visible_store is evaluator_store or getattr(visible_store, "root", None) == getattr(
            evaluator_store, "root", None
        ):
            raise AdapterError("separate_evaluator_store_required")
        self.repository = repository
        self.visible_store = visible_store
        self.evaluator_store = evaluator_store
        # Root capability isolation, including pre-existing permissive umasks.
        if hasattr(evaluator_store, "root"):
            evaluator_store.root.chmod(0o700)

    def register(self, sample, *, key):
        sample = capture(sample, self.catalog)
        owned = []

        def put(store, data):
            uri = store.stage(data)
            owned.append((store, uri))
            target = store.promote(uri, checksum(data))
            owned[-1] = (store, target)
            return ObjectReference(uri=target, checksum=checksum(data), byte_size=len(data))

        def build():
            token = uuid4().hex.upper()
            case_code = "CASE:ANON_" + token
            scope = ScopeReference(scope_type="CASE", case_code=case_code)
            provenance = Provenance(
                actor="PLANNER",
                created_at=datetime.now(UTC),
                generator_version="1.0.0",
                source_codes=(sample.dataset.code,),
            )
            images = []
            for index, (data, shape, eye, modality) in enumerate(sample.images):
                ref = put(self.visible_store, data)
                images.append(
                    Artifact(
                        schema_version="1.0",
                        contract_type="Artifact",
                        provenance=provenance,
                        scope=scope,
                        laterality=eye,
                        coordinate_space="PIXEL_2D",
                        unit="px",
                        asset_code=f"ASSET:ANON_{token}_{index}",
                        artifact_type_code="ARTIFACT:SLO_IMAGE"
                        if modality == "SLO"
                        else "ARTIFACT:FUNDUS_IMAGE",
                        uri=ref.uri,
                        checksum=ref.checksum,
                        media_type="image/png" if modality == "SLO" else "image/jpeg",
                        qc_status="UNASSESSED",
                        dimensions=shape,
                        research_only=True,
                    )
                )
            case = CaseBundle(
                schema_version="1.0",
                contract_type="CaseBundle",
                provenance=provenance,
                scope=scope,
                laterality="UNKNOWN" if len(images) == 1 else "BILATERAL",
                coordinate_space="NOT_APPLICABLE",
                unit=None,
                case_code=case_code,
                patient_hash=sample.patient_hash,
                encounter_code="ENCOUNTER:ANON_" + token,
                goals=(
                    {
                        "goal_code": sample.pack.definition.target_node_code,
                        "goal_level": sample.pack.definition.goal_level,
                    },
                ),
                artifacts=tuple(images),
            )
            validate_prepared_case_contract(sample.pack, case)
            profile = {}
            if sample.runtime_profile:
                from .reading import NPZ_KEYS

                metadata = {v["key"] for v in sample.fields["npz_metadata"]}
                profile["input_profile"] = RuntimeInputProfile(
                    sample=sample.fields["csv"]["filename"],
                    source_files=sample.files,
                    missing_optional_fields=tuple(sorted(set(NPZ_KEYS) - set(metadata))),
                    quarantined_field_names=tuple(sorted(set(metadata) - set(NPZ_KEYS))),
                )
            visible = seal(
                RuntimeVisibleManifest if sample.runtime_profile else VisibleManifest,
                dataset_ref=sample.dataset.reference(),
                pack_ref=sample.pack.reference(),
                visibility=sample.dataset.visibility,
                case=case,
                patient_identity=sample.identity_basis,
                text_review="pending" if sample.note is not None else "not_applicable",
                available_modalities=tuple(sorted({i[3] for i in sample.images})),
                **profile,
            )
            note_ref = put(self.evaluator_store, sample.note.encode()) if sample.note else None
            projection_value = projection(sample)
            projection_ref = (
                put(self.evaluator_store, projection_value.model_dump_json().encode())
                if projection_value
                else None
            )
            restricted = seal(
                RestrictedManifest,
                case_code=case_code,
                dataset_ref=sample.dataset.reference(),
                pack_ref=sample.pack.reference(),
                visibility=sample.dataset.visibility,
                source_files=sample.files,
                source_fields=put(self.evaluator_store, wire(sample.fields)),
                original_note=note_ref,
                projection=projection_ref,
                label_scope=(
                    "source_separated"
                    if sample.dataset.code == "SOURCE:DATASET:FAIRVLMED"
                    else "patient_coarse_and_derived_image_rows"
                ),
            )
            restricted_ref = put(self.evaluator_store, restricted.model_dump_json().encode())
            return seal(
                RuntimeRegistration if sample.runtime_profile else BoundRegistration,
                visible=visible,
                restricted_manifest=restricted_ref,
            )

        try:
            return self.repository.register(key, sample.fingerprint(), build)
        except (psycopg.OperationalError, psycopg.InterfaceError):
            # SQL COMMIT acknowledgement may be lost: preserve objects and resolve by same key.
            raise AdapterError("dataset_commit_unknown_retry_same_key") from None
        except Exception as error:
            for store, uri in reversed(owned):
                store.discard(uri)
            if isinstance(error, LegacyRegistrationError):
                raise AdapterError("dataset_legacy_key_requires_new_registration") from None
            if isinstance(error, IdempotencyConflictError):
                raise AdapterError("dataset_input_conflict") from None
            raise AdapterError("dataset_registration_failed") from None

    def read_visible(self, case_code):
        from .current_read_scope import _read_visible

        return _read_visible(
            self,
            case_code,
            lambda: VisibleInputReader(self.repository, self.visible_store).read(case_code),
        )

    def read_visible_components(self, case_code):
        """Full current input; scoped reads revalidate all chunks before return."""
        from .current_read_scope import _read_visible

        return _read_visible(
            self,
            case_code,
            lambda: VisibleInputReader(self.repository, self.visible_store).read_components(
                case_code
            ),
            kind="components",
        )

    def register_multimodal(self, sample, *, key):
        """Persist owner-rebuilt multimodal inputs; not a clinical execution grant."""
        from .multimodal.storage import register

        return register(self, sample, key=key)


class VisibleInputReader:
    """07B-facing port: no evaluator store, original text or source-row read API."""

    def __init__(self, repository, store):
        self.repository = repository
        self.store = store

    def read(self, case_code):
        return self._read(case_code)[0]

    def read_components(self, case_code):
        value, stored = self._read(case_code)
        if stored is None:
            raise AdapterError("multimodal_visible_input_required")
        return value, stored

    def _read(self, case_code):
        value = self.repository.get_visible(case_code)
        value = read_visible(value.model_dump_json())
        stored = None
        if value.schema_version == "visible-input/3.0.0":
            from .multimodal.storage import read_input

            stored = read_input(value, self.store)
        for artifact in value.case.artifacts:
            data = self.store.read_bounded(artifact.uri, 4194304)
            if checksum(data) != artifact.checksum:
                raise AdapterError("visible_object_checksum_mismatch")
        return value, stored


class RestrictedEvaluatorReader:
    """Provision only outside run. It is never a dependency of the visible reader."""

    def __init__(self, store):
        self.store = store

    def read(self, reference):
        ref = ObjectReference.model_validate_json(reference.model_dump_json())
        raw = self.store.read_bounded(ref.uri, ref.byte_size)
        if len(raw) != ref.byte_size or checksum(raw) != ref.checksum:
            raise AdapterError("evaluation_object_checksum_mismatch")
        return raw

    def manifest(self, reference):
        return RestrictedManifest.model_validate_json(self.read(reference))
