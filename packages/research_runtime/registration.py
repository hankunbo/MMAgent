"""Deployment-owned 06 request registration from the original 07A visible reader."""

import json
from datetime import UTC, datetime

from psycopg.types.json import Jsonb

from db.runtime.repository.case_pipeline import identity
from db.runtime.repository.ports import IdempotencyConflictError
from packages.grounding.models import digest, seal, validate
from packages.grounding.query_models import ArtifactMetadata, CaseScope
from packages.tool_admission.models import InputRecord, ResearchRequest
from packages.tool_workflow import WorkflowChoice


def single_choice(admission):
    if len(admission.retained_roots) != 1 or len(admission.retained) != 1:
        raise ValueError("research_single_admitted_tool_required")
    node = admission.retained[0]
    if node.identity != admission.retained_roots[0] or any(
        len(p.artifacts) != 1 or p.producer_refs for p in node.inputs
    ):
        raise ValueError("research_single_visible_input_required")
    return validate(
        WorkflowChoice,
        dict(
            roots=(node.identity,),
            nodes=(
                dict(
                    candidate_ref=node.identity,
                    inputs=[
                        dict(
                            port=p.port,
                            sources=[dict(kind="artifact", asset_code=p.artifacts[0].asset_code)],
                        )
                        for p in node.inputs
                    ],
                ),
            ),
        ),
    )


class VisibleResearchRequests:
    def __init__(
        self,
        *,
        runtime,
        visible_reader,
        objects,
        candidate,
        invocation,
        catalog_ref,
        policy,
        input_binding,
        modality,
        scope_terms,
        config_checksum,
    ):
        self.runtime, self.visible_reader, self.objects = runtime, visible_reader, objects
        self.candidate, self.invocation, self.catalog_ref = candidate, invocation, catalog_ref
        self.policy, self.input_binding, self.modality = policy, input_binding, modality
        self.scope_terms, self.config_checksum = tuple(scope_terms), config_checksum

    def _records(self, visible):
        case = visible.case
        if len(case.artifacts) != 1 or visible.available_modalities != ("SLO",):
            raise ValueError("research_single_SLO_visible_required")
        a = case.artifacts[0]
        if a.artifact_type_code != "ARTIFACT:SLO_IMAGE" or not a.research_only:
            raise ValueError("research_visible_type")
        raw = self.objects.read_bounded(a.uri, 4194304)
        from packages.object_store import checksum

        if checksum(raw) != a.checksum:
            raise ValueError("research_visible_bytes_changed")
        m = ArtifactMetadata(
            asset_code=a.asset_code,
            checksum=a.checksum,
            case_code=case.case_code,
            patient_hash=case.patient_hash,
            encounter_code=case.encounter_code,
            run_id=None,
            artifact_ref=self.input_binding.artifact_ref,
            modality=self.modality,
            laterality=a.laterality,
            coordinate_space=a.coordinate_space,
            unit=self.input_binding.unit_term,
            scope_terms=self.scope_terms,
            qc_status=a.qc_status,
            acquired_at=None,
        )
        return (
            InputRecord(
                artifact=a,
                type_binding=self.input_binding,
                metadata=m,
                byte_size=len(raw),
                content_kind="image",
                source_ref=self.input_binding.source_ref,
            ),
        )

    def read_input(self, scope, run_id, asset_code):
        visible = self.visible_reader.read(scope.case_code)
        case = visible.case
        actual = CaseScope(
            case_code=case.case_code,
            encounter_code=case.encounter_code,
            patient_hash=case.patient_hash,
            laterality=case.laterality,
        )
        if actual != scope:
            raise ValueError("research_visible_scope")
        return next(r for r in self._records(visible) if r.artifact.asset_code == asset_code)

    @staticmethod
    def request_code(key):
        return "RESEARCH:J03:" + digest(key)[7:39].upper()

    def register(self, visible, key):
        code = self.request_code(key)
        storage_key = identity("TASK", "research-registration:" + code)
        case = visible.case
        binding = digest(dict(visible=visible.checksum, config=self.config_checksum))
        with self.runtime._connect() as c:
            c.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,9))", (storage_key,))
            old = c.execute(
                "SELECT * FROM runtime.idempotency_request WHERE idempotency_key=%s", (storage_key,)
            ).fetchone()
            if old:
                if old["operation"] != "research_registration" or old["request_hash"] != binding:
                    raise IdempotencyConflictError("research_registration_conflict")
                return validate(ResearchRequest, old["response_payload"])
            records = self._records(visible)
            now = datetime.now(UTC)
            request = seal(
                ResearchRequest,
                code=code,
                version="1.0.0",
                purpose="shadow",
                compatibility_policy=self.policy,
                lock=dict(run_id=identity("RUN", "research:" + code), catalog_ref=self.catalog_ref),
                scope=dict(
                    case_code=case.case_code,
                    encounter_code=case.encounter_code,
                    patient_hash=case.patient_hash,
                    laterality=case.laterality,
                ),
                window_start=now,
                window_end=now,
                snapshot_at=now,
                satisfied_terms=self.scope_terms,
                artifacts=tuple(r.metadata for r in records),
                candidate=self.candidate,
                invocation=self.invocation,
                inputs=[
                    dict(port=p.port, asset_codes=[records[0].artifact.asset_code])
                    for p in self.candidate.inputs
                ],
            )
            c.execute(
                "INSERT INTO runtime.idempotency_request "
                "(idempotency_key,operation,request_hash,response_contract_type,response_payload) "
                "VALUES(%s,'research_registration',%s,'ResearchRequest',%s)",
                (storage_key, binding, Jsonb(request.model_dump(mode="json"))),
            )
            return request

    def for_key(self, key):
        storage_key = identity("TASK", "research-registration:" + self.request_code(key))
        with self.runtime._connect() as c:
            row = c.execute(
                "SELECT response_payload FROM runtime.idempotency_request "
                "WHERE idempotency_key=%s AND operation='research_registration'",
                (storage_key,),
            ).fetchone()
        if row is None:
            raise ValueError("research_registration_pending")
        return ResearchRequest.model_validate_json(json.dumps(row["response_payload"]))

    def read_request(self, code, version, checksum):
        storage_key = identity("TASK", "research-registration:" + code)
        with self.runtime._connect() as c:
            row = c.execute(
                "SELECT response_payload FROM runtime.idempotency_request "
                "WHERE idempotency_key=%s AND operation='research_registration'",
                (storage_key,),
            ).fetchone()
        if row is None:
            raise ValueError("research_registration_pending")
        request = validate(ResearchRequest, row["response_payload"])
        if (request.code, request.version, request.checksum) != (code, version, checksum):
            raise ValueError("research_registration_identity")
        return request
