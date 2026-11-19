"""Task-independent patient archive. Not a runtime input or medical capability grant."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.disease_packs.models import SealedDeclaration

from .models import Value
from .multimodal.models import SHA, Component


class PacketPayload(Value):
    id: str
    path: str = Field(pattern=r"^payloads/[a-f0-9]{64}\.(npy|json|jpg|png)$")
    checksum: str = Field(pattern=SHA)
    byte_size: int = Field(ge=1, le=67108864)
    media_type: Literal["application/x-npy", "application/json", "image/jpeg", "image/png"]


class PacketComponent(Component):
    # Reuse the original component value/status/shape invariants. File locations,
    # dataset names and unit qualifications live in provenance, not this projection.
    provenance_ref: str
    source_refs: tuple[str, ...]
    parse_status: Literal["readable", "invalid_payload", "no_payload"]
    parse_errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def opaque_sources(self):
        if self.source_paths or self.source_locator != self.provenance_ref:
            raise ValueError("patient_packet_raw_source_in_analysis")
        if self.unit_basis not in (None, self.provenance_ref):
            raise ValueError("patient_packet_unit_provenance")
        if (self.parse_status == "invalid_payload") != bool(self.parse_errors):
            raise ValueError("patient_packet_parse_status")
        return self


class BackgroundField(Value):
    id: str
    concept: str
    value: str | Annotated[float, Field(allow_inf_nan=False)] | None
    status: Literal["obtained", "missing"]
    laterality: Literal["LEFT", "RIGHT", "UNKNOWN"] = "UNKNOWN"
    visit: str = "unknown"
    provenance_ref: str
    source_refs: tuple[str, ...]
    meaning_status: Literal["source_declared", "code_mapping_unqualified"]

    @model_validator(mode="after")
    def presence(self):
        if (self.value is None) != (self.status == "missing"):
            raise ValueError("patient_background_presence")
        if isinstance(self.value, str) and len(self.value) > 300:
            raise ValueError("patient_background_value_length")
        return self


class Visit(Value):
    id: str
    laterality: Literal["LEFT", "RIGHT", "UNKNOWN"]
    source_visit: str
    interval_years: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    date: None = None
    time_basis: Literal["source_relative_years", "not_provided"]
    component_ids: tuple[str, ...]


class PatientPacket(SealedDeclaration):
    schema_version: Literal["standard-patient-packet/1.0.0"] = "standard-patient-packet/1.0.0"
    packet_id: str = Field(pattern=SHA)
    subject_id: str = Field(pattern=SHA)
    identity_basis: Literal["explicit_source_subject", "sample_only_unproven_patient"]
    collection_scope: Literal["all_recorded_eyes_and_visits", "selected_sample_only"]
    components: tuple[PacketComponent, ...] = Field(min_length=1, max_length=10000)
    background: tuple[BackgroundField, ...] = ()
    visits: tuple[Visit, ...]
    payloads: tuple[PacketPayload, ...]
    # Full archive MUST NOT silently serve as a time-limited task view.
    task_scope_selected: Literal[False] = False
    qualitative_analysis: Literal["not_run"] = "not_run"
    runtime_ready: Literal[False] = False
    score: None = None

    @model_validator(mode="after")
    def relationships(self):
        ids = [c.id for c in self.components] + [b.id for b in self.background]
        payloads = {p.id: p for p in self.payloads}
        if len(ids) != len(set(ids)) or len(payloads) != len(self.payloads):
            raise ValueError("patient_packet_duplicate_id")
        if len({v.id for v in self.visits}) != len(self.visits):
            raise ValueError("patient_packet_duplicate_visit")
        claimed = [i for v in self.visits for i in v.component_ids]
        if len(claimed) != len(set(claimed)) or set(claimed) != set(ids):
            raise ValueError("patient_packet_visit_partition")
        records = {c.id: c for c in (*self.components, *self.background)}
        for v in self.visits:
            for i in v.component_ids:
                c = records[i]
                if (c.visit, c.laterality) != (v.source_visit, v.laterality):
                    raise ValueError("patient_packet_visit_eye_binding")
                if isinstance(c, PacketComponent) and c.interval_years != v.interval_years:
                    raise ValueError("patient_packet_visit_time_binding")
        used = {p for c in self.components for p in c.payload_ids}
        if used != set(payloads):
            raise ValueError("patient_packet_payload_binding")
        return self
