"""Build one dataset-blind material projection and separate audit provenance."""

import io
from collections import defaultdict
from dataclasses import dataclass

from packages.disease_packs.models import seal
from packages.object_store import checksum

from .patient_decoders import ADAPTERS, decode
from .patient_models import BackgroundField, PacketComponent, PacketPayload, PatientPacket, Visit
from .reading import AdapterError

VERSION = "patient-packet-adapters/1.0.1"
SUFFIX = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "application/x-npy": "npy",
    "application/json": "json",
}


@dataclass(frozen=True, repr=False)
class PreparedPatient:
    packet: PatientPacket
    provenance: dict
    restricted: dict
    validation: dict
    contents: dict


def image_check(raw):
    """Decode pixels without assessing medical quality or invoking a model."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 16777216:
                return "pixel_count_limit"
            image.load()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        return "image_pixel_decode_failed"
    return None


def prepare_patient(adapter, root, subject, *, selection=None):
    decoded = decode(adapter, root, subject, selection=selection)
    sources, by_path, provenance, components = {}, {}, {}, []
    payloads, contents, image_errors = {}, {}, {}
    owner_versions = []
    for unit in decoded.units:
        owner_versions.append(unit["owner_version"])
        for s in unit["sources"]:
            old = by_path.get(s.path)
            if old and sources[old]["checksum"] != s.checksum:
                raise AdapterError("patient_packet_source_changed_during_read")
            if old is None:
                sid = "S" + str(len(sources) + 1)
                by_path[s.path] = sid
                sources[sid] = s.model_dump(mode="json")
        mapping = {}
        raw_map = dict(unit["contents"])
        for p in unit["payloads"]:
            raw = raw_map[p.id]
            if checksum(raw) != p.checksum or len(raw) != p.byte_size:
                raise AdapterError("patient_packet_original_payload_mismatch")
            # Equal pixel bytes do not merge two slice positions or two eyes.
            # Storage may share bytes; acquisition record identity remains distinct.
            pid = "P" + str(len(payloads) + 1)
            rel = "payloads/" + p.checksum[7:] + "." + SUFFIX[p.media_type]
            descriptor = PacketPayload(
                id=pid,
                path=rel,
                checksum=p.checksum,
                byte_size=p.byte_size,
                media_type=p.media_type,
            )
            if pid in payloads and payloads[pid] != descriptor:
                raise AdapterError("patient_packet_payload_type_conflict")
            payloads[pid] = descriptor
            contents[rel] = raw
            mapping[p.id] = pid
            if p.media_type.startswith("image/") and pid not in image_errors:
                image_errors[pid] = image_check(raw)
        for c in unit["components"]:
            cid = "C" + str(len(components) + 1)
            ref = "R" + str(len(provenance) + 1)
            provenance[ref] = dict(
                original_component=c.model_dump(mode="json"), owner_version=unit["owner_version"]
            )
            fields = c.model_dump()
            fields.update(
                id=cid,
                payload_ids=tuple(mapping[p] for p in c.payload_ids),
                source_paths=(),
                source_locator=ref,
                provenance_ref=ref,
                source_refs=tuple(by_path[p] for p in c.source_paths),
                unit_basis=ref if c.unit is not None else None,
            )
            errors = {image_errors.get(p) for p in fields["payload_ids"]} - {None}
            fields.update(
                parse_status="invalid_payload"
                if errors
                else "no_payload"
                if c.status in ("missing", "withheld")
                else "readable",
                parse_errors=tuple(sorted(errors)),
            )
            if errors:
                # A broken file remains an archived partial component, not a claim
                # that a model inspected it. Preserve the original descriptor above.
                fields.update(status="partial", reason=";".join(sorted(errors)))
                # Component's original series invariant requires full-member series
                # status to remain obtained; expose decode failure in validation instead.
                if c.kind == "image_series":
                    fields.update(status=c.status, reason=c.reason)
            components.append(PacketComponent(**fields))
    background = []
    for b in decoded.background:
        b = dict(b)
        ref = "R" + str(len(provenance) + 1)
        path, locator = b.pop("source_path"), b.pop("locator")
        provenance[ref] = dict(source_path=path, locator=locator)
        background.append(
            BackgroundField(
                id="B" + str(len(background) + 1),
                provenance_ref=ref,
                source_refs=(by_path[path],),
                **b,
            )
        )
    groups = defaultdict(list)
    for c in (*components, *background):
        groups[c.laterality, c.visit].append(c)
    visits = []
    for (eye, visit), records in groups.items():
        intervals = {c.interval_years for c in records if isinstance(c, PacketComponent)}
        if len(intervals) > 1:
            raise AdapterError("patient_packet_conflicting_visit_time")
        interval = next(iter(intervals), None)
        visits.append(
            Visit(
                id="V" + str(len(visits) + 1),
                laterality=eye,
                source_visit=visit,
                interval_years=interval,
                time_basis="source_relative_years" if interval is not None else "not_provided",
                component_ids=tuple(c.id for c in records),
            )
        )
    identity = checksum((adapter + ":" + subject).encode())
    packet = seal(
        PatientPacket,
        packet_id=identity,
        subject_id=identity,
        identity_basis=decoded.identity_basis,
        collection_scope="all_recorded_eyes_and_visits"
        if decoded.identity_basis == "explicit_source_subject"
        else "selected_sample_only",
        components=tuple(components),
        background=tuple(background),
        visits=tuple(visits),
        payloads=tuple(payloads.values()),
    )
    return PreparedPatient(
        packet=packet,
        contents=contents,
        provenance=dict(
            schema_version="patient-packet-provenance/1.0.0",
            adapter=adapter,
            adapter_version=VERSION,
            owner_versions=sorted(set(owner_versions)),
            subject_selector=subject,
            selection=ADAPTERS[adapter],
            sources=sources,
            records=provenance,
            notes=decoded.notes,
            packet_checksum=packet.checksum,
        ),
        restricted=dict(
            schema_version="patient-packet-restricted-references/1.0.0",
            records=decoded.restricted,
            label_values_exported=False,
            policy="references_only_not_for_qualitative_analyzer",
        ),
        validation=dict(
            schema_version="patient-packet-validation/1.0.0",
            image_payloads_checked=len(image_errors),
            image_decode_errors={k: v for k, v in image_errors.items() if v},
            clinical_quality_assessed=False,
            api_requests=0,
            model_runs=0,
            database_writes=0,
            notes=decoded.notes,
        ),
    )
