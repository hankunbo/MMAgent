from dataclasses import replace

import pytest

from packages.dataset_adapters.reading import AdapterError
from packages.object_store import checksum
from tests.database.test_runtime_repository_02b import database as database
from tests.dataset_adapters.support import fair, odir
from tests.dataset_adapters.test_registration import service


@pytest.mark.parametrize("change", ["note", "shape", "patient_hash"])
def test_changed_registration_material_must_not_replay_old_result(database, tmp_path, change):
    adapter, _ = (odir if change == "patient_hash" else fair)(tmp_path / "source")
    sample = adapter.prepare("11" if change == "patient_hash" else "data_00000.npz")
    svc = service(database, tmp_path)
    svc.register(sample, key="review-" + change)
    if change == "note":
        altered = replace(sample, note="Synthetic changed source note")
    elif change == "shape":
        data, shape, eye, modality = sample.images[0]
        altered = replace(sample, images=((data, (3, 2), eye, modality),))
    else:
        altered = replace(sample, patient_hash=checksum(b"OTHER_SYNTHETIC_PATIENT"))
    before = state(database, svc)
    with pytest.raises(AdapterError, match="^prepared_input_inconsistent$"):
        svc.register(altered, key="review-" + change)
    assert state(database, svc) == before


def test_mismatched_image_dimensions_must_not_be_registered(database, tmp_path):
    adapter, _ = fair(tmp_path / "source")
    sample = adapter.prepare("data_00000.npz")
    data, _, eye, modality = sample.images[0]
    altered = replace(sample, images=((data, (3, 2), eye, modality),))
    svc = service(database, tmp_path)
    before = state(database, svc)
    with pytest.raises(AdapterError, match="^prepared_input_inconsistent$"):
        svc.register(altered, key="review-new-shape")
    assert state(database, svc) == before


def state(database, svc):
    import json

    with database[3]._connect() as conn:
        rows = conn.execute(
            "SELECT idempotency_key, operation, request_hash, response_payload "
            "FROM runtime.idempotency_request ORDER BY idempotency_key"
        ).fetchall()
        cases = conn.execute(
            "SELECT case_code FROM runtime.case_record ORDER BY case_code"
        ).fetchall()
    objects = tuple(
        tuple(sorted(str(p) for p in store.root.rglob("*") if p.is_file()))
        for store in (svc.visible_store, svc.evaluator_store)
    )
    return json.dumps(rows, sort_keys=True), cases, objects


def test_snapshot_build_retry_and_legal_change_conflict(database, tmp_path):
    from packages.dataset_adapters.adapters import FAIR_FIELDS
    from packages.dataset_adapters.service import RestrictedEvaluatorReader
    from tests.dataset_adapters.support import csv_write

    a, rows = fair(tmp_path / "source")
    sample = a.prepare("data_00000.npz")
    svc = service(database, tmp_path)
    original_register = svc.repository.register

    def mutate_after_fingerprint(key, fingerprint, build):
        sample.note = "SECRET_CHANGED_AFTER_CAPTURE"
        sample.fields["csv"]["note"] = sample.note
        sample.images = ()
        return original_register(key, fingerprint, build)

    svc.repository.register = mutate_after_fingerprint
    result = svc.register(sample, key="snapshot")
    svc.repository.register = original_register
    reader = RestrictedEvaluatorReader(svc.evaluator_store)
    hidden = reader.manifest(result.restricted_manifest)
    assert reader.read(hidden.original_note).decode() == rows[0]["note"]
    assert result.visible.case.artifacts[0].dimensions == (2, 3)
    before = state(database, svc)
    retry = a.prepare("data_00000.npz")
    retry.fields = dict(reversed(list(retry.fields.items())))
    assert svc.register(retry, key="snapshot") == result
    assert state(database, svc) == before
    rows[0]["note"] += "合法变化"
    csv_write(tmp_path / "source/ReadMe/data_summary.csv", FAIR_FIELDS, rows)
    with pytest.raises(AdapterError, match="^dataset_input_conflict$"):
        svc.register(a.prepare("data_00000.npz"), key="snapshot")
    assert state(database, svc) == before


def test_legacy_readable_but_key_requires_explicit_new_registration(database, tmp_path):
    from psycopg.types.json import Jsonb

    from packages.dataset_adapters.adapters import wire
    from packages.dataset_adapters.models import Registration
    from packages.disease_packs.models import seal

    a, _ = fair(tmp_path / "source")
    sample = a.prepare("data_00000.npz")
    svc = service(database, tmp_path)
    current_register = svc.repository.register

    def historical_insert(key, unused_fingerprint, build):
        # Synthetic pre-upgrade row in our isolated cluster; production has no v1 fallback.
        value = build()
        legacy = seal(
            Registration, visible=value.visible, restricted_manifest=value.restricted_manifest
        )
        legacy_hash = checksum(
            wire(
                dict(
                    adapter="dataset-adapters/1.0.0",
                    projection="conservative-note/1.0.0",
                    dataset=sample.dataset.checksum,
                    pack=sample.pack.checksum,
                    files=[f.model_dump() for f in sample.files],
                    fields_checksum=checksum(wire(sample.fields)),
                    images=[checksum(i[0]) for i in sample.images],
                )
            )
        )
        safe_key = "TASK:OID_DSI_" + checksum(key.encode())[7:55].upper()
        with database[3]._connect() as conn:
            database[3]._create_case_bundle(
                conn, value.visible.case, safe_key.replace("_DSI_", "_DSC_")
            )
            conn.execute(
                "INSERT INTO runtime.idempotency_request "
                "(idempotency_key,operation,request_hash,response_contract_type,response_payload) "
                "VALUES (%s,'dataset_input',%s,'DatasetInput',%s)",
                (safe_key, legacy_hash, Jsonb(legacy.model_dump(mode="json"))),
            )
        return legacy

    svc.repository.register = historical_insert
    legacy = svc.register(sample, key="legacy")
    svc.repository.register = current_register
    before = state(database, svc)
    assert svc.read_visible(legacy.visible.case.case_code) == legacy.visible
    assert database[3].get_case_bundle(legacy.visible.case.case_code) == legacy.visible.case
    with pytest.raises(AdapterError, match="^dataset_legacy_key_requires_new_registration$"):
        svc.register(sample, key="legacy")
    with pytest.raises(AdapterError, match="^prepared_input_inconsistent$"):
        svc.register(replace(sample, note="SECRET_CHANGED"), key="legacy")
    assert state(database, svc) == before
    current = svc.register(sample, key="explicit-v2-new-key")
    assert current.schema_version == "dataset-registration/2.0.0"
    assert current.fingerprint_version == "registration-fingerprint/2.0.0"
    assert svc.register(sample, key="explicit-v2-new-key") == current
    assert svc.read_visible(legacy.visible.case.case_code) == legacy.visible
