import json

import psycopg
import pytest

from db.runtime.repository.dataset_input import DatasetInputRepository
from packages.dataset_adapters.adapters import FAIR_FIELDS
from packages.dataset_adapters.models import VisibleManifest
from packages.dataset_adapters.reading import AdapterError
from packages.dataset_adapters.service import DatasetPreparationService, RestrictedEvaluatorReader
from packages.object_store import LocalObjectStore
from tests.database.test_runtime_repository_02b import database as database
from tests.dataset_adapters.support import csv_write, fair, odir


def service(database, tmp_path):
    runtime = database[3]
    visible = LocalObjectStore(tmp_path / "visible")
    hidden = LocalObjectStore(tmp_path / "restricted")
    return DatasetPreparationService(DatasetInputRepository(runtime), visible, hidden)


def test_real_pg_roundtrip_no_hidden_json_or_logs(database, tmp_path, caplog, capsys):
    a, rows = fair(tmp_path / "source")
    s = a.prepare("data_00000.npz")
    svc = service(database, tmp_path)
    result = svc.register(s, key="synthetic-readback")
    case = result.visible.case
    assert database[3].get_case_bundle(case.case_code) == case
    assert svc.read_visible(case.case_code) == result.visible
    assert svc.register(s, key="synthetic-readback") == result
    reader = RestrictedEvaluatorReader(svc.evaluator_store)
    hidden = reader.manifest(result.restricted_manifest)
    assert reader.read(hidden.original_note).decode() == s.note
    assert json.loads(reader.read(hidden.source_fields))["csv"]["gpt4_summary"] == "SECRET_SUMMARY"
    with pytest.raises(FileNotFoundError):
        svc.visible_store.read(hidden.original_note.uri)
    with database[3]._connect() as conn:
        text = json.dumps(
            conn.execute("SELECT response_payload FROM runtime.idempotency_request").fetchall()
        )
    assert "SECRET_" not in text and "SECRET_" not in case.model_dump_json()
    assert not case.observations and case.report_uri is None and not case.extensions
    assert "SECRET_" not in caplog.text + capsys.readouterr().out
    rows[0]["note"] += "合法变化"
    csv_write(tmp_path / "source/ReadMe/data_summary.csv", FAIR_FIELDS, rows)
    changed = a.prepare("data_00000.npz")
    before = set(svc.visible_store.root.rglob("*"))
    with pytest.raises(AdapterError, match="conflict"):
        svc.register(changed, key="synthetic-readback")
    assert set(svc.visible_store.root.rglob("*")) == before
    wire = json.loads(result.visible.model_dump_json())
    wire["unknown"] = "SECRET"
    with pytest.raises(ValueError):
        VisibleManifest.model_validate_json(json.dumps(wire))


def test_odir_runtime_both_eyes(database, tmp_path):
    a, _ = odir(tmp_path / "source")
    svc = service(database, tmp_path)
    result = svc.register(a.prepare("11"), key="synthetic-odir")
    assert database[3].get_case_bundle(result.visible.case.case_code) == result.visible.case
    assert result.visible.patient_identity == "dataset_local_explicit_id"
    assert result.visible.patient_independence == "unproven"


@pytest.mark.parametrize("mode", ["object", "db"])
def test_object_database_rollback(database, tmp_path, monkeypatch, mode):
    a, _ = fair(tmp_path / "source")
    svc = service(database, tmp_path)
    with database[3]._connect() as c:
        before = c.execute("SELECT count(*) AS n FROM runtime.case_record").fetchone()["n"]
    if mode == "object":

        def fail(*args):
            raise OSError("SECRET_OBJECT_ERROR")

        monkeypatch.setattr(svc.evaluator_store, "promote", fail)
    else:
        original = database[3]._create_case_bundle

        def fail(*args):
            original(*args)
            raise psycopg.errors.CheckViolation("SECRET_DB_ERROR")

        monkeypatch.setattr(database[3], "_create_case_bundle", fail)
    with pytest.raises(AdapterError, match="dataset_registration_failed") as e:
        svc.register(a.prepare("data_00000.npz"), key="rollback-" + mode)
    assert "SECRET" not in str(e.value)
    for store in (svc.visible_store, svc.evaluator_store):
        assert not list(store.root.glob("objects/*")) and not list(store.root.glob("staging/*"))
    with database[3]._connect() as c:
        assert c.execute("SELECT count(*) AS n FROM runtime.case_record").fetchone()["n"] == before


def test_lost_commit_response_preserves_objects_and_recovers(database, tmp_path):
    a, _ = fair(tmp_path / "source")
    svc = service(database, tmp_path)
    s = a.prepare("data_00000.npz")
    original = svc.repository.register

    def lost(*args):
        original(*args)
        raise psycopg.OperationalError("ack lost")

    svc.repository.register = lost
    with pytest.raises(AdapterError, match="commit_unknown"):
        svc.register(s, key="ack-lost")
    before = set(svc.visible_store.root.rglob("*"))
    svc.repository.register = original
    result = svc.register(s, key="ack-lost")
    assert svc.read_visible(result.visible.case.case_code) == result.visible
    assert set(svc.visible_store.root.rglob("*")) == before
