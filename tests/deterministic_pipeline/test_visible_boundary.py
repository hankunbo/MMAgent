import json

import pytest

from db.runtime.repository.dataset_input import DatasetInputRepository
from packages.dataset_adapters.models import BoundRegistration, VisibleManifest
from packages.dataset_adapters.service import DatasetPreparationService
from packages.deterministic_pipeline.models import PipelineError
from packages.object_store import LocalObjectStore
from tests.dataset_adapters.support import fair
from tests.deterministic_pipeline.real_support import assembly


def test_draft_safe_report_and_complete_manifest(database, tmp_path, caplog):
    visible = LocalObjectStore(tmp_path / "visible")
    private = LocalObjectStore(tmp_path / "restricted")
    pipeline, cfg = assembly(database, visible)
    adapter, _ = fair(tmp_path / "fixture-source")
    preparer = DatasetPreparationService(DatasetInputRepository(pipeline.runtime), visible, private)
    registration = preparer.register(adapter.prepare("data_00000.npz"), key="07b-synthetic-visible")
    assert isinstance(registration, BoundRegistration)
    v = registration.visible
    report = pipeline.run(
        v.case.case_code, input_checksum=v.checksum, config_checksum=cfg.checksum, key="gap"
    )
    assert report["stop_reason"] == "knowledge_not_admitted"
    assert report["run_id"] is None and report["graph"] is None and report["observations"] == []
    assert VisibleManifest.model_validate_json(json.dumps(report["visible_manifest"])) == v
    assert v.patient_identity == "unknown" and v.text_review == "pending"
    assert (
        pipeline.run(
            v.case.case_code, input_checksum=v.checksum, config_checksum=cfg.checksum, key="gap"
        )
        == report
    )
    assert "SECRET_" not in json.dumps(report) + caplog.text
    with pipeline.runtime._connect() as c:
        assert (
            c.execute(
                "SELECT count(*) AS n FROM runtime.run WHERE case_code=%s", (v.case.case_code,)
            ).fetchone()["n"]
            == 0
        )
    second = preparer.register(
        adapter.prepare("data_00000.npz"), key="07b-owner-explicit-new-case"
    ).visible
    assert second.case.patient_hash == v.case.patient_hash
    assert second.case.case_code != v.case.case_code
    other = pipeline.run(
        second.case.case_code,
        input_checksum=second.checksum,
        config_checksum=cfg.checksum,
        key="gap",
    )
    assert other["case_code"] != report["case_code"]
    with pytest.raises(PipelineError, match="version_conflict"):
        pipeline.run(
            v.case.case_code,
            input_checksum="sha256:" + "f" * 64,
            config_checksum=cfg.checksum,
            key="gap",
        )
    with pytest.raises(PipelineError, match="synthetic_input_not_enabled"):
        pipeline.run_synthetic(v.case, pack_ref=v.pack_ref, config_checksum=cfg.checksum, key="gap")


@pytest.mark.parametrize("field", ["extensions", "observations", "report_uri"])
def test_visible_reader_revalidation_blocks_hidden_fields(database, tmp_path, field):
    visible = LocalObjectStore(tmp_path / "visible")
    pipeline, cfg = assembly(database, visible)
    adapter, _ = fair(tmp_path / "source")
    prep = DatasetPreparationService(
        DatasetInputRepository(pipeline.runtime), visible, LocalObjectStore(tmp_path / "private")
    )
    v = prep.register(adapter.prepare("data_00000.npz"), key=field).visible
    bad = v.case.model_copy(
        update={field: {"raw_note": "SECRET"} if field == "extensions" else "SECRET"}
    )

    class Reader:
        def read(self, case_code):
            return v.model_copy(update={"case": bad})

    pipeline.visible_reader = Reader()
    with pytest.raises(PipelineError, match="visible_input_unavailable") as err:
        pipeline.run(
            v.case.case_code, input_checksum=v.checksum, config_checksum=cfg.checksum, key="bad"
        )
    assert "SECRET" not in str(err.value)
