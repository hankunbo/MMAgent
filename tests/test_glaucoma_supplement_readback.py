import json
from types import SimpleNamespace

import pytest

from apps import glaucoma_case_supplement as supplement
from packages.contracts import Artifact
from packages.object_store import LocalObjectStore, checksum
from tests.database.test_runtime_repository_02b import EXAMPLES


@pytest.mark.parametrize("wrong_parent", [False, True])
def test_supplement_readback_accepts_contract_tuple_and_rejects_wrong_source(
    tmp_path, monkeypatch, wrong_parent
):
    raw = json.loads((EXAMPLES / "CaseBundle.valid.json").read_text())
    case_code = raw["case_code"]
    parent = raw["artifacts"][0]["asset_code"]
    input_sha = raw["artifacts"][0]["checksum"]
    value = {"case_code": case_code, "input_checksum": input_sha}
    body = json.dumps(value).encode()
    store = LocalObjectStore(tmp_path / "objects")
    manifest = {"synthetic": True}
    a = raw["artifacts"][0]
    a.update(
        asset_code="ASSET:OID_SUPPLEMENT_TEST",
        media_type="application/json",
        derived_from=["ASSET:OID_WRONG_PARENT" if wrong_parent else parent],
        checksum=checksum(body),
        uri=store.promote(store.stage(body), checksum(body)),
    )
    a["provenance"]["tool_versions"] = [
        {
            "code": "TOOL:SEGMAN:SLO",
            "version": "1.0.0",
            "checksum": "sha256:" + supplement.registry.checksum(manifest),
        }
    ]
    artifact = Artifact.model_validate(a)
    assert isinstance(artifact.derived_from, tuple)
    saved = {"case_code": case_code, "manifest": manifest, "artifacts": [a]}
    (tmp_path / "registered.json").write_text(json.dumps(saved))
    monkeypatch.setattr(supplement, "DESTINATION", tmp_path)
    repo = SimpleNamespace(get_case_bundle=lambda case: SimpleNamespace(artifacts=[artifact]))
    report = {
        "case_code": case_code,
        "input": {"asset_code": parent, "checksum": input_sha},
        "artifacts": [],
        "images": {},
    }
    if wrong_parent:
        with pytest.raises(ValueError, match="supplement_readback_mismatch"):
            supplement.attach_verified(report, repo, store)
    else:
        supplement.attach_verified(report, repo, store)
        assert report["supplement"]["case_code"] == case_code
