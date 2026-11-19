"""Case/input provenance and historical-score preservation, synthetic fixtures."""

import copy
import json

import pytest

from apps import glaucoma_vascular_followup as f
from apps.glaucoma_evidence_plan import checksum as snapshot_checksum
from db.runtime.repository.case_pipeline import identity
from packages.contracts import Artifact
from packages.evidence_details.models import body
from packages.object_store import checksum
from tests.database.test_runtime_repository_02b import EXAMPLES
from tests.test_av_vascular_metrics import av_fixture  # noqa: F401


@pytest.fixture(scope="module")
def vascular_example(request):
    record = copy.deepcopy(request.getfixturevalue("av_fixture")[1])
    image = json.loads((EXAMPLES / "Artifact.valid.json").read_text())
    case = "CASE:ANON_VASCULAR_FIXTURE"
    image.update(
        scope={"scope_type": "CASE", "case_code": case},
        asset_code="ASSET:OID_VASCULAR_IMAGE",
        dimensions=[256, 256],
        laterality="UNKNOWN",
        derived_from=[],
        research_only=True,
    )
    refs = []
    for port, kind in f.INPUTS.items():
        a = copy.deepcopy(image)
        a.update(
            asset_code=identity("ASSET", port),
            artifact_type_code=kind,
            checksum=record["input_bindings"][port]["sha256"],
            derived_from=[image["asset_code"]],
        )
        refs.append(Artifact.model_validate(a).model_dump(mode="json"))
    base = {
        "case_evidence": {"case_code": case, "input": image, "artifacts": refs},
        "research_scoring": {"schema_version": "synthetic_original", "clinical_acceptance": False},
    }
    base["checksum"] = snapshot_checksum(base)
    a = copy.deepcopy(image)
    a.update(
        asset_code="ASSET:OID_VASCULAR_OBSERVATION",
        artifact_type_code=f.TYPE,
        media_type="application/json",
        checksum=checksum(body(record)),
        dimensions=[],
        derived_from=sorted(x["asset_code"] for x in refs),
    )
    artifact = Artifact.model_validate(a).model_dump(mode="json")
    execution = {
        "tool": f.tool.CODE,
        "version": f.tool.VERSION,
        "implementation": f.tool.IMPLEMENTATION,
        "installation_checksum": "synthetic_software_fixture_not_installed",
        "direct_exit_code": 0,
    }
    return base, record, artifact, execution


def test_new_measurements_no_imputed_patient_tests_or_score_upgrade(vascular_example):
    base, record, artifact, execution = vascular_example
    value = f.attach(base, record, artifact, execution)
    assert f.validate_view(value) == base
    assert f.base_view(value) == base
    assert value["research_scoring"] == base["research_scoring"]
    assert len(value["vascular_followup"]["comparisons"]) == 4
    assert f.POLICY["primary_scope"] == "slo_image_research"
    assert "CRAE" in value["vascular_followup"]["comparisons"][3]["remaining"]


@pytest.mark.parametrize(
    "field",
    ["mask_checksum", "case", "size", "derivation", "tool", "exit", "observed", "primary_scope"],
)
def test_binding_rejects_rehashed_tampering(vascular_example, field):
    base, record, artifact, execution = copy.deepcopy(vascular_example)
    if field == "mask_checksum":
        record["input_bindings"]["artery_mask"]["sha256"] = "sha256:" + "a" * 64
        record["checksum"] = f.tool.validate.__globals__["checksum"](
            {k: v for k, v in record.items() if k != "checksum"}
        )
        artifact["checksum"] = checksum(body(record))
    elif field == "case":
        artifact["scope"]["case_code"] = "CASE:ANON_OTHER"
    elif field == "size":
        base["case_evidence"]["artifacts"][0]["dimensions"] = [128, 128]
    elif field == "derivation":
        artifact["derived_from"] = []
    elif field == "tool":
        execution["tool"] = "caller_selected_tool"
    elif field == "exit":
        execution["direct_exit_code"] = 1
    with pytest.raises(ValueError):
        value = f.attach(base, record, artifact, execution)
        if field == "observed":
            value["vascular_followup"]["comparisons"][0]["observed"] = "invented"
        elif field == "primary_scope":
            value["vascular_followup"]["policy"] = dict(f.POLICY, primary_scope="clinical")
        value["checksum"] = snapshot_checksum({k: v for k, v in value.items() if k != "checksum"})
        f.validate_view(value)


def test_known_methods_do_not_create_unmatched_cutoffs(vascular_example):
    rows = f.comparisons(vascular_example[1])
    assert all("source_ids" in r for r in rows)
    assert "小于1" in rows[2]["interpretation"]
    assert "0.70" in rows[0]["interpretation"]
    assert all("probability" not in r for r in rows)


def test_research_source_audit_does_not_impersonate_published_planner(monkeypatch):
    from apps import knowledge_source_tools, vascular_literature_followup
    from packages import object_store

    class VerificationStop(Exception):
        pass

    capabilities = []
    sentinel = object()

    def repository(capability):
        capabilities.append(capability)
        return sentinel

    def verify(entry, repository, store):
        assert entry["records"] and repository is sentinel and store is sentinel
        raise VerificationStop

    monkeypatch.setattr(knowledge_source_tools, "repository", repository)
    monkeypatch.setattr(object_store, "LocalObjectStore", lambda root: sentinel)
    monkeypatch.setattr(vascular_literature_followup, "verify", verify)
    with pytest.raises(VerificationStop):
        f.run("not_reached", "not_reached", repo=None, store=None)
    assert capabilities == ["editor"]
