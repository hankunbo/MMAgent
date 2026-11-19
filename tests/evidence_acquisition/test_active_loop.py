"""Synthetic wire + actual Planner, acceptance, installation and 6.1 scoring."""

import copy
import json
from types import SimpleNamespace

import pytest

from apps.active_evidence_materials import RULES, ActiveCaseEvidence
from apps.active_evidence_sources import Paragraphs, verify
from packages.evidence_acquisition.models import Requirement
from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest
from tests.expert_agent import test_batch_assembly as helper


def active_assembly(monkeypatch):
    original = helper.assemble

    def adapted(**kwargs):
        evidence = kwargs["evidence"]
        evidence.targeted_acquisition = True
        evidence.manifest = {"synthetic": True, "source_acquisition": "fixture"}
        evidence.requirements = lambda state: tuple(
            Requirement(
                assessment_id=t.assessment_id,
                purpose="disease_relation",
                acquisition="image",
                required_inputs=("synthetic_image",),
            )
            for t in project_review_tasks(state).tasks
        )
        return original(**kwargs)

    monkeypatch.setattr(helper, "assemble", adapted)
    return helper.assembly(monkeypatch)


def test_active_real_owners_targeted_wire_checkpoints_and_scores(monkeypatch, tmp_path):
    f = active_assembly(monkeypatch)
    completed, roles = helper.stages(f, tmp_path)
    assert roles == ("source", "vision", "review")
    assert [r["status"] for r in completed] == ["accepted"] * 3
    assert f.ports.budget.used == 5
    payload = json.loads(f.wire.calls[1][0]["messages"][1]["content"][0]["text"])
    assert payload["schema_version"] == "targeted-vision-input/1.0.0"
    assert (
        payload["vision_task"]["observation"]["code"]
        == f.initial.material.observations[0].reference.code
    )
    assert f.owner.owner.role == "review"
    assert len([k for k in f.documents.saved if k.startswith("active-acquisition:")]) == 7
    assert len([k for k in f.documents.saved if k.startswith("research-owner-install:")]) == 3
    records = f.review.acceptance.read(key=completed[-1]["acceptance_key"])[0]
    assert len(records["batch_results"]) == 3
    assert "计算依赖与判读背景" in (tmp_path / "review-report.html").read_text()
    prior = dict(f.documents.saved)
    assert helper.stages(f, tmp_path)[0] == completed
    assert f.documents.saved == prior and f.ports.budget.used == 5


def test_active_checkpoint_confirmation_loss_is_not_new_call(monkeypatch, tmp_path):
    f = active_assembly(monkeypatch)
    write = f.documents.write
    failed = [False]

    def lost(key, value):
        write(key, value)
        if key == "active-acquisition:before-vision" and not failed[0]:
            failed[0] = True
            raise OSError("synthetic_ack_lost")

    f.documents.write = lost
    with pytest.raises(OSError, match="ack_lost"):
        helper.stages(f, tmp_path)
    f.documents.write = write
    assert all(r["status"] == "accepted" for r in helper.stages(f, tmp_path)[0])
    assert f.ports.budget.used == 5


def test_active_stopped_control_cannot_start_observation(monkeypatch, tmp_path):
    from packages.planner_control.models import Command

    f = active_assembly(monkeypatch)
    state = f.planner.read(f.binding.run_id)
    f.planner.step(
        f.binding.run_id, Command(key="test-stop", action="stop", expected_revision=state.revision)
    )
    assert helper.stages(f, tmp_path)[0] == []
    assert f.ports.budget.used == 0 and not f.wire.calls


def test_active_current_source_rejection_precedes_http(monkeypatch, tmp_path):
    f = active_assembly(monkeypatch)

    def reject(_):
        raise ValueError("source_withdrawn")

    f.owner.owner.verify_dependencies = reject
    with pytest.raises(ValueError, match="source_withdrawn"):
        helper.stages(f, tmp_path)
    assert f.ports.budget.used == 0 and not f.wire.calls


def test_new_round_explicit_fresh_identity_and_original_provider_config():
    from pathlib import Path

    from apps.vectorengine_round import ActiveAcquisitionRound, load_round
    from apps.vectorengine_single_case import load_config
    from packages.grounding.models import validate

    r = load_round("configs/research/vectorengine_active_acquisition_v1_case.json")
    assert (
        r.run_id
        != load_round(
            "configs/research/vectorengine_batch_integrated_r01_case_runtime_r01.json"
        ).run_id
    )
    assert load_config(r.provider_config) == load_config(
        "configs/agents/vectorengine_batch_integrated_r01.json"
    )
    raw = json.loads(
        Path("configs/research/vectorengine_active_acquisition_v1_case.json").read_text()
    )
    raw["round_id"] += "-restart"
    with pytest.raises(ValueError):
        validate(ActiveAcquisitionRound, raw)


def test_web_extraction_preserves_inline_words_and_no_scripts():
    parser = Paragraphs()
    parser.feed("<script>ignore me</script><p>rim <b>width</b> &lt; 0.1</p>")
    assert parser.parts == ["rim width < 0.1"]


@pytest.mark.parametrize("target", RULES)
def test_definitions_reference_verified_packet_and_keep_scope(monkeypatch, target):
    from tests.agent_provider.support import materials

    owner = object.__new__(ActiveCaseEvidence)
    packet = materials()[0].state.material.sources[0]
    monkeypatch.setattr(owner, "packet", lambda: packet)
    value = owner.definition(target)
    assert value.source_refs == (packet.reference,)
    assert value.reference.checksum == digest(value.model_dump(mode="json", exclude={"reference"}))
    assert "SLO" in value.applicability_conditions
    assert owner.read_definition(target, value.reference) == value
    with pytest.raises(ValueError, match="binding"):
        owner.read_definition(
            target, value.reference.model_copy(update={"checksum": "sha256:" + "0" * 64})
        )


def test_archive_rechecks_current_owner_and_bytes(tmp_path):
    from db.knowledge import models as k
    from packages.object_store import LocalObjectStore, checksum

    # Use an immutable public citation fixture; no DB/network or patient access.
    entry = json.loads(
        __import__("pathlib").Path("docs/experiments/active-acquisition-v1/phene.json").read_text()
    )
    records = [
        k.parse_record_json(k.KnowledgeKind(r["kind"]), json.dumps(r)) for r in entry["records"]
    ]
    by_code = {r.code: r for r in records}
    chunks = {c["reference"]["code"]: c["text"].encode() for c in entry["chunks"]}

    class Repo:
        withdrawn = False

        def get_many(self, refs):
            return tuple(
                SimpleNamespace(
                    record=by_code.get(r.code, SimpleNamespace(enabled=True)),
                    withdrawn=self.withdrawn,
                    status=SimpleNamespace(value="draft"),
                )
                for r in refs
            )

    repo = Repo()
    # Withdrawal is checked before reading any object; a forged local cache cannot authorize it.
    repo.withdrawn = True
    with pytest.raises(ValueError, match="withdrawn"):
        verify(repo, LocalObjectStore(tmp_path), entry)
    bad = copy.deepcopy(entry)
    bad["schema_version"] = "invented"
    with pytest.raises(ValueError, match="identity"):
        verify(repo, LocalObjectStore(tmp_path), bad)
    assert all(checksum(body) == by_code[code].content_checksum for code, body in chunks.items())
