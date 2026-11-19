"""Real packet owner, real persisted recovery; only external visual response injected."""

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from apps.qualitative_analyzer import main
from packages.dataset_adapters.patient_packet import prepare_patient
from packages.dataset_adapters.patient_storage import persist
from packages.llm_provider.transport import Reply
from packages.qualitative_analysis import runner
from packages.qualitative_analysis.storage import read, seal, write_once
from tests.dataset_adapters.test_multimodal import gamma

CONFIG = Path("configs/research/qualitative_analysis_v1.json").absolute()


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "source"
    gamma(root)
    b = io.BytesIO()
    Image.new("RGB", (8, 6), (50, 80, 90)).save(b, format="JPEG")
    for p in root.rglob("*.jpg"):
        p.write_bytes(b.getvalue())
    value = prepare_patient("gamma", root, "0001")
    packet_dir = tmp_path / "packet"
    persist(value, packet_dir, source_root=root)
    scope = dict(
        mode="archive_description",
        component_ids=[c.id for c in value.packet.components],
        background_ids=[],
    )
    out = tmp_path / "analysis"
    runner.plan(packet_dir, out, CONFIG, scope)
    return packet_dir, out


def candidate(view="V1", extra=None):
    value = dict(
        schema_version="qualitative-visual-result/1.0.0",
        disease_assessment="not_performed",
        annotations=[
            dict(
                category=c,
                label="label",
                value="visible content",
                basis="content_observation",
                view_id=view,
                location="center",
                evidence="pixels",
                certainty="tentative",
                limitation="sampled only",
            )
            for c in ["type_modality", "content_regions", "visibility"]
        ],
        unconfirmed=["unit unknown"],
    )
    if extra:
        value.update(extra)
    return json.dumps(value).encode()


class Transport:
    simulated = True

    def __init__(self, reply=None):
        self.calls = []
        self.reply = reply or Reply(content=candidate(), api_requests=0, response_confirmed=True)

    def send(self, config, body, **kw):
        self.calls.append(body)
        return self.reply


def test_independent_plan_real_packet_no_requirements_or_hidden_data(prepared):
    _, out = prepared
    p = read(out / "plan.json")
    assert len(p["jobs"]) == 2
    t = Transport()
    result = runner.run(out, live=True, transport=t)
    assert result["all_planned_jobs_completed"] and len(t.calls) == 2
    body = json.dumps(t.calls)
    for secret in [
        "SECRET_DIAGNOSIS",
        "SECRET_LEFT",
        "dataset",
        "advanced_glaucoma",
        "sample_selector",
    ]:
        assert secret not in body
    result_json = read(result["json"])
    assert result_json["disease_score"] is None and not result_json["runtime_ready"]
    assert result_json["execution"]["simulated"]
    assert all(r["semantics"]["source_unit"] is None for r in result_json["components"])
    runner.verify(out)
    runner.run(out, live=True, transport=t)
    assert len(t.calls) == 2


def test_local_only_and_scope_selection(prepared, tmp_path):
    packet, out = prepared
    t = Transport()
    result = runner.run(out, transport=t)
    assert not result["all_planned_jobs_completed"] and not t.calls
    dst = tmp_path / "selected"
    runner.plan(
        packet,
        dst,
        CONFIG,
        dict(mode="explicit_selection", component_ids=["C1"], background_ids=[]),
    )
    result = runner.run(dst, live=True, transport=t)
    assert len(t.calls) == 1
    assert read(result["json"])["scope_excluded"] == ["C2"]
    with pytest.raises(ValueError, match="scope_unknown"):
        runner.plan(
            packet,
            tmp_path / "bad",
            CONFIG,
            dict(mode="explicit_selection", component_ids=["C999"], background_ids=[]),
        )


def test_unknown_not_resent_and_other_jobs_continue(prepared):
    _, out = prepared
    t = Transport(Reply(error="timeout", api_requests=1, response_confirmed=None))
    result = runner.run(out, live=True, transport=t)
    assert set(result["jobs"].values()) == {"unknown"}
    assert len(t.calls) == 2
    runner.run(out, live=True, transport=t)
    assert len(t.calls) == 2


@pytest.mark.parametrize("mode", ["cancel", "expired", "budget"])
def test_control_guards(prepared, mode):
    _, out = prepared
    p = read(out / "plan.json")
    if mode == "cancel":
        write_once(out / "cancel.json", seal(dict(reason="test")))
    elif mode == "expired":
        write_once(
            out / "window.json", seal(dict(plan_checksum=p["checksum"], started_at=0, deadline=1))
        )
    else:
        # A changed frozen budget must be bound in a fresh plan, not mutated during recovery.
        p.pop("checksum")
        p["config"]["max_requests"] = 1
        (out / "plan.json").unlink()
        write_once(out / "plan.json", seal(p))
    t = Transport()
    result = runner.run(out, live=True, transport=t)
    assert len(t.calls) == (1 if mode == "budget" else 0)
    assert not result["all_planned_jobs_completed"]


@pytest.mark.parametrize("view,extra", [("V99", None), ("V1", {"disease_assessment": "glaucoma"})])
def test_wrong_response_rejected_without_correction_loop(prepared, view, extra):
    _, out = prepared
    t = Transport(Reply(content=candidate(view, extra), response_confirmed=True))
    result = runner.run(out, live=True, transport=t)
    assert set(result["jobs"].values()) == {"contract_rejected"}
    runner.run(out, live=True, transport=t)
    assert len(t.calls) == 2


def test_export_failure_cold_recovery_no_new_request(prepared, monkeypatch):
    _, out = prepared
    t = Transport()
    from packages.qualitative_analysis import report

    original = report.render

    def fail(_):
        raise OSError("injected html write failure")

    monkeypatch.setattr(report, "render", fail)
    with pytest.raises(OSError):
        runner.run(out, live=True, transport=t)
    assert len(list((out / "journal").glob("*.response.json"))) == 2
    monkeypatch.setattr(report, "render", original)
    first = runner.run(out, live=True, transport=t)
    second = runner.run(out, live=True, transport=t)
    assert first["json"] == second["json"] and len(t.calls) == 2
    assert Path(first["html"]).exists()


def test_inflight_intent_cold_recovery(prepared):
    _, out = prepared
    p = read(out / "plan.json")
    j = read(out / "jobs/J1.json")
    write_once(
        out / "journal/J1.intent.json",
        seal(
            dict(
                plan_checksum=p["checksum"],
                job_checksum=j["checksum"],
                request_checksum="sha256:" + "0" * 64,
                reservation=1,
                dispatched_at=0,
            )
        ),
    )
    t = Transport()
    result = runner.run(out, live=True, transport=t)
    assert result["jobs"]["J1"] == "unknown" and len(t.calls) == 1


@pytest.mark.parametrize("kind", ["view", "response", "packet"])
def test_tampering_rejected(prepared, kind):
    packet, out = prepared
    if kind == "view":
        next((out / "views").glob("*.png")).write_bytes(b"changed")
    elif kind == "packet":
        (packet / "case-input.json").write_text("{}")
    else:
        runner.run(out, live=True, transport=Transport())
        path = out / "journal/J1.response.json"
        r = read(path)
        r.pop("checksum")
        r["job_checksum"] = "sha256:" + "0" * 64
        path.unlink()
        write_once(path, seal(r))
    with pytest.raises((ValueError, RuntimeError)):
        runner.run(out, live=True, transport=Transport())


def test_cli_and_html_escape(prepared, capsys):
    _, out = prepared
    assert main(["status", "--input", str(out)]) == 0
    assert main(["verify", "--input", str(out)]) == 0
    t = Transport(
        Reply(
            content=candidate(extra={"unconfirmed": ["<script>alert(1)</script>"]}),
            response_confirmed=True,
        )
    )
    result = runner.run(out, live=True, transport=t)
    text = Path(result["html"]).read_text()
    assert "<script>" not in text and "&lt;script&gt;" in text


def test_local_lock_exclusion_and_host_binding(prepared):
    from packages.qualitative_analysis.storage import locked

    _, out = prepared
    with locked(out):
        with pytest.raises(BlockingIOError):
            with locked(out):
                pass
    host = out / "execution-host.json"
    host.unlink()
    write_once(host, seal(dict(boot_id="another-host", pid_namespace="other")))
    with pytest.raises(ValueError, match="immutable_record_conflict"):
        runner.run(out)


def test_source_unit_and_code_mapping_never_guessed():
    from types import SimpleNamespace

    from packages.qualitative_analysis.projection import unit_semantics

    assert (
        unit_semantics(SimpleNamespace(unit=None, unit_basis=None, value=93))["normalized_unit"]
        is None
    )
    r = unit_semantics(SimpleNamespace(unit="um", unit_basis="R1", value=93))
    assert r["normalized_unit"] == "µm" and r["normalized_value"] == 93 and r["unit_basis"] == "R1"
    r = unit_semantics(SimpleNamespace(unit="0=natural_lens;1=removed", unit_basis="R2", value=0))
    assert r["normalized_unit"] is None and r["category_mapping"]


def test_export_tampering_detected(prepared):
    _, out = prepared
    result = runner.run(out, live=True, transport=Transport())
    Path(result["html"]).write_text("changed")
    with pytest.raises(ValueError, match="export_binding"):
        runner.verify(out)


def test_no_semantic_candidate_overwrites_source(prepared):
    _, out = prepared
    value = json.loads(candidate())
    a = dict(value["annotations"][0])
    a.update(category="semantics", attribute="unit", label="unit", value="mm", basis="inference")
    value["annotations"].append(a)
    result = runner.run(
        out,
        live=True,
        transport=Transport(Reply(content=json.dumps(value).encode(), response_confirmed=True)),
    )
    rows = read(result["json"])["components"]
    assert all(
        r["semantics"]["source_unit"] is None and r["semantics"]["normalized_unit"] is None
        for r in rows
    )
    assert all(r["semantics"]["candidate_annotations"][0]["value"] == "mm" for r in rows)


def test_grape_slash_metadata_is_missing_in_new_packet(tmp_path):
    from packages.dataset_adapters.multimodal.formats import sheets
    from tests.dataset_adapters.test_multimodal import grape, workbook

    grape(tmp_path)
    path = tmp_path / "VF_and_clinical_information.xlsx"
    tables = {name: [dict(c) for _, c in rows] for name, rows in sheets(path.read_bytes())}
    tables["Follow-up"][0].update(G="Acquisition Device", H="Resolusion")
    tables["Follow-up"][2].update(G="/", H="/")
    workbook(path, tables)
    value = prepare_patient("grape", tmp_path, "1")
    selected = [
        b
        for b in value.packet.background
        if b.visit == "1" and b.concept in {"acquisition_device", "source_resolution"}
    ]
    assert len(selected) == 2
    assert all(b.value is None and b.status == "missing" for b in selected)
    assert all(
        "visits_after_cutoff" not in r.get("fields", []) for r in value.restricted["records"]
    )
