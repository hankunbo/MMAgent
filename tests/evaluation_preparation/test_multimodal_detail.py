"""Fine input projections never invent anatomy, labels, probabilities or votes."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.preparation import persist
from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.grounding.models import canonical, digest
from packages.multimodal_research import material
from packages.multimodal_research.correspondence import describe
from packages.multimodal_research.numeric_detail import expand
from packages.multimodal_research.single_proposition import REGISTRY, selected
from tests.dataset_adapters.test_multimodal import gamma, gdp
from tests.evaluation_preparation.test_multimodal_campaign import sources


@pytest.mark.parametrize(
    "label", ["non_glaucoma", "early_glaucoma", "intermediate_glaucoma", "advanced_glaucoma"]
)
def test_binary_single_proposition_definitions_no_joint_policy(tmp_path, label):
    p = prepare("gamma", gamma(tmp_path), "0001")
    task = load_intent(label, REGISTRY)[0]
    assert task.output_kind == "binary_evidence" and not task.ordered_labels
    assert selected(task)[0] == label
    ledger = material.requirements(p.manifest, sources().initial, task)
    assert len(ledger) == 7 and len({r["id"] for r in ledger}) == 7
    assert all("e>0支持这个命题" in r["definition"].applicability_conditions for r in ledger)


def test_numeric_complete_vectors_order_negative_zero_null_and_region_coordinates(tmp_path):
    p = prepare("harvard_gdp", gdp(tmp_path / "source", nonfinite=True), "data_0001.npz")
    ledger = material.requirements(
        p.manifest, sources().initial, load_intent("glaucoma_detection")[0]
    )
    payloads = dict(p.payload_bytes)
    rows, observations = expand(ledger, {}, p.manifest, payloads)
    chunks = [json.loads(observations[r["id"]].value) for r in rows if ":values:" in r["id"]]
    assert sum(len(c["values"]) for c in chunks) == 52
    assert sum(v is None for c in chunks for v in c["values"]) == 1
    assert [c["source_range"] for c in chunks] == [[0, 24], [24, 48], [48, 52]]
    assert all(len(o.value) <= 1000 for o in observations.values())
    assert "99123" not in canonical(observations).decode()
    regions = [json.loads(o.value) for k, o in observations.items() if ":region:" in k]
    assert len(regions) == 4 and all("not_anatomical" in r["meaning"] for r in regions)


def test_detail_worker_real_images_manifest_and_legacy_unchanged(tmp_path):
    root = gamma(tmp_path / "source")
    # The adapter fixture uses header-only JPEGs. This worker test needs decoded pixels.
    encoded = subprocess.run(
        [
            "/opt/conda/bin/python",
            "-c",
            "import io,sys; from PIL import Image; b=io.BytesIO(); "
            "Image.new('RGB',(8,8),'white').save(b,format='JPEG'); "
            "sys.stdout.buffer.write(b.getvalue())",
        ],
        capture_output=True,
        check=True,
    ).stdout
    for file in root.rglob("*.jpg"):
        file.write_bytes(encoded)
    p = prepare("gamma", root, "0001")
    path = persist(p, tmp_path / "prepared")
    for profile in ("overview-v1", "detail-v1"):
        result = subprocess.run(
            [
                "/opt/conda/bin/python",
                "-m",
                "packages.multimodal_research.views_worker",
                str(path),
                str(tmp_path / profile),
                "--profile",
                profile,
            ],
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
    old = json.loads((tmp_path / "overview-v1/views.json").read_bytes())
    new = json.loads((tmp_path / "detail-v1/views.json").read_bytes())
    assert old["schema_version"] == "multimodal-inspection-views/1.0.0"
    assert new["schema_version"] == "multimodal-inspection-views/1.1.0"
    a, b = (next(v for v in x["views"] if v["component"] == "oct") for x in (old, new))
    assert a["indices"] == b["indices"] and b["width"] == 2 * a["width"]
    assert a["input_refs"] == b["input_refs"] and b["transform"]["anatomical_registration"] is False


def test_correspondence_unknown_eye_different_visit_no_extra_vote():
    def c(i, m, e, t):
        return SimpleNamespace(id=i, modality=m, laterality=e, visit=t, status="obtained")

    value = describe(
        SimpleNamespace(
            checksum=digest("input"),
            components=[
                c("a", "OCT", "UNKNOWN", "0"),
                c("b", "VF", "UNKNOWN", "0"),
                c("c", "fundus", "OD", "1"),
            ],
        )
    )
    assert value["new_score_rows"] == 0
    assert all(not r["same_eye_visit_eligible"] for r in value["pairs"])


@pytest.mark.parametrize(
    "error,pending,cancelled,expected",
    [
        ("schema", None, False, False),
        ("timeout", None, False, False),
        ("scope", None, False, False),
        ("schema", object(), False, False),
        ("schema", None, True, False),
    ],
)
def test_outer_error_without_original_journal_never_authorizes_continue(
    error, pending, cancelled, expected, monkeypatch
):
    from packages.multimodal_research.campaign import MultimodalCampaign
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    monkeypatch.setattr(RuntimeAgentJournal, "read", lambda _: ())
    campaign = object.__new__(MultimodalCampaign)
    campaign.base = SimpleNamespace(detailed=True)
    campaign.sessions = SimpleNamespace(
        read=lambda _: SimpleNamespace(phase="stopped", pending=pending, cancelled=cancelled)
    )
    campaign.guard = lambda: None
    campaign.documents = lambda _: None
    assert (
        campaign.local_acquisition_gap(dict(target="check:a", error=error, run_id="RUN:X"))
        is expected
    )


@pytest.mark.parametrize(
    "error,role,completed,expected",
    [
        ("timeout", "vision", True, "timeout"),
        ("schema", "vision", True, "schema"),
        ("timeout", "vision", False, None),
        ("scope", "vision", True, None),
        ("auth", "vision", True, None),
        ("schema", "review", True, None),
    ],
)
def test_local_read_only_outcome_requires_completed_journal(error, role, completed, expected):
    from packages.multimodal_research.local_outcomes import reason

    events = [dict(phase="reserved", request_id="request:x")]
    if completed:
        events.append(
            dict(phase="completed", record=dict(request_id="request:x", role=role, error=error))
        )
    assert reason(events) == expected
