"""Classification semantics without torch, datasets, labels or model execution."""

import inspect
import json
import math
import struct

import pytest

from packages.fairclip_research.core import (
    CLASSES,
    CONFIG_CHECKSUM,
    TEXTS,
    WEIGHT_SHA256,
    classify_features,
    scores_from_cosines,
)
from packages.fairclip_research.models import FairCLIPOutput, read_output
from packages.tool_admission.models import AdmissionPolicy


def output(cosines=(0.2, 0.8)):
    scores, index = scores_from_cosines(cosines)
    return dict(
        input_checksum="sha256:" + "a" * 64,
        model_checksum="sha256:" + WEIGHT_SHA256,
        config_checksum=CONFIG_CHECKSUM,
        raw_cosines=cosines,
        normalized_model_scores=scores,
        predicted_index=index,
        predicted_class=CLASSES[index],
    )


def test_fixed_class_order_softmax_and_no_case_label_port():
    cosines, scores, index = classify_features([3.0, 4.0], [[1.0, 0.0], [0.0, 1.0]])
    assert cosines == (0.6, 0.8) and index == 1
    expected = math.exp(0.8) / (math.exp(0.6) + math.exp(0.8))
    assert scores[1] == struct.unpack("e", struct.pack("e", expected))[0]
    assert TEXTS == ("A photo of non-glaucoma", "A photo of glaucoma")
    assert tuple(inspect.signature(classify_features).parameters) == ("image", "texts")
    assert scores_from_cosines((0.3, 0.3)) == ((0.5, 0.5), 0)
    assert FairCLIPOutput(**output()).calibration == "unverified"


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_checksum", "sha256:" + "b" * 64),
        ("config_checksum", "sha256:" + "b" * 64),
        ("raw_cosines", (float("nan"), 0.0)),
        ("raw_cosines", (float("inf"), 0.0)),
        ("normalized_model_scores", (0.01, 0.99)),
        ("predicted_index", 0),
        ("predicted_class", "non-glaucoma"),
        ("calibration", "validated"),
        ("demographics_used", ("race",)),
    ],
)
def test_output_drift_invalid_scores_and_semantics_rejected(field, value):
    with pytest.raises(ValueError):
        FairCLIPOutput(**{**output(), field: value})


def test_invalid_features_and_bounded_large_resource():
    for image in ([0.0, 0.0], [float("nan"), 0.0], [float("inf"), 0.0]):
        with pytest.raises(ValueError):
            classify_features(image, [[1.0, 0.0], [0.0, 1.0]])
    assert AdmissionPolicy(max_total_bytes=1500000000).max_total_bytes == 1500000000
    assert AdmissionPolicy().max_total_bytes == 67108864
    with pytest.raises(ValueError):
        AdmissionPolicy(max_total_bytes=2147483649)


def test_duplicate_json_and_resource_drift_rejected(tmp_path):
    from packages.fairclip_research.backend import module_from_reviewed_file, verify_checkpoint

    raw = json.dumps(output())
    with pytest.raises(ValueError, match="duplicate_field"):
        read_output(raw[:-1] + ', "predicted_index": 0}')
    malicious = tmp_path / "unreviewed.py"
    malicious.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="resource_drift"):
        module_from_reviewed_file(malicious, "0" * 64)
    with pytest.raises(ValueError, match="weight_binding"):
        verify_checkpoint(malicious)


def test_pending_cli_does_not_construct_deployment(monkeypatch, capsys, tmp_path):
    from pathlib import Path

    from apps import research_single_case
    from packages.grounding.models import digest

    # A real local installation must not change this uninstalled-configuration test.
    config = json.loads(Path("configs/research/fairclip_single_case.json").read_bytes())
    config["installed_bundle"] = None
    config["checksum"] = digest({k: v for k, v in config.items() if k != "checksum"})
    path = tmp_path / "uninstalled.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(research_single_case, "assemble", lambda *a: pytest.fail("uninstalled"))
    assert (
        research_single_case.main(["--config", str(path), "--execute", "--key", "not-a-case"]) == 2
    )
    assert json.loads(capsys.readouterr().out)["execution_authorized"] is False


def test_preprocessing_configuration_drift(tmp_path):
    from pathlib import Path

    from packages.grounding.models import digest
    from packages.research_runtime.deployment import load_configuration

    doc = json.loads(Path("configs/research/fairclip_single_case.json").read_text())
    doc["classification"]["preprocess"]["resize_short_edge"] = 128
    doc["checksum"] = digest({k: v for k, v in doc.items() if k != "checksum"})
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="classification_drift"):
        load_configuration(path)


def test_H_decoder_fixed_sources_and_clinical_version_unchanged(tmp_path):
    from packages.case_score_ledger.models import ScoreDefinition
    from packages.case_score_ledger.service import CaseScoreService, ToolObservation
    from packages.contracts import VersionedReference
    from tests.tool_workflow.support import prepared

    ready = prepared(tmp_path)
    plan = ready.compiler.compile(ready.context, ready.choice)

    definition = ScoreDefinition(
        code="fairclip.raw.0",
        layer="model",
        metric="logit",
        unit="logit",
        definition="Uncalibrated fixed class cosine",
        direction="higher",
    )
    observation = ToolObservation(
        key="test",
        plan_json=plan.to_bundle_json(),
        compiler=ready.compiler,
        outputs=(("ARTIFACT:FAIRCLIP", definition, "fairclip-output/1.0.0", "raw_cosine_0"),),
        model_versions=(
            VersionedReference(
                code="SOURCE:FAIRCLIP_WEIGHT", version="1.0.0", checksum="sha256:" + WEIGHT_SHA256
            ),
        ),
    )
    options = dict(
        runtime=None, executions=None, objects=None, observations=(observation,), synthetic=True
    )
    with pytest.raises(ValueError, match="research_score_only"):
        CaseScoreService(**options)
    service = CaseScoreService(**options, context_version="research-run/1.0.0")
    assert service._policy()["version"] == "score-assembly/1.2.0"
    assert (
        CaseScoreService(
            runtime=None, executions=None, objects=None, observations=(), synthetic=True
        )._policy()["version"]
        == "score-assembly/1.1.0"
    )
    assert (
        service._decode(
            json.dumps(output()).encode(), "fairclip-output/1.0.0", "raw_cosine_0", definition
        )
        == 0.2
    )
    with pytest.raises(ValueError, match="score_field"):
        service._decode(
            json.dumps(output()).encode(), "fairclip-output/1.0.0", "probability", definition
        )
    from dataclasses import replace

    with pytest.raises(ValueError, match="fixed_model_version"):
        CaseScoreService(
            **{**options, "observations": (replace(observation, model_versions=()),)},
            context_version="research-run/1.0.0",
        )


def test_report_predictions_only_from_available_H_objects(tmp_path):
    from types import SimpleNamespace

    from packages.object_store import LocalObjectStore, checksum
    from packages.research_runtime.single_case import verified_model_outputs

    objects = LocalObjectStore(tmp_path / "objects")
    data = FairCLIPOutput(**output()).model_dump_json().encode()
    uri = objects.promote(objects.stage(data), checksum(data))

    class Source(SimpleNamespace):
        def model_dump(self, **kw):
            return {"artifact_checksum": checksum(data)}

    source = Source(
        output_schema_version="fairclip-output/1.0.0",
        object_ref=SimpleNamespace(uri=uri, byte_size=len(data), checksum=checksum(data)),
    )
    entry = SimpleNamespace(source=source, status="available")
    ledger = SimpleNamespace(entries=(entry, entry))
    assert len(verified_model_outputs(ledger, objects)) == 1
    assert verified_model_outputs(ledger, objects)[0]["output"]["predicted_class"] == "glaucoma"
    entry.status = "unavailable"
    assert verified_model_outputs(ledger, objects) == []
    entry.status = "available"
    objects._path(uri).write_bytes(b"{}")
    with pytest.raises(ValueError, match="source_drift"):
        verified_model_outputs(ledger, objects)


def test_invalid_key_rejected_before_creating_case():
    from packages.research_runtime.single_case import ResearchSingleCaseService

    service = ResearchSingleCaseService(
        runtime=None,
        research=None,
        choice=None,
        bind_plan=None,
        config_checksum="unused",
        budget_seconds=1,
    )
    for key in ("", "x" * 81, "unsafe key", "标签", None):
        with pytest.raises(ValueError, match="single_case_key"):
            service.run(None, key=key)


def test_unbound_installation_cannot_claim_a_case_plan():
    from packages.research_runtime.readiness import UnboundSingleCasePolicy
    from packages.tool_execution.trusted_models import TrustedLocalPolicy

    assert UnboundSingleCasePolicy.model_fields["allowed_plan_checksums"].default == ()
    assert TrustedLocalPolicy.model_fields["allowed_plan_checksums"].is_required()


def test_json_QC_requires_exact_input_and_output_definition():
    from types import SimpleNamespace

    from packages.fairclip_research.qc import FairCLIPQCPolicy, FairCLIPQCRegistry
    from packages.grounding.models import Term, digest
    from packages.grounding.synthetic import ref

    artifact = ref("artifact", "ARTIFACT:FAIRCLIP")
    unit = Term(state="known", code="CONCEPT:UNIT_LOGIT")
    policy = FairCLIPQCPolicy(
        artifact_ref=artifact, artifact_type_code="ARTIFACT:FAIRCLIP", unit_term=unit
    )
    qc_ref = ref("policy", "POLICY:FAIRCLIP").model_copy(update={"checksum": digest(policy)})
    registry = FairCLIPQCRegistry({qc_ref: policy})
    spec = SimpleNamespace(
        qc_policy_ref=qc_ref,
        qc_port=registry.port,
        constraint=SimpleNamespace(artifact_ref=artifact, unit=unit),
    )
    rule = SimpleNamespace(
        media_types=("application/json",), dimensions_min=(2,), dimensions_max=(2,)
    )
    raw = json.dumps(output()).encode()
    inputs = (SimpleNamespace(checksum=output()["input_checksum"]),)
    assert registry.evaluate(spec, rule, raw, inputs).status.value == "PASS"
    with pytest.raises(ValueError, match="input_drift"):
        registry.evaluate(spec, rule, raw, (SimpleNamespace(checksum="sha256:" + "b" * 64),))
    bad = {**output(), "normalized_model_scores": [0.0, 1.0]}
    with pytest.raises(ValueError, match="softmax_mismatch"):
        registry.evaluate(spec, rule, json.dumps(bad).encode(), inputs)


def test_controlled_rejection_without_attempt_finishes_failed(monkeypatch):
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from packages.research_runtime.single_case import ResearchSingleCaseService

    runtime = SimpleNamespace(run_status=lambda rid: "EXECUTING")
    driver = ResearchSingleCaseService(
        runtime=runtime,
        research=None,
        choice=None,
        bind_plan=None,
        config_checksum="unused",
        budget_seconds=60,
    )
    transitions = []

    def record(state, phase, reason=None, target=None, report=None):
        transitions.append((phase, reason, target))
        return {**state, "phase": phase}

    monkeypatch.setattr(driver, "_record", record)
    monkeypatch.setattr(driver, "_finish", lambda state, scores: state)
    state = dict(
        run_id="synthetic",
        phase="PREPARED",
        binding={"execution_key": "key"},
        deadline=(datetime.now(UTC) + timedelta(seconds=60)).isoformat(),
    )
    executor = SimpleNamespace(
        recover=lambda *a: None,
        execute=lambda *a, **kw: SimpleNamespace(state="technical_rejection"),
    )
    assert driver._drive(state, None, executor, None, None, False)["phase"] == "SCORING"
    assert transitions[-1] == (
        "SCORING",
        "controlled_tool_rejected_before_attempt",
        "FAILED_TECHNICAL",
    )


def test_deployment_JSON_references_use_original_07A_types(tmp_path):
    from packages.research_runtime.deployment import installed_training_adapter
    from tests.dataset_adapters.support import fair

    adapter, _ = fair(tmp_path / "synthetic")
    selection = dict(
        root=str(adapter.root),
        dataset_ref=adapter.dataset.reference().model_dump(mode="json"),
        pack_ref=adapter.pack.reference().model_dump(mode="json"),
    )
    installed = installed_training_adapter(selection, adapter.catalog)
    assert installed.dataset == adapter.dataset and installed.pack == adapter.pack
    sample = installed.prepare("data_00000.npz", split="training")
    assert len(sample.images) == 1
