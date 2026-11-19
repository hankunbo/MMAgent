"""No dataset, Key, provider or tool is needed to export an exact stored report."""

import json
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import case_execution, generic_input, generic_run
from packages.grounding.models import canonical, digest
from packages.object_store import checksum


def fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = generic_input.directory("export-test")
    out.mkdir(parents=True)
    report = dict(result="synthetic", scores=None)
    values = [
        report,
        dict(html="<!doctype html><p>Synthetic result</p>", report_checksum=digest(report)),
    ]
    raw = {}
    artifacts = []
    for index, value in enumerate(values):
        uri = "synthetic:" + str(index)
        envelope = dict(
            schema_version="agent-runtime-document/1.0.0",
            run_id="RUN:SYNTHETIC",
            root_checksum=digest("root"),
            key_checksum=digest("dynamic-case-report"),
            index=index,
            value=value,
        )
        raw[uri] = canonical(envelope)
        a = SimpleNamespace(
            uri=uri, checksum=checksum(raw[uri]), scope=SimpleNamespace(case_code="CASE:SYNTHETIC")
        )
        a.model_dump = lambda *, mode, i=index: dict(index=i)
        artifacts.append(a)
    reference = dict(
        schema_version="generic-export-reference/1.0.0",
        run_key="export-test",
        bundle_key="TASK:SYNTHETIC",
        case_code="CASE:SYNTHETIC",
        run_id="RUN:SYNTHETIC",
        root_checksum=digest("root"),
        report_key_checksum=digest("dynamic-case-report"),
        artifacts=[a.model_dump(mode="json") for a in artifacts],
    )
    (out / "export-reference.json").write_text(json.dumps(reference))
    monkeypatch.setattr(
        case_execution,
        "owners",
        lambda: (
            SimpleNamespace(read_observation_bundle=lambda _: artifacts),
            SimpleNamespace(read_bounded=lambda uri, bound: raw[uri]),
            None,
        ),
    )
    monkeypatch.setattr(generic_run, "load", lambda _: pytest.fail("export cannot read dataset"))
    return out, raw, artifacts


def test_export_exact_retry_without_input_or_api(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    first = generic_run.export("export-test")
    assert first == generic_run.export("export-test")
    assert first["api_requests"] == first["model_starts"] == 0
    assert (tmp_path / first["html"]).read_text().startswith("<!doctype")


@pytest.mark.parametrize("change", ["content", "run", "case", "html"])
def test_export_rejects_tampering_and_does_not_overwrite(tmp_path, monkeypatch, change):
    _, raw, artifacts = fixture(tmp_path, monkeypatch)
    generic_run.export("export-test")
    path = tmp_path / "docs/exports/FairVLMed_export-test.html"
    old = path.read_bytes()
    if change == "content":
        raw[artifacts[0].uri] += b" "
    elif change == "run":
        value = json.loads(raw[artifacts[0].uri])
        value["run_id"] = "RUN:OTHER"
        raw[artifacts[0].uri] = canonical(value)
        artifacts[0].checksum = checksum(raw[artifacts[0].uri])
    elif change == "case":
        artifacts[0].scope.case_code = "CASE:OTHER"
    else:
        path.write_text("existing user file")
        old = path.read_bytes()
    with pytest.raises(ValueError):
        generic_run.export("export-test")
    assert path.read_bytes() == old


def test_recover_requires_existing_operation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = generic_input.directory("not-started")
    out.mkdir(parents=True)
    monkeypatch.setattr(generic_run, "load", lambda _: {})
    monkeypatch.setattr(generic_run, "register", lambda _: pytest.fail("must not register"))
    with pytest.raises(ValueError, match="existing_run"):
        generic_run.run("not-started", recover=True)


def test_new_run_does_not_silently_recover(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = generic_input.directory("started")
    out.mkdir(parents=True)
    (out / "started.json").write_text("{}")
    monkeypatch.setattr(generic_run, "load", lambda _: {})
    monkeypatch.setattr(generic_run, "register", lambda _: pytest.fail("must not register"))
    with pytest.raises(ValueError, match="use_recover_or_export"):
        generic_run.run("started")
