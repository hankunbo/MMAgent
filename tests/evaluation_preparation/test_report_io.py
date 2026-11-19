"""Local publication faults never mutate original facts or historical files."""

import json
from types import SimpleNamespace

import pytest

from packages.grounding.models import canonical
from packages.multimodal_research import report_io, stages


@pytest.mark.parametrize("cut", [0, 1, 12, -2])
def test_prefix_requires_authority_and_preserves_original(tmp_path, cut):
    value = {"text": "观察中文", "score": 0.03}
    payload = canonical(value) + b"\n"
    original = tmp_path / "report.json"
    fragment = payload[:cut]
    original.write_bytes(fragment)
    with pytest.raises(ValueError, match="partial_requires_recovery"):
        report_io.publish_json(original, value)
    calls = []
    recovered = report_io.publish_json(original, value, recover=lambda: calls.append(True))
    assert calls == [True] and recovered != original
    assert original.read_bytes() == fragment
    assert recovered.read_bytes() == payload
    assert report_io.publish_json(original, value, recover=lambda: None) == recovered
    recovered.write_bytes(payload[:4])
    with pytest.raises(ValueError, match="recovery_mirror_incomplete"):
        report_io.publish_json(original, value, recover=lambda: None)
    assert len(list(tmp_path.iterdir())) == 2


def test_semantic_json_and_conflicting_or_foreign_fragments(tmp_path):
    path = tmp_path / "value.json"
    value = {"score": 0, "label": "中性"}
    old = json.dumps(value, indent=4).encode()
    path.write_bytes(old)
    assert report_io.publish_json(path, value) == path
    assert path.read_bytes() == old
    for raw in (b'{"wrong":1}', b'{"wrong":', b"\xff"):
        path.write_bytes(raw)
        with pytest.raises(ValueError, match="content_conflict"):
            report_io.publish_json(path, value, recover=lambda: None)
        assert path.read_bytes() == raw


@pytest.mark.parametrize("where", ["write", "flush", "fsync", "link", "lost_ack"])
def test_atomic_failure_does_not_publish_partial(tmp_path, monkeypatch, where):
    path = tmp_path / "report.json"
    value = {"saved": True}
    fdopen, link = report_io.os.fdopen, report_io.os.link

    class BrokenStream:
        def __init__(self, *args):
            self.stream = fdopen(*args)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, payload):
            self.stream.write(payload[:3] if where == "write" else payload)
            if where == "write":
                raise OSError("injected write")
            return len(payload)

        def flush(self):
            raise OSError("injected flush")

    def fail(*args):
        if where == "lost_ack":
            link(*args)
        raise OSError("injected " + where)

    with monkeypatch.context() as patch:
        if where in {"write", "flush"}:
            patch.setattr(report_io.os, "fdopen", BrokenStream)
        else:
            patch.setattr(report_io.os, "fsync" if where == "fsync" else "link", fail)
        with pytest.raises(OSError, match="injected"):
            report_io.publish_json(path, value)
    assert path.exists() is (where == "lost_ack")
    assert not list(tmp_path.glob(".report-publish-*"))
    report_io.publish_json(path, value)
    assert json.loads(path.read_bytes()) == value


def test_html_recovery_and_denied_owner(tmp_path):
    path = tmp_path / "report.html"
    html = "<html>原报告中文</html>"
    raw = html.encode()[:9]
    path.write_bytes(raw)

    def denied():
        raise ValueError("source_withdrawn")

    with pytest.raises(ValueError, match="source_withdrawn"):
        report_io.publish_html(path, html, recover=denied)
    assert len(list(tmp_path.iterdir())) == 1
    recovered = report_io.publish_html(path, html, recover=lambda: None)
    assert recovered.read_text() == html and path.read_bytes() == raw


@pytest.mark.parametrize("local_closure", [False, True])
def test_recovery_context_covers_original_and_reconstructed_phase(tmp_path, local_closure):
    campaign = SimpleNamespace(output=tmp_path)
    review = {"schema_version": "dynamic-phase/1.0.0"}
    if local_closure:
        review = {"schema_version": "dynamic-phase-local-closure/1.0.0", "closure": {}}
    with pytest.raises(OSError):
        with stages.recovered_report(campaign, review):
            assert campaign.local_export_recovery is True
            raise OSError("exit")
    assert not hasattr(campaign, "local_export_recovery")


def test_durable_stage_receipt_survives_partial_local_mirror(tmp_path, monkeypatch):
    protocol = {"id": "test"}
    value = {"binding": stages.digest(protocol), "run_id": "RUN:SAVED"}
    (tmp_path / "stages").mkdir()
    path = tmp_path / "stages" / "export.json"
    payload = canonical(value) + b"\n"
    path.write_bytes(payload[:7])
    campaign = SimpleNamespace(
        output=tmp_path,
        protocol=protocol,
        documents=lambda run: SimpleNamespace(read=lambda key: (value,)),
    )
    monkeypatch.setattr(stages, "check_plan", lambda c: None)
    monkeypatch.setattr(stages, "durable_receipt", lambda c, stage: value)
    assert stages.verify(campaign, "export") == value
    assert path.read_bytes() == payload[:7]
    path.write_text('{"different":1}')
    with pytest.raises(ValueError, match="content_conflict"):
        stages.verify(campaign, "export")
