"""P2 capability boundaries without patients, database startup, or model calls."""

import json
import shutil
from types import SimpleNamespace

import pytest

from apps import deterministic_case_pipeline as visible_assembly
from apps import patient_service_assembly as assembly
from db.manage import ConnectionTarget
from packages.deterministic_pipeline.models import PipelineConfig, PipelineError
from packages.deterministic_pipeline.service import DeterministicCasePipeline
from packages.deterministic_pipeline.validation import DeterministicValidator
from packages.grounding.models import seal
from packages.object_store import LocalObjectStore, checksum
from tests.deterministic_pipeline.test_service_assembly_pg import (
    owned_object_roots as owned_object_roots,
)


def target(channel):
    return ConnectionTarget(
        "127.0.0.1", 5432, "mmagent_06e_benchmark_v1", channel, "TEST_BOUNDARY", None
    )


@pytest.fixture
def offline_connections(monkeypatch):
    """Replace only database I/O; retain real parsing, validator and composition."""
    checked = []
    monkeypatch.setattr(assembly, "check_runtime_bridge", lambda t: checked.append(t.user))
    reader = SimpleNamespace(_role="mmagent_knowledge_reader")

    def repository(t, *, capability):
        assert t.user == "knowledge_reader" and capability == "reader"
        return reader

    monkeypatch.setattr(assembly, "KnowledgeRepository", repository)
    for name in (
        "CasePipelineRepository",
        "PostgresAuditWriter",
        "PostgresAuditReader",
        "RuntimeGroundingLocks",
    ):
        monkeypatch.setattr(visible_assembly, name, lambda *_: object())
    return checked, reader


def test_knowledge_object_reader_exposes_only_promoted_bounded_reads(tmp_path):
    store = LocalObjectStore(tmp_path / "knowledge")
    reader = assembly.KnowledgeObjectReader(store)
    uri = store.stage(b"generic source fixture")
    with pytest.raises(ValueError, match="published_object_required"):
        reader.read(uri)
    uri = store.promote(uri, checksum(b"generic source fixture"))
    assert reader.read(uri) == b"generic source fixture"
    with pytest.raises(ValueError, match="exceeds"):
        reader.read_bounded(uri, 2)
    for method in ("stage", "promote", "discard", "root"):
        assert not hasattr(reader, method)


@pytest.mark.parametrize("purpose", ["visible", "knowledge"])
def test_other_roots_rejected_before_any_connection(tmp_path, purpose):
    with pytest.raises(assembly.AssemblyUnavailable, match="object_root_not_authorized"):
        assembly.check_object_root(tmp_path, purpose)


def test_resolver_only_gets_pipeline_channels_and_errors_hide_credentials(monkeypatch):
    monkeypatch.setattr(assembly, "check_object_root", lambda *_: None)
    requested = []

    def resolve(channel):
        requested.append(channel)
        return ConnectionTarget(
            "127.0.0.1", 5432, "wrong_database", channel, "TEST_SECRET", "never-print-this"
        )

    with pytest.raises(assembly.AssemblyUnavailable, match="target_or_identity") as error:
        assembly.assemble_patient_services(resolve)
    assert requested == ["runtime", "audit_writer", "audit_reader", "knowledge_reader"]
    assert "never-print-this" not in str(error.value)
    assert "evaluation" not in requested


def test_connection_failure_is_unavailable_and_never_runs_pipeline(monkeypatch):
    monkeypatch.setattr(assembly, "check_object_root", lambda *_: None)
    requested = []

    def resolve(channel):
        requested.append(channel)
        raise RuntimeError("secret-from-connection-boundary")

    with pytest.raises(assembly.AssemblyUnavailable, match="service_assembly_unavailable") as error:
        assembly.assemble_patient_services(resolve)
    assert "secret-from" not in str(error.value)
    assert requested == ["runtime"]


def test_constructed_services_still_explicitly_not_ready_for_patient_runs():
    from packages.deterministic_pipeline.models import PipelineConfig

    config = PipelineConfig.model_validate_json(
        (assembly.ROOT / "configs/deterministic_pipeline/fairvlmed_gap.json").read_bytes()
    )
    status = assembly.PatientServices(object(), config).readiness()
    assert status["service_assembly_available"]
    assert not status["content_publication_ready"]
    assert not status["patient_run_ready"]
    assert "mapping_not_bound" in status["reasons"]
    assert "p3_demand_gap_ports_not_implemented" in status["reasons"]


def test_factory_injects_visible_and_read_only_knowledge_without_privileged_channels(
    owned_object_roots, offline_connections, monkeypatch
):
    roots = owned_object_roots
    checked, reader = offline_connections
    injected = {}

    def compose(**kwargs):
        injected.update(kwargs)
        return visible_assembly.assemble_visible_pipeline(**kwargs)

    monkeypatch.setattr(assembly, "assemble_visible_pipeline", compose)
    channels = []

    def resolve(channel):
        channels.append(channel)
        return target(channel)

    services = assembly.assemble_patient_services(resolve)
    pipeline = services.pipeline
    assert type(pipeline) is DeterministicCasePipeline
    assert type(pipeline.validator) is DeterministicValidator
    assert services.config.checksum == (
        "sha256:fe15b81fa397960d8a11999bbf23a90f8ea3ab986d0f136cf5772a9985cd66bd"
    )
    assert pipeline.configs == {services.config.checksum: services.config}
    assert pipeline.validator.reference.model_dump(mode="json") == dict(
        code="POLICY:DETERMINISTIC_VALIDATION",
        version="1.1.0",
        checksum="sha256:721ef93b90a0afa968bfd4e0f457d65e66920e0f1ececf44fbde6e4ab2247574",
    )
    assert services.config.service_refs == (pipeline.validator.reference,)
    assert pipeline.catalog.pack(services.config.pack_ref).status == "Draft"
    assert services.config.mapping_refs == ()
    assert channels == list(assembly.CHANNELS)
    assert checked == ["runtime"]
    assert injected["knowledge_reader"] is reader
    assert injected["visible_store"].root == roots["visible"]
    assert isinstance(injected["knowledge_store"], assembly.KnowledgeObjectReader)
    assert not set(injected) & {"evaluation", "resolve_target", "tools", "mappings"}
    assert not set(vars(pipeline)) & {"evaluation", "evaluator_store", "resolve_target"}
    assert pipeline.tools is pipeline.mappings is None
    assert pipeline.validator.knowledge is pipeline.disease._repository is reader
    for method in ("stage", "promote", "discard", "root"):
        assert not hasattr(injected["knowledge_store"], method)
    assert not services.readiness()["content_publication_ready"]
    assert not services.readiness()["patient_run_ready"]


@pytest.mark.parametrize(
    "change",
    ["old_lock", "validator_version", "validator_code", "validator_checksum", "content", "mapping"],
)
def test_factory_rejects_rehashed_configuration_changes(
    tmp_path, owned_object_roots, monkeypatch, change
):
    raw = json.loads(
        (assembly.ROOT / "configs/deterministic_pipeline/fairvlmed_gap.json").read_bytes()
    )
    if change == "old_lock":
        raw["service_refs"][0].update(
            version="1.0.0",
            checksum="sha256:d419fc80b43ae2b60f368a9066424a96636f0baa010117470d529e334ea4d00d",
        )
    elif change == "validator_version":
        raw["service_refs"][0]["version"] = "1.1.1"
    elif change == "validator_code":
        raw["service_refs"][0]["code"] = "POLICY:UNREVIEWED"
    elif change == "validator_checksum":
        raw["service_refs"][0]["checksum"] = "sha256:" + "a" * 64
    elif change == "content":
        raw["query"] = "different content"
    else:
        raw["mapping_refs"] = [
            dict(
                kind="mapping",
                code="BINDING:OID_09ER01SYNTHETIC",
                version="1.0.0",
                checksum="sha256:" + "a" * 64,
            )
        ]
    raw.pop("checksum")
    config = seal(PipelineConfig, **raw)
    if change == "old_lock":
        assert config.checksum == (
            "sha256:98e8b7deec1ae5809dd2a073a29b45e4d3dc6719513fe8937ecb2ba92d3f0a82"
        )
    root = tmp_path / "config-root"
    shutil.copytree(assembly.ROOT / "configs/disease_packs", root / "configs/disease_packs")
    path = root / "configs/deterministic_pipeline/fairvlmed_gap.json"
    path.parent.mkdir()
    path.write_text(config.model_dump_json())
    monkeypatch.setattr(assembly, "ROOT", root)

    def forbidden(*_):
        pytest.fail("changed baseline must fail before database I/O")

    monkeypatch.setattr(assembly, "check_runtime_bridge", forbidden)
    with pytest.raises(
        assembly.AssemblyUnavailable, match="^p2_baseline_changed_requires_new_handoff$"
    ):
        assembly.assemble_patient_services(target)


@pytest.mark.parametrize("defect", ["host", "port", "database", "duplicate_identity"])
def test_factory_rejects_target_or_identity_before_connection(
    owned_object_roots, monkeypatch, defect
):
    from dataclasses import replace

    def resolve(channel):
        changes = {
            "host": dict(host="192.0.2.1"),
            "port": dict(port=5433),
            "database": dict(database="unapproved"),
            "duplicate_identity": dict(user="same"),
        }
        return replace(target(channel), **changes[defect])

    def forbidden(*_):
        pytest.fail("target must fail before database I/O")

    monkeypatch.setattr(assembly, "check_runtime_bridge", forbidden)
    with pytest.raises(assembly.AssemblyUnavailable, match="^service_target_or_identity_conflict$"):
        assembly.assemble_patient_services(resolve)


def test_factory_still_checks_installed_validator_lock(
    owned_object_roots, offline_connections, monkeypatch
):
    class WrongInstalledValidator(DeterministicValidator):
        def __init__(self, *args):
            super().__init__(*args)
            self.reference = self.reference.model_copy(update={"version": "1.0.0"})

    monkeypatch.setattr(visible_assembly, "DeterministicValidator", WrongInstalledValidator)

    def compose(**kwargs):
        with pytest.raises(PipelineError, match="^validator_version_not_locked$") as error:
            visible_assembly.assemble_visible_pipeline(**kwargs)
        raise error.value

    monkeypatch.setattr(assembly, "assemble_visible_pipeline", compose)
    with pytest.raises(assembly.AssemblyUnavailable, match="^service_assembly_unavailable$"):
        assembly.assemble_patient_services(target)
