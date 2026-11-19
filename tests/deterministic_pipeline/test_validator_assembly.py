"""Software-only port installation; no approved medical rule or real model is supplied."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from apps import deterministic_case_pipeline as assembly
from packages.contracts import VersionedReference
from packages.deterministic_pipeline.models import PipelineConfig, PipelineError
from packages.deterministic_pipeline.validation import DeterministicValidator
from packages.deterministic_pipeline.validation_ports import assemble_evidence_validator
from packages.graph_engine import canonical_checksum
from packages.grounding.models import seal
from tests.deterministic_pipeline.test_reporting import facts as facts


def reseal(value):
    return seal(type(value), **value.model_dump(mode="json", exclude={"checksum"}))


def config():
    return PipelineConfig.model_validate_json(
        Path("configs/deterministic_pipeline/fairvlmed_gap.json").read_bytes()
    )


class SyntheticPortFixture(DeterministicValidator):
    """An installation fixture; deliberately inherits the real-input refusal."""

    def __init__(self, knowledge, objects, published):
        super().__init__(knowledge, objects, published)
        self.reference = VersionedReference(
            code="POLICY:SYNTHETIC_VALIDATOR_PORT_TEST",
            version="1.0.0",
            checksum=canonical_checksum({"synthetic_port_fixture": 1}),
        )


def locked_config():
    return reseal(
        config().model_copy(
            update={"service_refs": (SyntheticPortFixture(None, None, None).reference,)}
        )
    )


def test_legacy_reference_and_configuration_unchanged():
    validator, configs = assemble_evidence_validator(
        DeterministicValidator, None, None, None, (config(),)
    )
    assert validator.reference == config().service_refs[0]
    assert configs == (config(),)


def test_explicit_factory_receives_original_readers_and_requires_new_lock():
    knowledge, objects, published = object(), object(), object()
    with pytest.raises(PipelineError, match="validator_version_not_locked"):
        assemble_evidence_validator(
            SyntheticPortFixture, knowledge, objects, published, (config(),)
        )
    validator, configs = assemble_evidence_validator(
        SyntheticPortFixture, knowledge, objects, published, (locked_config(),)
    )
    assert validator.knowledge is knowledge and validator.objects is objects
    assert validator.published is published
    assert configs[0].checksum != config().checksum


@pytest.mark.parametrize("change", ["version", "checksum"])
def test_old_or_wrong_reference_cannot_install(change):
    cfg = locked_config()
    update = {change: "2.0.0" if change == "version" else "sha256:" + "a" * 64}
    cfg = reseal(
        cfg.model_copy(update={"service_refs": (cfg.service_refs[0].model_copy(update=update),)})
    )
    with pytest.raises(PipelineError, match="validator_version_not_locked"):
        assemble_evidence_validator(SyntheticPortFixture, None, None, None, (cfg,))


@pytest.mark.parametrize(
    "factory", [None, "module:caller_supplied", lambda *_: None, lambda *_: object()]
)
def test_invalid_factory_does_not_fall_back(factory):
    with pytest.raises(PipelineError, match="invalid_evidence_validator_installation"):
        assemble_evidence_validator(factory, None, None, None, (config(),))


def test_factory_error_has_no_raw_input_or_secret():
    def broken(*_):
        raise ValueError("PRIVATE_FIXTURE_VALUE")

    with pytest.raises(PipelineError) as error:
        assemble_evidence_validator(broken, None, None, None, (config(),))
    assert str(error.value) == "invalid_evidence_validator_installation"


def test_constructed_invalid_config_revalidated():
    cfg = locked_config().model_copy(update={"checksum": "sha256:" + "b" * 64})
    with pytest.raises(PipelineError, match="invalid_evidence_validator_installation"):
        assemble_evidence_validator(SyntheticPortFixture, None, None, None, (cfg,))


def test_installation_never_makes_nondecisive_or_unreviewed_outputs_eligible(facts):
    validator, _ = assemble_evidence_validator(
        SyntheticPortFixture, None, None, None, (locked_config(),)
    )
    assert validator.eligible(
        SimpleNamespace(synthetic=False), None, None, facts.step, facts.plan
    ) == (False, "nondecisive_research_observation")
    # 06 technical records cannot self-authorize medical evidence, even after installation.
    from packages.grounding.models import validate
    from packages.tool_execution.models import StepResult

    step = facts.step.model_copy(
        update={"result": facts.step.result.model_copy(update={"decisive_evidence": True})}
    )
    with pytest.raises(ValueError, match="executor_cannot_approve_evidence"):
        validate(StepResult, step)


def test_public_composition_rejects_unlocked_factory_before_runtime(monkeypatch):
    monkeypatch.setattr(assembly, "PublishedKnowledgeService", lambda *_: object())

    def forbidden(*_):
        pytest.fail("Runtime assembly must not occur before validator lock check")

    monkeypatch.setattr(assembly, "CasePipelineRepository", forbidden)
    with pytest.raises(PipelineError, match="validator_version_not_locked"):
        assembly.assemble_visible_pipeline(
            runtime_target=None,
            audit_writer_target=None,
            audit_reader_target=None,
            knowledge_reader=None,
            knowledge_store=None,
            visible_store=None,
            catalog=None,
            configs=(config(),),
            evidence_validator_factory=SyntheticPortFixture,
        )


def test_public_composition_installs_exact_fixture_without_executing_it(monkeypatch):
    for name in (
        "CasePipelineRepository",
        "PublishedKnowledgeService",
        "PostgresAuditWriter",
        "PostgresAuditReader",
        "DatasetInputRepository",
        "VisibleInputReader",
        "DiseasePackService",
        "RuntimeGroundingLocks",
        "DemandResolver",
    ):
        monkeypatch.setattr(assembly, name, lambda *args, **kwargs: object())
    monkeypatch.setattr(
        assembly, "DeterministicCasePipeline", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    value = assembly.assemble_visible_pipeline(
        runtime_target=None,
        audit_writer_target=None,
        audit_reader_target=None,
        knowledge_reader=None,
        knowledge_store=None,
        visible_store=None,
        catalog=None,
        configs=(locked_config(),),
        evidence_validator_factory=SyntheticPortFixture,
    )
    assert type(value.validator) is SyntheticPortFixture
    assert value.validator.reference in value.configs[0].service_refs


def test_empty_config_or_one_unlocked_consumer_rejected():
    for configs in ((), (locked_config(), config())):
        with pytest.raises(PipelineError, match="validator_version_not_locked"):
            assemble_evidence_validator(SyntheticPortFixture, None, None, None, configs)
