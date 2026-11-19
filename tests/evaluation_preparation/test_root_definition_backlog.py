"""Confirmed empty proposal + unknown later Source does not erase baseline work."""

import copy
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.discovery_continuation import ConfirmedDefinitionContinuation
from tests.evaluation_preparation.test_confirmed_definition_backlog import preceding_empty_proposal

MODE = "root_technical_backlog_no_source_replay"
CONFIRMED_MODE = "root_confirmed_schema_backlog_no_source_replay"


def root_parent(tmp_path, *, confirmed_schema=False):
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.multimodal_research.campaign import configuration
    from packages.multimodal_research.single_proposition import root_direction_spec

    c, ref, failure, records, writes = preceding_empty_proposal(tmp_path)
    parent = Path(ref["parent"])
    spec = json.loads((parent / "entry.json").read_bytes())
    plan = json.loads((parent / "protocol.json").read_bytes())
    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    config = configuration(
        directional=True,
        resilient_vision=True,
        exact_primary_status=True,
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
    )
    path = (
        "configs/research/multimodal_structural_coverage_v1.json"
        if confirmed_schema
        else "configs/research/multimodal_harvard_visual_priority_v2.json"
    )
    spec.update(
        schema_version="multimodal-live-entry/1.26.0",
        dataset="harvard_gdp",
        input_semantics="harvard-gdp-v1",
        intent=task.model_dump(mode="json"),
        provider=config.model_dump(mode="json"),
        execution_profile="relay-root-question-v1",
        root_direction_review=root_direction_spec(task),
        review_batch_size=3,
        active_acquisition=dict(path=path, checksum=digest(json.loads(Path(path).read_bytes()))),
    )
    plan.update(
        schema_version="multimodal-live-campaign/1.16.0",
        task=task.checksum,
        root_direction_review=spec["root_direction_review"],
        review_batch_size=3,
        execution_profile=spec["execution_profile"],
        provider=digest(config),
    )
    plan["started_at"] = (
        datetime.fromisoformat(plan["deadline"]) - timedelta(seconds=5400)
    ).isoformat()
    failure.update(
        error_code="timeout", failure_kind="dispatched_unknown", protocol_checksum=digest(plan)
    )
    failure["transport"]["remote_response_unknown"] = 1
    failed = records["journal:3"][0]["record"]
    failed.update(error="timeout", output_checksum=None)
    records["definition:response:" + failed["request_id"]][0]["raw_text"] = ""
    if confirmed_schema:
        failure.update(error_code="schema", failure_kind="confirmed_response_or_domain_refusal")
        failure["transport"]["remote_response_unknown"] = 0
        text = '{"definitions":[{"method":null}]}'
        failed.update(error="schema", output_checksum=digest(text.encode().hex()))
        records["definition:response:" + failed["request_id"]][0]["raw_text"] = text
    c.base.task, c.config = task, config
    for name, value in (("entry", spec), ("protocol", plan), ("stage-failure", failure)):
        (parent / (name + ".json")).write_text(json.dumps(value))
    ref.update(
        entry=digest(spec),
        protocol=digest(plan),
        failure=digest(failure),
        mode=CONFIRMED_MODE if confirmed_schema else MODE,
    )
    return c, ref, spec, plan, failure, records, writes


def test_root_unknown_keeps_public_wire_and_original_single_claim(tmp_path):
    c, ref, spec, plan, failure, records, writes = root_parent(tmp_path)
    before = copy.deepcopy(records)
    value = ConfirmedDefinitionContinuation(c, ref)
    assert value.root_backlog and value.defer and not value.confirmed_backlog
    assert value.check_attempts()[0]["requests"] == 3
    assert len(value.wire_proofs) == 1
    assert (
        ConfirmedDefinitionContinuation(c, ref, read_only=True).check_attempts() == value.attempts
    )
    assert ConfirmedDefinitionContinuation(c, ref).check_attempts() == value.attempts
    assert len(writes) == 1 and all(records[k] == v for k, v in before.items())
    assert json.loads((Path(ref["parent"]) / "protocol.json").read_bytes()) == plan
    assert json.loads((Path(ref["parent"]) / "stage-failure.json").read_bytes()) == failure


@pytest.mark.parametrize(
    "mutation", ["wire_unknown", "first_unresolved", "increment", "batch", "question"]
)
def test_root_backlog_rejects_unconfirmed_material_or_changed_binding(tmp_path, mutation):
    c, ref, spec, _, _, records, writes = root_parent(tmp_path)
    if mutation == "wire_unknown":
        records.pop("definition:wire:0:outcome")
    elif mutation == "first_unresolved":
        records.pop("definition:gap:0:decision")
    elif mutation == "increment":
        records["definition:gap:0:decision"][0]["candidate"]["added"] = ["useful"]
    else:
        if mutation == "batch":
            spec["review_batch_size"] = 1
        else:
            spec["root_direction_review"]["title"] = "other"
        (Path(ref["parent"]) / "entry.json").write_text(json.dumps(spec))
        ref["entry"] = digest(spec)
    expected = {
        "wire_unknown": "definition_successor_wire_unknown",
        "first_unresolved": "definition_successor_uncommitted_increment",
        "increment": "definition_successor_uncommitted_increment",
        "batch": "definition_successor_root_binding",
        "question": "definition_successor_root_binding",
    }[mutation]
    with pytest.raises(ValueError, match=expected):
        ConfirmedDefinitionContinuation(c, ref)
    assert not writes


@pytest.mark.parametrize("confirmed_schema", [False, True])
def test_original_prepare_and_preflight_root_backlog(tmp_path, monkeypatch, confirmed_schema):
    from packages.multimodal_research.entry import assemble_entry, prepare_entry
    from tests.dataset_adapters.test_multimodal import gamma, gdp

    c, ref, spec, plan, failure, _, _ = root_parent(tmp_path, confirmed_schema=confirmed_schema)
    if confirmed_schema:
        import subprocess

        encoded = subprocess.check_output(
            [
                "/opt/conda/bin/python",
                "-c",
                "import io,sys; from PIL import Image; b=io.BytesIO(); "
                "Image.new('RGB',(64,64),(80,90,100)).save(b,format='JPEG'); "
                "sys.stdout.buffer.write(b.getvalue())",
            ]
        )
        monkeypatch.setattr("tests.dataset_adapters.test_multimodal.jpeg", lambda: encoded)
    root = gamma(tmp_path / "data") if confirmed_schema else gdp(tmp_path / "data")
    common = dict(
        dataset="gamma" if confirmed_schema else "harvard_gdp",
        root=root,
        sample="0001" if confirmed_schema else "data_0001.npz",
        selection="training" if confirmed_schema else None,
        task="advanced_glaucoma",
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition=spec["active_acquisition"]["path"],
        resilient_vision=True,
        view_profile="detail-v1",
        input_semantics=None if confirmed_schema else "harvard-gdp-v1",
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
    )
    prepare_entry(**common, output=tmp_path / "seed")
    real = json.loads((tmp_path / "seed/entry.json").read_bytes())
    spec.update(real)
    plan.update(input=real["input_checksum"], implementation=real["implementation"])
    failure.update(input_checksum=real["input_checksum"], protocol_checksum=digest(plan))
    for name, value in (("entry", spec), ("protocol", plan), ("stage-failure", failure)):
        (Path(ref["parent"]) / (name + ".json")).write_text(json.dumps(value))
    monkeypatch.setattr(
        "packages.multimodal_research.entry.owners", lambda: pytest.fail("live owners")
    )
    out = tmp_path / "prepared"
    result = prepare_entry(
        **common,
        output=out,
        definition_failure_of=ref["parent"],
        defer_failed_definition_expansion=True,
    )
    current = json.loads((out / "entry.json").read_bytes())
    assert result["api_requests"] == result["database_writes"] == 0
    assert current["schema_version"] == (
        "multimodal-live-entry/1.28.0" if confirmed_schema else "multimodal-live-entry/1.27.0"
    )
    assert current["definition_failure_of"]["mode"] == (
        CONFIRMED_MODE if confirmed_schema else MODE
    )
    assert current["provider"] == spec["provider"]
    assert current["active_acquisition"] == spec["active_acquisition"]
    assert assemble_entry(out, preflight_only=True)["status"] == "local_preflight_passed"
    current["definition_failure_of"]["mode"] = "technical_backlog_no_source_replay"
    (out / "entry.json").write_text(json.dumps(current))
    with pytest.raises(ValueError, match="deferral_version"):
        assemble_entry(out, preflight_only=True)


@pytest.mark.parametrize("confirmed_schema", [False, True])
def test_root_backlog_freeze_and_cold_report_keep_original_window(
    tmp_path, monkeypatch, confirmed_schema
):
    from types import SimpleNamespace

    from packages.multimodal_research import implementation
    from packages.multimodal_research.campaign import MultimodalCampaign
    from packages.multimodal_research.report_recovery import restore_execution_context

    c, ref, spec, old, _, _, _ = root_parent(tmp_path, confirmed_schema=confirmed_schema)
    continuation = ConfirmedDefinitionContinuation(c, ref)
    c = object.__new__(MultimodalCampaign)
    c.output = tmp_path / "campaign"
    c.output.mkdir()
    c.campaign_id = "root-backlog"
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.multimodal_research.campaign import configuration

    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    sha = SimpleNamespace(checksum=digest("fixture"))
    c.base = SimpleNamespace(
        manifest=sha,
        visible_value=sha,
        task=task,
        view_checksum=digest("views"),
        sources=SimpleNamespace(initial=[]),
        execution_profile=spec["execution_profile"],
        root_direction_review=spec["root_direction_review"],
        review_batch_size=3,
        active_start={k: old[k] for k in ("started_at", "deadline")},
    )
    c.config = configuration(
        directional=True,
        resilient_vision=True,
        exact_primary_status=True,
        execution_profile=spec["execution_profile"],
        review_batch_size=3,
    )
    c.targets = ()
    c.definition_continuation = continuation
    c.previous_attempts = continuation.check_attempts
    monkeypatch.setattr(implementation, "freeze", lambda: {})
    plan = c.freeze()
    assert plan["schema_version"] == (
        "multimodal-live-campaign/1.18.0" if confirmed_schema else "multimodal-live-campaign/1.17.0"
    )
    assert plan["definition_continuation"] == ref
    assert plan["previous_attempts"] == continuation.attempts
    assert plan["deadline"] == old["deadline"] and plan["started_at"] == old["started_at"]
    assert c.freeze() == plan
    current = dict(
        spec,
        schema_version=(
            "multimodal-live-entry/1.28.0" if confirmed_schema else "multimodal-live-entry/1.27.0"
        ),
        definition_failure_of=ref,
    )
    base = SimpleNamespace(task=task)
    restore_execution_context(base, current, plan)
    assert (
        base.review_batch_size == 3 and base.root_direction_review == spec["root_direction_review"]
    )
    with pytest.raises(ValueError, match="root_backlog_binding"):
        restore_execution_context(base, current, dict(plan, definition_continuation={}))


def test_confirmed_root_schema_uses_original_revision_without_source_replay(tmp_path):
    from packages.multimodal_research.definition_expansion import DefinitionExpansion
    from packages.multimodal_research.discovery_continuation import definition_successor_policy
    from tests.evaluation_preparation.test_definition_expansion import fixture

    c, ref, spec, plan, failure, records, writes = root_parent(tmp_path, confirmed_schema=True)
    before = copy.deepcopy(records)
    profile = json.loads(Path(spec["active_acquisition"]["path"]).read_bytes())
    definition_successor_policy(spec, profile, defer=True, root_confirmed=True)
    continuation = ConfirmedDefinitionContinuation(c, ref)
    seed = fixture(tmp_path)
    seed.campaign.definition_continuation = continuation
    seed.campaign.base.task.target_code = profile["definition_expansion"]["public_target"]
    expansion = DefinitionExpansion(
        seed.campaign,
        documents=seed.documents,
        sources=seed.sources,
        provider=seed.provider,
        budget=seed.budget,
        scope=seed.scope,
        policy=profile["definition_expansion"],
    )
    revision = expansion.advance()
    assert revision["rows"] == seed.campaign.base.ledger
    assert revision["rejected"][0]["reason"] == "source_schema_confirmed_not_replayed"
    assert not revision["rejected"][0]["medical_no_addition_claimed"]
    assert seed.budget.used == 0 and not revision["added"]
    assert expansion.advance() == revision
    assert continuation.check_attempts()[0]["requests"] == 3
    assert continuation.check_attempts()[0]["kind"] == "confirmed_source_schema_deferred"
    assert (
        ConfirmedDefinitionContinuation(c, ref, read_only=True).check_attempts()
        == continuation.attempts
    )
    assert len(writes) == 1 and all(records[k] == v for k, v in before.items())


@pytest.mark.parametrize("fault", ["unknown", "response", "increment", "wire_unknown"])
def test_confirmed_root_schema_never_discards_unknown_or_adopted_work(tmp_path, fault):
    c, ref, spec, plan, failure, records, writes = root_parent(tmp_path, confirmed_schema=True)
    expected = "definition_successor_original_owner"
    if fault == "unknown":
        records["journal:3"][0]["record"]["error"] = "timeout"
    elif fault == "response":
        failed = records["journal:3"][0]["record"]
        records["definition:response:" + failed["request_id"]][0]["raw_text"] = "{}"
        expected = "definition_successor_response_binding"
    elif fault == "increment":
        records["definition:gap:0:decision"][0]["candidate"]["merged"] = ["existing"]
        expected = "definition_successor_uncommitted_increment"
    else:
        records.pop("definition:wire:0:outcome")
        expected = "definition_successor_wire_unknown"
    with pytest.raises(ValueError, match="^" + expected + "$"):
        ConfirmedDefinitionContinuation(c, ref)
    assert not writes


def test_confirmed_root_first_source_schema_without_new_public_wire(tmp_path):
    c, ref, spec, plan, failure, records, writes = root_parent(tmp_path, confirmed_schema=True)
    records["journal:0"] = records.pop("journal:2")
    records["journal:1"] = records.pop("journal:3")
    records.pop("definition:response:r0")
    records.pop("definition:gap:0:decision")
    records.pop("definition:wire:0:outcome")
    failure["new_requests"] = 1
    failure["transport"]["dispatched_observed"] = 1
    failure["operations"] = [dict(kind="local_pool_reconciliation"), dict(kind="llm")]
    (Path(ref["parent"]) / "stage-failure.json").write_text(json.dumps(failure))
    ref["failure"] = digest(failure)
    before = copy.deepcopy(records)
    continuation = ConfirmedDefinitionContinuation(c, ref)
    assert continuation.check_attempts()[0]["requests"] == 1
    assert continuation.wire_proofs == []
    assert len(writes) == 1 and all(records[k] == v for k, v in before.items())
    c.output = tmp_path / "different-successor"
    c.output.mkdir()
    with pytest.raises(ValueError, match="^definition_successor_already_claimed$"):
        ConfirmedDefinitionContinuation(c, ref)
    assert len(writes) == 1
