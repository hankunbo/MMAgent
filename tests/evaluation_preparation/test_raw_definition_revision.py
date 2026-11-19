"""Same raw-input examination: new relation, original fact, no duplicate vote."""

import copy
import json
from pathlib import Path

import pytest

from packages.dataset_adapters.multimodal.models import Component
from packages.evidence_reuse.definition_acceptance import requirement_rows
from packages.llm_provider import DeterministicFakeProvider
from packages.multimodal_research.definition_expansion import (
    DefinitionExpansion,
    definition_config,
    read_expansion_policy,
)
from packages.multimodal_research.material import observation
from tests.evaluation_preparation.test_definition_expansion import fixture
from tests.evaluation_preparation.test_owner_dependency_definition import raw


@pytest.mark.parametrize("number", [0.0, -1.5])
@pytest.mark.parametrize("explicit_method", [True, False])
@pytest.mark.parametrize("new_discovery", [False, True])
def test_numeric_revision_original_provider_owner_and_fact(
    tmp_path, number, explicit_method, new_discovery
):
    old_owner, manifest, sources, _, input_base, value = raw()
    manifest.components += (
        Component(
            id="metric",
            kind="scalar",
            concept="metric",
            modality="measurement",
            status="obtained",
            value=number,
            unit="unit-x",
            unit_basis="synthetic public schema",
            source_locator="synthetic",
        ),
    )
    value.update(
        schema_version="source-definition-draft/6.1.0",
        target="GOAL:ADVANCED_GLAUCOMA",
        examination="Synthetic numeric examination",
        modalities=["measurement"],
        input_concepts=["metric"],
        acquisition="source_measurement",
        method="source_numeric_read",
        unit="unit-x",
        input_aliases=["input:1"],
    )
    value["dependency_contracts"][0].update(
        input_alias="input:1",
        predicate="source_numeric_available",
        modality="measurement",
        concept="metric",
        method="source_numeric_read",
        unit="unit-x",
    )
    for excerpt in value["excerpts"]:
        excerpt.pop("start", None)
        excerpt.pop("end", None)
    old_owner.binding["target"] = value["target"]
    previous = old_owner.accept("old", value)
    row = requirement_rows(previous, manifest, fact_reader=old_owner.fact_reader)[0]
    row.update(id="check:metric", parent="EXISTING")
    row.pop("definition_receipt_checksum")
    if not explicit_method:
        # Production adapter rows predate the definition-draft expected_* fields;
        # their actual observation still carries the owner's exact method/unit.
        row.pop("expected_method")
        row.pop("expected_unit")
    original = copy.deepcopy(row)
    value["interpretation"] = "New source-grounded synthetic relation, not a clinical rule."
    seed = fixture(tmp_path)
    base = seed.campaign.base
    base.manifest, base.ledger = manifest, [row]
    base.task.target_code = value["target"]
    base.read_dependency_input = lambda: manifest
    base.read_dependency_fact = old_owner.fact_reader
    base.prepared = input_base.prepared
    base.observations = {
        row["id"]: observation(row["id"], number, method="source_numeric_read", unit="unit-x")
    }
    before = copy.deepcopy(base.observations)
    policy = json.loads(
        Path("configs/research/multimodal_definition_raw_revision_v1.json").read_text()
    )
    if new_discovery:
        policy.update(
            schema_version="campaign-definition-expansion/1.30.0",
            prompt_version="definition-draft-6.4.0",
            unresolved_relation_retrieval="unseen-first-for-owner-gap/1.0.0",
        )
    policy["query_requirements"] = {q: [row["id"]] for q in policy["public_queries"]}
    sources.index_keys = ("synthetic-index",)
    sources.restore_indexes = lambda _: None
    sources.search = lambda **kw: dict(
        schema_version="research-definition-search/1.1.0",
        materials=[sources.read(sources.value["code"])],
        acquired_indexes=[],
        backlog=[],
    )
    sources.reconcile_pool = lambda **kw: dict(
        schema_version="research-definition-pool/1.0.0",
        materials=[sources.read(sources.value["code"])],
        backlog=[],
        retrieval={"synthetic": True},
    )

    def response(body):
        prompt = body["messages"][0]["content"]
        assert "may also REVISE" in prompt and "does NOT forbid" in prompt
        payload = json.loads(body["messages"][1]["content"][0]["text"])
        assert payload["confirmed_observation_capabilities"] == []
        assert (
            payload["existing_requirement_context"]["requirements"][0]["method"]
            == "source_numeric_read"
        )
        return dict(
            schema_version="research-definition-proposal/5.0.0",
            definitions=[value],
            no_addition_reason=None,
        )

    provider = DeterministicFakeProvider(
        definition_config(
            seed.provider.config, prompt_version=policy["prompt_version"], source_output_tokens=8192
        ),
        [response],
    )
    expansion = DefinitionExpansion(
        seed.campaign,
        documents=seed.documents,
        sources=sources,
        provider=provider,
        budget=seed.budget,
        scope=seed.scope,
        policy=policy,
    )
    revision = expansion.advance()
    assert len(revision["replaced"]) == 1 and not revision["added"]
    assert revision["replaced"][0]["reason"] == "target_relation_rebound"
    assert revision["denominator_before"] == revision["denominator_after"] == 1
    assert base.observations == before and seed.campaign.targets == ()
    assert row == original and base.ledger[0]["id"] == row["id"]
    assert base.ledger[0]["definition"].interpretation == value["interpretation"]
    assert expansion.advance() == revision and seed.budget.used == 1
    assert old_owner.read("old", historical=True) == previous
    base.observations[row["id"]] = observation(
        row["id"], number, method="wrong-method", unit="unit-x"
    )
    with pytest.raises(ValueError):
        expansion.read_revision()
    base.observations = before
    assert expansion.read_revision() == revision


def test_new_opt_in_policy_old_contract_unchanged():
    from packages.multimodal_research.implementation import freeze
    from packages.object_store import checksum

    path = Path("configs/research/multimodal_definition_raw_revision_v1.json")
    assert freeze()[str(path)] == checksum(path.read_bytes())
    new = json.loads(
        Path("configs/research/multimodal_definition_raw_revision_v1.json").read_text()
    )
    assert read_expansion_policy(new).prompt_version == "definition-draft-6.2.0"
    old = dict(
        new,
        schema_version="campaign-definition-expansion/1.25.0",
        prompt_version="definition-draft-6.1.0",
        unresolved_relation_retrieval="unseen-first-for-owner-gap/1.0.0",
    )
    assert read_expansion_policy(old).prompt_version == "definition-draft-6.1.0"
    with pytest.raises(ValueError):
        read_expansion_policy(dict(new, schema_version="campaign-definition-expansion/1.25.0"))


@pytest.mark.parametrize("fault", [None, "unknown", "error", "incomplete", "old", "increment"])
def test_completed_zero_increment_parent_uses_original_terminal_gate(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace

    from packages.multimodal_research import campaign_renewal as renewal

    plan = dict(
        schema_version="multimodal-campaign-renewal/1.11.0",
        mode="definitions",
        definition_increment_only=True,
        input="input",
        task="task",
        previous_attempts=[],
    )
    if fault == "old":
        plan["schema_version"] = "multimodal-campaign-renewal/1.1.0"
    phase = dict(run_id="RUN:END", target=None)
    result = dict(
        complete=fault != "incomplete",
        case_code="CASE:SYNTHETIC",
        task={},
        phases=[phase],
        new_requests=1,
        previous_attempts=[],
    )
    end = {
        "multimodal-case-result": (result,),
        "dynamic-phase-result": (phase,),
        "dynamic-campaign": (plan,),
    }
    source = {
        "definition:revision": ({"added": [], "accepted": {}},),
        "journal:0": ({"phase": "reserved", "request_id": "request:source"},),
    }
    if fault != "unknown":
        source["journal:1"] = (
            {
                "phase": "completed",
                "record": {
                    "request_id": "request:source",
                    "role": "source",
                    "error": "transport" if fault == "error" else None,
                },
            },
        )
    if fault == "increment":
        source["definition:revision"] = ({"added": ["new-check"], "accepted": {}},)
    parent = SimpleNamespace(
        base=SimpleNamespace(
            binding={"case_code": "CASE:SYNTHETIC"},
            manifest=SimpleNamespace(checksum="input"),
            task=SimpleNamespace(checksum="task", model_dump=lambda **kw: {}),
        ),
        sessions=SimpleNamespace(read=lambda _: SimpleNamespace(phase="stopped", pending=None)),
        documents=lambda rid: SimpleNamespace(
            run_id=rid, read=(end if rid == "RUN:END" else source).get
        ),
    )
    (tmp_path / "protocol.json").write_text(json.dumps(plan))
    (tmp_path / "result.json").write_text(json.dumps(result))
    monkeypatch.setattr(renewal, "assemble", lambda *a, **kw: parent)
    if fault is None:
        assert renewal.completed_parent(tmp_path) == (parent, result)
    else:
        error = (
            "renewal_definition_source_unconfirmed"
            if fault in {"unknown", "error"}
            else "renewal_definition_continuation_requires_unstarted_proposal"
            if fault == "increment"
            else "renewal_definition_source_already_started"
        )
        with pytest.raises(ValueError, match=error):
            renewal.completed_parent(tmp_path)
