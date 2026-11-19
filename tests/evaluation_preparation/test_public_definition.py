"""Injected public transport; original connector/stage/budget/definition owner.

These are software checks, not evidence of fresh retrieval or medical validity.
The existing current material is deliberately retained when search is empty.
"""

import copy
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_definition_public as public
from packages.evaluation_preparation import generic_definition_stage as module
from packages.evaluation_preparation.generic_definition_successor import DefinitionReview
from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
from packages.grounding.models import digest
from packages.llm_provider.provider import ProviderError
from packages.multimodal_research.definition_expansion import read_expansion_policy
from tests.evaluation_preparation.test_generic_definition_stage import setup as original_setup


def policy_of(old):
    value = copy.deepcopy(old)
    value.update(
        schema_version="campaign-definition-expansion/1.43.0",
        retrieval_scope="generic-bounded-public/1.0.0",
        unresolved_relation_retrieval="unseen-first-for-owner-gap/1.0.0",
        max_candidates=5,
        max_search_requests=1,
        max_fetch_requests=2,
    )
    for plan in value["passage_plans"].values():
        plan.update(schema_version="definition-passage-plan/5.0.0", max_per_document=3)
    return read_expansion_policy(value)


def setup(monkeypatch, tmp_path):
    f = original_setup(monkeypatch, tmp_path)
    goal = dict(
        schema_version="generic-original-public-proposition/1.0.0",
        title="glaucoma",
        original_root_key="glaucoma",
        target=f.view.task.target_code,
        synthetic_fixture_only=True,
    )
    f.view.public_goal = goal
    f.view.public_goal_checksum = digest(goal)
    f.policy["public_proposition_checksum"] = digest(goal)
    monkeypatch.setattr(
        "packages.evaluation_preparation.generic_definition_input.public_proposition",
        lambda *a: goal,
    )
    selected = policy_of(f.policy)
    policy = selected.model_dump(mode="json")
    f.parent.__class__.__bases__ = (DefinitionReview,)
    # Original test owner overrides registration and current case input only.
    # The public budget is the real DefinitionReview.usage implementation.
    f.parent.protocol["max_new_requests"] = 10
    f.parent.protocol["observations"]["definition_stage"] = module.binding(f.view, policy)
    f.parent.checked = SimpleNamespace(
        policy=selected,
        binding=f.parent.protocol["observations"]["definition_stage"],
        view=f.view,
    )
    sources = KnowledgeDefinitionSources(
        repository=None,
        store=None,
        index_keys=selected.index_keys,
        failure_policy=selected.search_failure_policy,
        material_policy=selected.material_policy,
        passage_plans=policy["passage_plans"],
    )
    material = f.facts.sources.read(f.facts.sources.value["code"])
    monkeypatch.setattr(sources, "_context_query", lambda *a: [material])
    monkeypatch.setattr(sources, "read", f.facts.sources.read)
    monkeypatch.setattr(sources, "read_many", lambda codes: [sources.read(c) for c in codes])
    monkeypatch.setattr(
        sources,
        "reconcile_pool",
        lambda **kw: dict(
            schema_version="research-definition-pool/1.0.0",
            materials=[material],
            backlog=[],
            retrieval={"test_only": True},
        ),
    )
    # The rank call remains original port-shaped; no fabricated new source.
    sources.repository = SimpleNamespace(
        rank_definition_discovery_hits=lambda **kw: dict(
            schema_version="definition-fetch-order/1.0.0",
            hits=[],
            index_keys=list(sources.index_keys),
            eligibility_granted=False,
            order=[],
            decisions=[],
        )
    )
    stage = module.GenericDefinitionStage(f.parent, f.view, policy, sources=sources)
    calls = []

    def search(repository, providers, topic, **kwargs):
        calls.append((repository, providers, topic, kwargs))
        governor = kwargs["request_guard"]
        governor.before_send("PROVIDER:SYNTHETIC")
        governor.completed_wire(dict(status="empty", synthetic=True))
        return dict(status="empty", results=[], runs=[])

    monkeypatch.setattr("apps.knowledge_literature_tools.search", search)
    f.stage, f.policy, f.calls = stage, policy, calls
    return f


def test_public_connector_and_source_share_original_budget(monkeypatch, tmp_path):
    f = setup(monkeypatch, tmp_path)
    revision = f.stage.advance(provider_factory=f.factory)
    assert len(f.calls) == 1
    assert "public_target" not in f.calls[0][3]
    assert f.calls[0][2] == "glaucoma"
    assert f.stage.budget.used == 1
    assert f.parent.usage() == 2  # one public search plus one Source, not two model calls
    assert public.accounting(f.parent)["public_wire_reservations"] == 1
    assert len(f.stage.review_keys()) == 1
    assert f.stage.advance() == revision
    assert len(f.calls) == 1 and f.parent.usage() == 2
    assert not revision["added"]  # existing observation interpretation, not fabricated novelty


def test_cold_read_has_no_connector_or_new_requests(monkeypatch, tmp_path):
    from packages.evaluation_preparation.generic_definition_input import GenericDefinitionInputView

    f = setup(monkeypatch, tmp_path)
    f.stage.advance(provider_factory=f.factory)
    saved = copy.deepcopy(f.documents.saved)
    # Clear only this test's in-memory connector; persisted records remain intact.
    sources = f.stage.sources
    sources.search_public = sources.fetch_hit = None
    restored = module.GenericDefinitionStage(
        f.parent, GenericDefinitionInputView(f.facts.owner), f.policy, sources=sources
    )
    monkeypatch.setattr(module, "StructuredProvider", lambda *a, **kw: pytest.fail("no provider"))
    restored.restore()
    assert restored.public_connector is None
    assert f.parent.usage() == 2 and len(f.calls) == 1
    assert f.documents.saved == saved
    with pytest.raises(ValueError, match="restore_is_read_only"):
        restored.advance()


def test_wire_unknown_is_charged_and_not_resent(monkeypatch, tmp_path):
    f = setup(monkeypatch, tmp_path)
    attempts = []

    def lost(*args, **kwargs):
        attempts.append(1)
        kwargs["request_guard"].before_send("PROVIDER:SYNTHETIC")
        raise TimeoutError("synthetic lost reply after dispatch")

    monkeypatch.setattr("apps.knowledge_literature_tools.search", lost)
    with pytest.raises(TimeoutError, match="synthetic lost reply"):
        f.stage.advance(provider_factory=f.factory)
    assert f.parent.usage() == 1 and f.stage.budget.used == 0
    with pytest.raises(ProviderError, match="unknown_no_repeat"):
        f.stage.advance()
    assert attempts == [1] and f.parent.usage() == 1


def test_cancelled_or_review_reserve_does_not_search(monkeypatch, tmp_path):
    f = setup(monkeypatch, tmp_path)
    f.parent.cancelled = True
    with pytest.raises(ProviderError, match="cancelled"):
        f.stage.advance(provider_factory=f.factory)
    assert not f.calls
    f.parent.cancelled = False
    f.parent.protocol["max_new_requests"] = f.stage.policy.reserve_review_requests
    # A deliberately changed protocol must be rejected before dispatch.
    with pytest.raises(ValueError, match="frozen_accounting_binding"):
        f.stage.advance(provider_factory=f.factory)
    assert not f.calls


@pytest.mark.parametrize("fault", ["index", "receipt", "callback", "topics"])
def test_caller_cannot_extend_indexes_or_replace_connector(monkeypatch, tmp_path, fault):
    f = setup(monkeypatch, tmp_path)
    f.stage.initialize(provider_factory=f.factory)
    if fault == "index":
        f.stage.sources.index_keys += (digest("caller-index"),)
    elif fault == "receipt":
        f.stage.sources.index_receipts[digest("caller-index")] = {"key": digest("caller-index")}
    elif fault == "callback":
        f.stage.sources.search_public = lambda **kw: []
    else:
        f.stage.public_connector.topics = {"caller-question": ["glaucoma"]}
    with pytest.raises(ValueError, match="public_(index|connector)"):
        f.stage._check_binding()
    assert not f.calls


@pytest.mark.parametrize("fault", ["binding", "key", "kind", "gap"])
def test_operation_binding_is_not_a_caller_count(monkeypatch, tmp_path, fault):
    f = setup(monkeypatch, tmp_path)
    e = f.stage.initialize(provider_factory=f.factory)
    e.reserve_wire("PROVIDER:SYNTHETIC", kind="search")
    assert f.parent.usage() == 1
    key = "definition:operation:0"
    row = copy.deepcopy(f.documents.saved[key][0])
    if fault == "binding":
        row["binding"] = {}
    elif fault == "key":
        row["key"] = "foreign-wire"
    elif fault == "kind":
        row["wire_kind"] = "caller-model"
    else:
        del f.documents.saved[key]
        key = "definition:operation:1"
    f.documents.saved[key] = (row,)
    with pytest.raises(ValueError, match="public_(operation|wire)"):
        f.parent.usage()


def test_new_version_cannot_silently_keep_local_only_passages(monkeypatch, tmp_path):
    f = original_setup(monkeypatch, tmp_path)
    value = policy_of(f.policy).model_dump(mode="json")
    next(iter(value["passage_plans"].values()))["schema_version"] = "definition-passage-plan/7.0.0"
    with pytest.raises(ValueError):
        read_expansion_policy(value)


def test_public_root_does_not_guess_a_staging_goal(monkeypatch, tmp_path):
    from apps.knowledge_literature_tools import PROVIDERS
    from packages.knowledge_search.broad import broad_plan
    from packages.knowledge_search.targeted import targeted_plan

    f = setup(monkeypatch, tmp_path)
    f.stage.initialize(provider_factory=f.factory)
    assert f.stage.public_connector.search_target() is None
    with pytest.raises(ValueError, match="unsupported_public_proposition"):
        targeted_plan((PROVIDERS["europepmc"],), target=f.view.task.target_code)
    # The chosen public query is the original existing general-glaucoma builder.
    plan = broad_plan((PROVIDERS["europepmc"],), topics=("cup_disc_ratio",), limit=5)
    assert plan.query_plan_version == "5.0.0"
    assert f.view.task.target_code not in plan.model_dump_json()
    changed = dict(f.view.public_goal, title="other disease", original_root_key="other disease")
    f.view.public_goal = changed
    with pytest.raises(ValueError, match="public_query_root_scope"):
        public.public_root(f.view, f.stage.policy)
