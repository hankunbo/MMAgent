"""Synthetic stage integration; original definition/provider/budget, memory Runtime."""

import copy
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_definition_stage as module
from packages.evaluation_preparation.dynamic_campaign import Campaign
from packages.evaluation_preparation.generic_review import GenericReview
from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
from packages.grounding.models import digest
from packages.llm_provider import definition_passages as passages
from packages.llm_provider.provider import DeterministicFakeProvider, ProviderError
from packages.research_runtime.agent_records import RuntimeAgentJournal
from tests.evaluation_preparation.test_generic_definition_input import setup as input_setup


def setup(monkeypatch, tmp_path):
    f, wire, state, view, old, policy = input_setup(monkeypatch, tmp_path)
    material = f.sources.read(f.sources.value["code"])

    class PublicSources(KnowledgeDefinitionSources):
        def __init__(self):
            super().__init__(
                repository=None,
                store=None,
                index_keys=policy["index_keys"],
                failure_policy=policy.get("search_failure_policy"),
                material_policy=policy.get("material_policy"),
                passage_plans=policy.get("passage_plans"),
            )

        def read(self, code):
            return f.sources.read(code)

        def read_many(self, codes):
            return [self.read(code) for code in codes]

        def restore_indexes(self, value):
            pass

        def search(self, **kwargs):
            return dict(
                schema_version="research-definition-search/1.1.0",
                materials=[material],
                acquired_indexes=[],
                backlog=[],
            )

        def reconcile_pool(self, **kwargs):
            return dict(
                schema_version="research-definition-pool/1.0.0",
                materials=[material],
                backlog=[],
                retrieval={"test_only": True},
            )

    class Successor(GenericReview):
        # The production campaign's state/accounting machinery is used below;
        # only input registration and persistence are memory test fixtures.
        def __init__(self):
            self.acquisition, self.base = f.owner.campaign, f.owner.campaign.base
            self.spec, self.review_key = dict(key="synthetic-original"), "synthetic-successor"
            self.output = tmp_path
            self.config = old.provider.config
            self.cancelled = False
            self.protocol = dict(
                previous=dict(reservations=10),
                observations=dict(definition_stage=module.binding(view, policy)),
                max_new_requests=5,
                deadline=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            )
            self.sessions = SimpleNamespace(read=lambda _: None)
            self.contexts = SimpleNamespace(read_research_run=lambda _: context)
            self.registered = 0

        def paths(self):
            return [tmp_path / "definition", tmp_path / "review"]

        def phase_path(self, target):
            assert target is None
            return tmp_path / "review"

        def prior(self):
            return dict(reservations=10)

        def guard(self, *, dispatch=False):
            if self.cancelled:
                raise ProviderError("cancelled")
            return Campaign.guard(self, dispatch=dispatch)

        def accepted_observations(self):
            return {}

        def read_model(self):
            return None

        def documents(self, run_id):
            assert run_id == state.material.scope.run_id
            return old.documents

        def register_phase(self, evidence, key):
            self.registered += 1
            return context, state

    request = SimpleNamespace(
        model_dump=lambda **_: dict(lock=dict(run_id=state.material.scope.run_id))
    )
    context = SimpleNamespace(run_id=state.material.scope.run_id, request=request)
    # Use a serializable request value for the original digest protocol.
    from pydantic import BaseModel

    class Request(BaseModel):
        lock: dict

    context.request = Request(lock=dict(run_id=state.material.scope.run_id))
    old.documents.run_id = state.material.scope.run_id
    old.documents.sessions = SimpleNamespace(
        origin=lambda _: (datetime.now(UTC) - timedelta(hours=1), None)
    )
    monkeypatch.setattr(
        module,
        "GenericPhase",
        lambda *a, **kw: SimpleNamespace(project=lambda **kw: (state, None), verify=lambda _: None),
    )
    parent = Successor()
    stage = module.GenericDefinitionStage(parent, view, policy, sources=PublicSources())
    draft = dict(
        f.candidate,
        schema_version="source-definition-draft/7.2.0",
        excerpts=[
            dict(
                passage_id=next(
                    iter(
                        passages.choices(
                            dict(target=view.task.target_code, source_materials=[material])
                        )
                    )
                )
            )
        ],
    )
    response = dict(
        schema_version="research-definition-proposal/6.0.0",
        definitions=[draft],
        no_addition_reason=None,
    )

    def factory(config):
        return DeterministicFakeProvider(config, [response])

    return SimpleNamespace(
        stage=stage,
        parent=parent,
        view=view,
        facts=f,
        policy=policy,
        state=state,
        documents=old.documents,
        factory=factory,
        wire=wire,
    )


def test_source_is_in_original_campaign_budget_and_definition_reaches_review(monkeypatch, tmp_path):
    from packages.evaluation_preparation.generic_definition_projection import (
        project_interpretations,
    )
    from tests.agent_provider.batch_support import generate

    f = setup(monkeypatch, tmp_path)
    before = digest(f.state), digest(f.view.observations)
    revision = f.stage.advance(provider_factory=f.factory)
    assert revision["added"] == []
    assert [r["id"] for r in revision["replaced"]] == [f.facts.target]
    assert f.stage.budget.used == f.parent.usage() == 1
    assert f.stage.previous_attempts() == [dict(requests=10)]
    keys = f.stage.review_keys()
    assert len(keys) == 1
    state, manifest = project_interpretations(f.state, f.stage.expansion.owner, keys)
    f.wire.owner.state = state
    submission = generate(f.wire)
    f.wire.acceptance.submit(submission, key="stage-source-review")
    assert f.wire.acceptance.accept(key="stage-source-review").status == "accepted"
    assert manifest["whole_case_aggregation_required"]
    assert before == (digest(f.state), digest(f.view.observations))
    assert f.stage.advance() == revision
    assert f.stage.budget.used == f.parent.usage() == 1


def test_cold_restore_does_not_register_or_construct_provider(monkeypatch, tmp_path):
    f = setup(monkeypatch, tmp_path)
    first = f.stage.advance(provider_factory=f.factory)
    prior = copy.deepcopy(f.documents.saved)
    # A cold input view starts at the immutable historical definition.
    from packages.evaluation_preparation.generic_definition_input import GenericDefinitionInputView

    new_view = GenericDefinitionInputView(f.facts.owner)
    restored = module.GenericDefinitionStage(f.parent, new_view, f.policy, sources=f.stage.sources)
    monkeypatch.setattr(module, "StructuredProvider", lambda *a, **kw: pytest.fail("no provider"))
    assert restored.restore() == first
    assert restored.review_keys() == f.stage.review_keys()
    assert f.parent.registered == 1
    assert f.documents.saved == prior
    assert f.parent.usage() == 1
    with pytest.raises(ValueError, match="restore_is_read_only"):
        restored.advance()


def test_restore_without_registration_cannot_create_a_run(monkeypatch, tmp_path):
    f = setup(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="no_saved_stage"):
        f.stage.restore()
    assert f.parent.registered == 0 and f.parent.usage() == 0


@pytest.mark.parametrize(
    "fault", ["paths_missing", "paths_duplicate", "protocol", "prior", "cancel"]
)
def test_scope_or_cancel_refuses_before_source_dispatch(monkeypatch, tmp_path, fault):
    f = setup(monkeypatch, tmp_path)
    if fault == "paths_missing":
        f.parent.paths = lambda: [tmp_path / "review"]
    elif fault == "paths_duplicate":
        f.parent.paths = lambda: [tmp_path / "definition"] * 2
    elif fault == "protocol":
        f.parent.protocol["max_new_requests"] += 1
    elif fault == "prior":
        f.parent.prior = lambda: dict(reservations=0)
    else:
        f.parent.cancelled = True
    with pytest.raises(
        (ValueError, ProviderError), match="frozen_accounting_binding|cancelled|scope"
    ):
        f.stage.advance(provider_factory=f.factory)
    assert f.parent.registered == 0
    assert not (tmp_path / "definition/request.json").exists()


def test_reservation_unknown_survives_cold_budget_without_reissue(monkeypatch, tmp_path):
    f = setup(monkeypatch, tmp_path)
    expansion = f.stage.initialize(provider_factory=f.factory)
    f.stage.budget.reserve(
        expansion.scope, f.stage.budget.deadline, digest(expansion.provider.config)
    )
    assert f.parent.usage() == 1
    assert f.stage.budget.summary().pending_unknown == 1
    journal = RuntimeAgentJournal(f.documents)
    assert sum(v.get("phase") == "reserved" for v in journal.read()) == 1
    # Source's own operation journal owns resend decisions. The stage preserves
    # the original request budget, never interprets UNKNOWN as unsent.
    new = module.GenericDefinitionStage(f.parent, f.view, f.policy, sources=f.stage.sources)
    new.initialize(restore_only=True)
    assert new.budget.used == 1 and new.budget.summary().pending_unknown == 1
    assert f.parent.registered == 1


def test_review_reserve_and_wire_cancellation_still_apply(monkeypatch, tmp_path):
    f = setup(monkeypatch, tmp_path)
    e = f.stage.initialize(provider_factory=f.factory)
    # Existing RequestBudget reservation followed by the original campaign's
    # before-wire guard; no transport or model is invoked by this test.
    f.stage.budget.reserve(e.scope, f.stage.budget.deadline, digest(e.provider.config))
    f.parent.cancelled = True
    with pytest.raises(ProviderError, match="cancelled"):
        f.stage.budget.before_wire(None, None, None, None)
    f.parent.cancelled = False
    while f.parent.usage() < 2:
        f.stage.budget.reserve(e.scope, f.stage.budget.deadline, digest(e.provider.config))
    with pytest.raises(ProviderError, match="definition_review_reserve"):
        f.stage.budget.reserve(e.scope, f.stage.budget.deadline, digest(e.provider.config))
    assert f.parent.usage() == 2


def test_tampered_saved_stage_cannot_restore(monkeypatch, tmp_path):
    f = setup(monkeypatch, tmp_path)
    f.stage.initialize(provider_factory=f.factory)
    request_path = tmp_path / "definition/request.json"
    raw = json.loads(request_path.read_bytes())
    raw["lock"]["run_id"] = "RUN:OTHER"
    request_path.write_text(json.dumps(raw))
    new = module.GenericDefinitionStage(f.parent, f.view, f.policy, sources=f.stage.sources)
    with pytest.raises(ValueError, match="original_request_changed"):
        new.restore()
    assert f.parent.registered == 1


def test_original_review_journal_consumes_shared_source_allowance(monkeypatch, tmp_path):
    from packages.llm_provider.provider import RequestBudget
    from tests.agent_provider.support import Documents

    f = setup(monkeypatch, tmp_path)
    expansion = f.stage.initialize(provider_factory=f.factory)
    review = Documents()
    review.run_id = "RUN:ORIGINAL_REVIEW"
    review.sessions = f.documents.sessions
    review_path = tmp_path / "review"
    review_path.mkdir()
    (review_path / "request.json").write_text(json.dumps(dict(lock=dict(run_id=review.run_id))))
    f.parent.documents = lambda run: review if run == review.run_id else f.documents
    budget = RequestBudget(
        run_id=review.run_id,
        deadline=f.stage.budget.deadline,
        max_requests=f.parent.config.max_requests,
        journal=RuntimeAgentJournal(review),
    )
    scope = expansion.scope.model_copy(update=dict(run_id=review.run_id))
    budget.reserve(scope, budget.deadline, digest(f.parent.config))
    assert f.parent.usage() == 1 and f.stage.budget.used == 0
    f.stage.budget.reserve(expansion.scope, budget.deadline, digest(expansion.provider.config))
    assert f.parent.usage() == 2 and f.stage.budget.used == 1
    with pytest.raises(ProviderError, match="definition_review_reserve"):
        f.stage.budget.reserve(expansion.scope, budget.deadline, digest(expansion.provider.config))
    assert f.parent.usage() == 2


@pytest.mark.parametrize("fault", ["indexes", "passage", "public_dispatch"])
def test_source_owner_configuration_is_frozen(monkeypatch, tmp_path, fault):
    f = setup(monkeypatch, tmp_path)
    if fault == "indexes":
        f.stage.sources.index_keys = (digest("unselected-index"),)
    elif fault == "passage":
        f.stage.sources.passage_plans = {}
    else:
        f.stage.sources.search_public = lambda **_: pytest.fail("must not dispatch")
    with pytest.raises(ValueError, match="frozen_accounting_binding"):
        f.stage.advance(provider_factory=f.factory)
    assert f.parent.registered == 0


def test_review_time_reserve_refuses_source_without_extending_deadline(monkeypatch, tmp_path):
    f = setup(monkeypatch, tmp_path)
    deadline = (datetime.now(UTC) + timedelta(minutes=10)).isoformat()
    f.parent.protocol["deadline"] = deadline
    stage = module.GenericDefinitionStage(f.parent, f.view, f.policy, sources=f.stage.sources)
    expansion = stage.initialize(provider_factory=f.factory)
    with pytest.raises(ProviderError, match="definition_review_time_reserve"):
        stage.budget.reserve(
            expansion.scope, stage.budget.deadline, digest(expansion.provider.config)
        )
    assert stage.budget.used == 0 and f.parent.protocol["deadline"] == deadline


def test_accounting_does_not_rebuild_fact_owner_and_each_guard_checks_once(monkeypatch, tmp_path):
    f = setup(monkeypatch, tmp_path)
    read = f.facts.owner._read
    reads = []

    def counted():
        reads.append("current-facts")
        return read()

    f.facts.owner._read = counted
    assert f.stage.usage() == 0
    assert f.stage.previous_attempts() == [dict(requests=10)]
    assert reads == []
    f.stage.guard_definition_materials()
    assert reads == ["current-facts"]
    f.stage.guard_definition_materials(dispatch=True)
    assert reads == ["current-facts", "current-facts"]
    f.facts.current["facts"][f.facts.target]["origin"]["tool_result_checksum"] = digest("changed")
    with pytest.raises(ValueError, match="generic_definition_original_changed"):
        f.stage.guard_definition_materials(dispatch=True)


@pytest.mark.parametrize("terminal", ["deadline", "cancelled"])
def test_report_only_cold_restore_retains_current_sources_without_dispatch(
    monkeypatch, tmp_path, terminal
):
    from packages.evaluation_preparation.generic_definition_input import GenericDefinitionInputView
    from packages.multimodal_research import definition_expansion

    f = setup(monkeypatch, tmp_path)
    revision = f.stage.advance(provider_factory=f.factory)
    saved = copy.deepcopy(f.documents.saved)
    if terminal == "deadline":

        class ExpiredClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.fromisoformat(f.parent.protocol["deadline"]) + timedelta(seconds=1)

        monkeypatch.setattr(definition_expansion, "datetime", ExpiredClock)
    else:
        f.parent.sessions.read = lambda _: SimpleNamespace(cancelled=True, phase="stopped")
    fresh = module.GenericDefinitionStage(
        f.parent, GenericDefinitionInputView(f.facts.owner), f.policy, sources=f.stage.sources
    )
    monkeypatch.setattr(module, "StructuredProvider", lambda *a, **kw: pytest.fail("no provider"))
    with pytest.raises(ProviderError, match=terminal):
        fresh.restore()
    assert fresh.restore(report_only=True) == revision
    assert fresh.review_keys() == f.stage.review_keys()
    assert f.documents.saved == saved
    assert f.parent.registered == 1 and f.parent.usage() == 1
    with pytest.raises(ValueError, match="restore_is_read_only"):
        fresh.advance()
    f.stage.sources.read = lambda _: None
    with pytest.raises(ValueError, match="source_changed"):
        fresh.restore(report_only=True)
