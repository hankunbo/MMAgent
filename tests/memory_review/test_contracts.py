"""Bounded pure 10D probes; original Provider and original Planner remain executable."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest, validate
from packages.llm_provider import DeterministicFakeProvider, MemoryJournal, RequestBudget
from packages.llm_provider.config import load_provider_config
from packages.llm_provider.transport import Reply
from packages.memory_review.models import Advice, Policy, Query, ReviewInput
from packages.retrieval.memory.service import mismatch
from tests.agent_provider.support import materials
from tests.memory_review.support import binding, conditions, policy


def test_real_policy_cannot_be_approved_and_no_implicit_threshold():
    p = policy()
    raw = p.model_dump(mode="json")
    raw.update(
        approved=False,
        min_cases=None,
        min_reviewers=None,
        reviewers=[],
        purpose="research_offline",
        test_context=None,
    )
    default = validate(Policy, raw)
    assert not default.approved and default.min_cases is None
    for field, value in [
        ("purpose", "research_offline"),
        ("min_cases", None),
        ("min_cases", 1),
        ("min_reviewers", None),
        ("reviewers", ()),
        ("test_context", None),
    ]:
        with pytest.raises(ValueError):
            validate(Policy, p.model_copy(update={field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("goal", "GOAL:OTHER"),
        ("template", "TEMPLATE:OTHER"),
        ("graph_conditions", ()),
        ("modalities", ("OCT",)),
        ("inputs", ()),
        ("patient_conditions", {"laterality": "unknown"}),
        ("patient_conditions", {}),
        ("rule_versions", {"synthetic_rule": "2.0.0"}),
        ("failure_types", ()),
        ("domain", "other"),
    ],
)
def test_every_required_dimension_and_unknown_fail_closed(field, value):
    q = Query(
        **conditions().model_dump(),
        run_id="RUN:SYNTHETIC",
        case_code="CASE:SYNTHETIC",
        test_context="fixture",
    )
    q = validate(Query, q.model_copy(update={field: value}))
    assert mismatch(binding(), q) == "memory_match_" + field


def test_tool_versions_and_zero_missing_dimensions_do_not_infer_match():
    b = binding()
    c = b.conditions.model_copy(update={"tool_versions": {"tool": "1.0.0"}})
    b = b.model_copy(update={"conditions": c})
    q = Query(
        **conditions().model_dump(),
        run_id="RUN:SYNTHETIC",
        case_code="CASE:SYNTHETIC",
        test_context="fixture",
    )
    assert mismatch(b, q) == "memory_match_tool_versions"
    assert mismatch(binding(), q) is None


def provider_fixture(replies):
    config = load_provider_config(environ={})
    provider = DeterministicFakeProvider(config, replies=replies, sleep=lambda _: None)
    reader, _ = materials(config=config)
    material = reader.current().material
    budget = RequestBudget(
        run_id=material.scope.run_id,
        deadline=datetime.now(UTC) + timedelta(minutes=2),
        max_requests=config.max_requests,
        journal=MemoryJournal(),
    )
    payload = ReviewInput(
        scope=material.scope.model_dump(mode="json"), binding=binding(), sources=(digest("source"),)
    )
    return SimpleNamespace(provider=provider, budget=budget, payload=payload, scope=material.scope)


def test_shared_provider_retry_budget_callbacks_and_admission():
    a = Advice(
        target_checksum=binding().target_checksum,
        binding_checksum=digest(binding()),
        decision="needs_more_cases",
        reasons=("synthetic_missing_support",),
    )
    x = provider_fixture((Reply(error="timeout"), Reply(content=a.model_dump_json().encode())))
    before = []
    admitted = []
    result = x.provider.generate_memory_review(
        input_payload=x.payload,
        scope=x.scope,
        budget=x.budget,
        deadline=x.budget.deadline,
        before_request=lambda: before.append(x.budget.used),
        admit=lambda v: admitted.append(v),
    )
    assert result == a and before == [0, 1] and admitted == [a]
    assert x.budget.used == 2 and x.budget.summary().api_requests_observed == 0
    assert all(
        r.simulated and r.prompt_version == "memory-review-1.0.0" for r in x.budget.records()
    )


def test_retry_rechecks_before_reserving_again():
    x = provider_fixture((Reply(error="timeout"),))

    def check():
        if x.budget.used:
            raise ValueError("source_withdrawn")

    with pytest.raises(ValueError, match="source_withdrawn"):
        x.provider.generate_memory_review(
            input_payload=x.payload,
            scope=x.scope,
            budget=x.budget,
            deadline=x.budget.deadline,
            before_request=check,
        )
    assert x.budget.used == 1


def test_wrong_target_return_rejected_and_deadline_budget_preserved():
    a = Advice(
        target_checksum=digest("wrong"),
        binding_checksum=digest(binding()),
        decision="candidate_supported",
        reasons=("fixture",),
    )
    x = provider_fixture((Reply(content=a.model_dump_json().encode()),))

    def reject(a):
        assert a.target_checksum == binding().target_checksum

    with pytest.raises(ValueError, match="scope"):
        x.provider.generate_memory_review(
            input_payload=x.payload,
            scope=x.scope,
            budget=x.budget,
            deadline=x.budget.deadline,
            admit=reject,
        )
    assert x.budget.used == 1 and x.budget.records()[0].error == "scope"
    with pytest.raises(ValueError, match="deadline"):
        x.provider.generate_memory_review(
            input_payload=x.payload,
            scope=x.scope,
            budget=x.budget,
            deadline=datetime.now(UTC) - timedelta(seconds=1),
        )
    assert x.budget.used == 1


def test_separate_groups_deduplicate_cases_unknown_io_is_not_empty_library():
    from packages.memory_review.models import Publication
    from packages.retrieval.memory import MemoryRetrieval
    from tests.memory_review.support import QueryOwner

    p = policy()
    b = binding()
    pub = Publication(
        key="pub",
        binding=b,
        policy_key=p.key,
        policy_checksum=digest(p),
        opinions=("review",),
        valid_from=p.valid_from,
        expires_at=p.expires_at,
        test_context=p.test_context,
    )
    # Pure retrieval boundary only. Real owner reconstruction is tested in the PG gate.
    # Serializable source/record values are used to preserve exact checksum comparisons.
    from packages.grounding.models import Value

    class Src(Value):
        case_code: str

    class Rec(Value):
        key: str
        schema_version: str
        sources: tuple[Src, ...]

    record = Rec(
        key=b.target, schema_version=b.content_version, sources=(Src(case_code="CASE:PAST"),)
    )

    class Repo:
        def __init__(self):
            self.contexts = SimpleNamespace(
                read_research_run=lambda _: SimpleNamespace(
                    synthetic=True,
                    request=SimpleNamespace(scope=SimpleNamespace(case_code="CASE:NOW")),
                )
            )
            self.failure = None

        def publication_keys(self):
            return ("pub", "pub2")

        def read_publication(self, key):
            if self.failure:
                raise self.failure
            return pub if key == "pub" else pub.model_copy(update={"key": "pub2"})

        def read(self, key):
            return SimpleNamespace(record=record)

    repo = Repo()
    owner = QueryOwner("RUN:NOW", "CASE:NOW")
    retrieval = MemoryRetrieval(repository=repo, query_reader=owner)
    result = retrieval.retrieve("RUN:NOW")
    assert result.status == "ok" and len(result.groups["recovery"]) == 1
    assert not result.groups["success"] and not result.groups["counterfactual"]
    assert result.rejected[-1]["reason"] == "memory_duplicate_support"
    forged = dict(result.groups["recovery"][0], branches=("ILLEGAL",))
    with pytest.raises(ValueError, match="memory_hit_binding"):
        retrieval.recheck(result.query, [forged])
    repo.failure = ValueError("unexpected_decoder_failure")
    result = retrieval.retrieve("RUN:NOW")
    assert result.status == "technical_failure" and not any(result.groups.values())
    repo.failure = ValueError("memory_qualification_revoked")
    result = retrieval.retrieve("RUN:NOW")
    assert result.status == "ok" and not any(result.groups.values()) and len(result.rejected) == 2
    owner.tool_expired = True
    assert retrieval.retrieve("RUN:NOW").status == "not_applicable"
