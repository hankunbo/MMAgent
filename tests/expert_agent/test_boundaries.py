import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest

from agents.expert_agent.formal import validate_formal_assessment
from agents.expert_agent.reporting import render_research_html
from packages.contracts import ExpertAssessment, ExpertAssessmentDecision, PlanAction
from packages.grounding.models import digest, seal
from packages.llm_provider.transport import Reply
from packages.planner_control.models import LegalActionSet
from packages.rule_engine import LegalAction
from tests.agent_provider.support import review_from_wire
from tests.expert_agent.support import expert_reply, run, setup


def test_formal_v1_preserved_published_gate_and_no_research_cast():
    raw = json.loads(Path("packages/contracts/examples/v1/ExpertAssessment.valid.json").read_text())
    raw.update(
        decision=ExpertAssessmentDecision.SELECT_ALLOWED_ALTERNATIVE,
        selected_action_code=PlanAction.STOP,
        allowed_action_codes=[PlanAction.STOP],
    )
    assessment = ExpertAssessment.model_validate_json(json.dumps(raw))
    legal = (LegalAction(PlanAction.STOP, None, 1, "sha256:" + "a" * 64, "RULE:SYNTHETIC_RULE"),)

    class Published:
        def __init__(self):
            self.calls = []

        def validate_reference(self, reference, **kwargs):
            self.calls.append((reference, kwargs))
            if kwargs["purpose"] != "approved_formal_use":
                raise ValueError("purpose_rejected")

    owner = Published()
    a, selected = validate_formal_assessment(
        assessment,
        legal_actions=legal,
        published_owner=owner,
        expected_scope=assessment.scope,
        expected_proposal=assessment.proposal_code,
        purpose="approved_formal_use",
    )
    assert a == assessment and selected == legal[0] and len(owner.calls) == len(a.knowledge_refs)
    with pytest.raises(ValueError, match="purpose_rejected"):
        validate_formal_assessment(
            assessment,
            legal_actions=legal,
            published_owner=owner,
            expected_scope=assessment.scope,
            expected_proposal=assessment.proposal_code,
            purpose="research_material",
        )
    raw["knowledge_refs"] = []
    with pytest.raises(ValueError):
        ExpertAssessment.model_validate_json(json.dumps(raw))
    with pytest.raises(ValueError):
        validate_formal_assessment(
            dict(schema_version="expert-proposal/1.0.0"),
            legal_actions=legal,
            published_owner=owner,
            expected_scope=assessment.scope,
            expected_proposal=assessment.proposal_code,
            purpose="research_material",
        )


@pytest.mark.parametrize(
    "change", ["candidate_set", "reset", "request_limit", "deadline_extension"]
)
def test_precise_set_reset_and_budget_binding_changes_fail_closed(change):
    f = setup(facts={"required": True})

    def mutate(_):
        if change == "candidate_set":
            fields = f.context.frame.actions.model_dump(mode="json", exclude={"checksum"})
            fields["actions"] = [a for a in fields["actions"] if a["id"] != f.proposed.id]
            f.context.frame = f.frame.model_copy(update={"actions": seal(LegalActionSet, **fields)})
        elif change == "reset":
            from packages.grounding.models import validate
            from packages.llm_provider.models import Material

            raw = f.reader.state.material.model_dump(mode="json")
            raw["scope"]["graph"]["checksum"] = digest("reset_graph")
            raw["scope"].pop("material_checksum")
            raw["scope"]["material_checksum"] = digest(raw)
            f.reader.state = f.reader.state.model_copy(update={"material": validate(Material, raw)})
        elif change == "request_limit":
            f.budget.max_requests -= 1
        else:
            f.budget.deadline += timedelta(seconds=10)

    f.provider.transport.replies = [expert_reply(mutate=mutate)]
    assert run(f)["status"] == "safe_stop"
    assert f.budget.used == 1
    assert not f.context.installs


def test_exhausted_budget_required_stop_and_bounded_transport_retries():
    f = setup([Reply(error="timeout")] * 20, facts={"required": True})
    result = run(f)
    assert result["status"] == "safe_stop"
    assert f.budget.used == f.provider.config.transport.max_retries + 1
    assert result["recorded_agent_invocations"] == 1
    assert len(f.provider.transport.calls) == f.budget.used
    assert run(f) == result
    other = setup(facts={"required": True})
    for _ in range(other.budget.max_requests):
        other.budget.reserve(
            other.frame.scope, other.binding.deadline, digest(other.provider.config)
        )
    assert run(other)["status"] == "safe_stop"
    assert len(other.provider.transport.calls) == 0


def test_abc_conflict_triggers_required_review_before_acceptance():
    def negative(body):
        out = review_from_wire(body)
        out["items"][0]["judgment"].update(value=-0.3, lower=-0.6, upper=0.0)
        return out

    f = setup([review_from_wire, negative, review_from_wire, expert_reply("reject")])
    result = run(f, role="review", abc=True)
    assert result["status"] == "reject"
    decision = f.service.store.read(f.frame.event.code + ":decision")[0]
    assert decision["required"] and "conflict" in decision["triggers"]
    assert f.acceptance.read(key=f.frame.event.code + ":domain") is None
    assert f.budget.used == 4


def test_source_binding_html_tamper_rejected():
    f = setup([review_from_wire])
    run(f, role="review")
    report = f.service.store.read(f.frame.event.code + ":report")[0]
    altered = copy.deepcopy(report)
    altered["source_uses"][0]["reference"]["code"] = "SOURCE:OTHER"
    with pytest.raises(ValueError, match="binding"):
        render_research_html(altered)


def test_parent_consumed_does_not_generate_again_under_new_event():
    f = setup([review_from_wire])
    parent = f.reader.state
    run(f, role="review")
    f.reader.state = parent
    f.context.checkpoints.clear()
    f.frame = f.context.frame = f.frame.model_copy(
        update={"event": f.frame.event.model_copy(update={"code": "review:new"})}
    )
    assert run(f, role="review")["status"] == "safe_stop"
    assert f.budget.used == 1


def test_saved_decision_reply_lost_recovers_without_second_expert(monkeypatch):
    f = setup([expert_reply()], facts={"required": True})
    write = f.documents.write

    def lost(key, values):
        write(key, values)
        if key.endswith(":decision"):
            raise OSError("decision_confirmation_lost")

    monkeypatch.setattr(f.documents, "write", lost)
    with pytest.raises(OSError):
        run(f)
    monkeypatch.setattr(f.documents, "write", write)
    assert run(f)["status"] == "accept"
    assert f.budget.used == 1


def test_trace_expert_disagreement_shares_provider_budget_and_never_runs_tool():
    from packages.planner_control.actions import baseline

    f = setup(facts={"conflict": True})
    ranking = baseline(f.frame.actions)
    # Trace advises an existing observation; Expert rejects that exact proposal.
    raw = ranking.model_dump(mode="json")
    raw["ranked_ids"] = [f.proposed.id]
    f.provider.transport.replies = [raw, expert_reply("reject")]
    trace = f.ports.trace.rank(f.frame.actions, deadline=f.binding.deadline)
    assert trace.ranked_ids[0] == f.proposed.id
    assert run(f)["status"] == "reject"
    records = f.budget.records()
    assert [r.role for r in records] == ["trace", "expert"]
    assert len({r.invocation_id for r in records}) == 2
    assert f.budget.summary().simulated_attempts == 2
    assert f.budget.summary().api_requests_observed == 0
    assert not f.context.installs


def test_model_calls_outside_document_sessions():
    from contextlib import contextmanager

    from tests.agent_provider.support import Documents

    class Tracked(Documents):
        active = 0

        @contextmanager
        def session(self):
            self.active += 1
            try:
                yield
            finally:
                self.active -= 1

    docs = Tracked()

    def reply(body):
        assert docs.active == 0
        return review_from_wire(body)

    f = setup([reply], documents=docs)
    assert run(f, role="review")["status"] == "accepted"


def test_candidate_port_scope_cannot_be_swapped():
    f = setup([review_from_wire])
    with pytest.raises(ValueError, match="candidate_port_not_authorized"):
        f.service.run(event=f.frame.event, proposed_action_id=f.proposed.id, role="review")
    assert f.budget.used == 0


def test_live_assembly_is_still_disabled():
    from agents.expert_agent import ResearchReview
    from tests.expert_agent.support import protocol

    f = setup()
    f.provider.config = f.provider.config.model_copy(update={"mode": "live"})
    with pytest.raises(ValueError, match="live_domain_acceptance_not_enabled"):
        ResearchReview(
            acceptance=f.acceptance, ports=f.ports, context=f.context, protocol=protocol()
        )
    assert f.budget.used == 0 and not f.provider.transport.calls


def test_expert_acceptance_cannot_override_unknown_domain_fact():
    from tests.agent_provider.support import review
    from tests.agent_provider.test_review_repair import observation_state

    f = setup()
    observation_state(f.reader, "unknown", None)
    f = setup(
        [review(f.reader.state.material.scope).model_dump(mode="json"), expert_reply()],
        reader=f.reader,
        binding=f.binding,
        facts={"required": True},
    )
    assert run(f, role="review")["reason"] == "domain_submission_rejected"
    assert f.budget.used == 2
    assert f.acceptance.read(key=f.frame.event.code + ":domain") is None
    assert not f.context.installs


def test_optional_expert_failure_can_use_domain_checked_baseline():
    f = setup([review_from_wire, Reply(error="refusal")], facts={"uncertain": True})
    result = run(f, role="review")
    assert result["status"] == "accepted"
    decision = f.service.store.read(f.frame.event.code + ":decision")[0]
    assert decision["decision"] == "baseline" and decision["reason"] == "optional_review_failed"
    assert f.budget.used == 2


def test_cross_scope_material_rejected_before_request():
    f = setup([review_from_wire])
    f.context.frame = f.frame.model_copy(
        update={"scope": f.frame.scope.model_copy(update={"run_id": "RUN:OTHER"})}
    )
    with pytest.raises(ValueError):
        run(f, role="review")
    assert f.budget.used == 0 and not f.context.installs


def test_saved_expert_advice_recovery_revalidates_before_return(monkeypatch):
    f = setup([expert_reply()], facts={"required": True})
    write = f.documents.write

    def lost(key, values):
        write(key, values)
        if key.endswith(":decision"):
            raise OSError("decision_confirmation_lost")

    monkeypatch.setattr(f.documents, "write", lost)
    with pytest.raises(OSError):
        run(f)
    monkeypatch.setattr(f.documents, "write", write)
    f.reader.withdrawn = True
    assert run(f)["status"] == "safe_stop"
    assert f.budget.used == 1
