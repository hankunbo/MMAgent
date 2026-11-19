"""Synthetic replies and original persistence owners; no socket or patient."""

from datetime import timedelta
from types import SimpleNamespace

from agents.planner.service import Planner
from agents.provider_agents import StructuredRoleAgent
from agents.provider_agents.batched_review import run_batches
from apps.vectorengine_single_case import load_config
from packages.evidence_details.finalization import ConfirmedReviewPermission, RecordedRequests
from packages.evidence_details.live_permission import ResearchLiveGrant, ResearchLivePermission
from packages.grounding.models import digest
from packages.llm_provider.provider import RequestBudget, StructuredProvider
from packages.planner_control.models import Command
from tests.agent_provider.batch_support import CONFIG, batch_reply
from tests.agent_provider.support import Documents, Reader
from tests.agent_provider.test_bounded_live import SyntheticJournal, SyntheticWire
from tests.expert_agent.batch_case_support import initial_state
from tests.planner.support import FakeOwner, MemoryRuntime, binding


def fixture(
    pg=None,
    resource=None,
    *,
    partial=False,
    corrected=False,
    reasoned=False,
    deadline_partial=False,
    deadline_cleanup_cancel=False,
    expired_clock=True,
    dynamic_policy=False,
    citation_rejection=False,
    corrected_partial=False,
    raw_only_deadline=False,
):
    config = load_config(
        "configs/agents/vectorengine_batch_integrated.json" if corrected else CONFIG
    )
    if reasoned:
        from packages.multimodal_research.campaign import configuration

        config = configuration(directional=True)
    b = pg.binding if pg else binding()
    docs = pg.documents if pg else Documents()
    if pg:
        # pg_support already owns a Fake budget; do not reset it to a live budget.
        from packages.grounding.models import seal
        from packages.research_runtime.agent_records import RuntimeDocuments
        from packages.research_runtime.models import ResearchRunContext
        from packages.tool_admission.models import ResearchRequest

        old = docs.contexts.read_research_run(b.run_id)
        raw = old.request.model_dump(mode="json", exclude={"checksum"})
        raw["code"], raw["lock"]["run_id"] = "RESEARCH:FINALIZATION", "RUN:OID_FINALIZATION"
        request = seal(ResearchRequest, **raw)
        context = seal(
            ResearchRunContext,
            request=request,
            synthetic=True,
            scope_checksum=digest(request.scope),
        )
        docs.contexts.create_research_run(context, "finalization-synthetic-context")
        started, context_sha = docs.sessions.origin(context.run_id)
        b = b.model_copy(
            update=dict(
                run_id=context.run_id,
                started_at=started,
                run_context_checksum=context_sha,
                deadline=started + timedelta(seconds=300),
            )
        )
        docs = RuntimeDocuments(
            repo=pg.repo,
            contexts=docs.contexts,
            sessions=docs.sessions,
            store=pg.store,
            run_id=context.run_id,
            root=pg.root,
        )
    state, _ = initial_state(b, config, count=4)
    if partial or dynamic_policy:
        from packages.evidence_details.acceptance_models import DomainState
        from packages.grounding.models import validate
        from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture

        if reasoned:
            from tests.evaluation_preparation.test_directional_review import fixture as directional

            raw = directional().owner.current().model_dump(mode="json")
        else:
            raw = dynamic_fixture().owner.current().model_dump(mode="json")
        material = raw["material"]
        material["scope"].update(
            case_code=b.case_code,
            run_id=b.run_id,
            graph=b.graph.model_dump(mode="json"),
            input_checksum=b.input_checksum,
            visible_checksum=b.visible_checksum,
        )
        material["scope"].pop("material_checksum")
        material["scope"]["material_checksum"] = digest(material)
        raw["config_checksum"] = digest(config)
        if corrected:
            raw["prompt_checksum"] = next(
                r.prompt_checksum for r in config.roles if r.role == "review"
            )
        state = validate(DomainState, raw)
    if resource is not None:
        from apps.vectorengine_case_materials import ref
        from packages.evidence_details.acceptance_models import DomainState
        from packages.grounding.models import validate

        raw = state.model_dump(mode="json")
        raw["resources"] = [
            ref("resource", resource.asset_code, resource.checksum).model_dump(mode="json")
        ]
        for item in raw["policy"]["items"]:
            item["dependency_paths"] = [[resource.asset_code]]
            if item.get("quality") is not None:
                item["quality"]["source"] = raw["resources"][0]
        raw["material"]["scope"]["policy_checksum"] = digest(raw["policy"])
        raw["material"]["policy_ref"]["checksum"] = digest(raw["policy"])
        raw["material"]["scope"].pop("material_checksum")
        raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
        state = validate(DomainState, raw)
    owner = Reader(state)
    if not pg:
        docs.run_id = b.run_id
        docs.root = SimpleNamespace(checksum=b.graph.checksum)
        docs.sessions = MemoryRuntime()
    planner = Planner(runtime=docs.sessions, owner=FakeOwner(b))
    planner.begin(b)
    if pg:
        from packages.research_runtime.agent_records import RuntimeAgentJournal

        journal = RuntimeAgentJournal(docs)
    else:
        journal = SyntheticJournal()
        journal.documents = docs
    budget = RequestBudget(run_id=b.run_id, deadline=b.deadline, max_requests=40, journal=journal)
    old = ResearchLivePermission(
        grant=ResearchLiveGrant(
            enabled=True,
            case_code=b.case_code,
            run_id=b.run_id,
            input_checksum=b.input_checksum,
            visible_checksum=b.visible_checksum,
            policy_checksum=state.material.scope.policy_checksum,
            config_checksum=digest(config),
            root_checksum=docs.root.checksum,
            purpose="bounded_single_case_research",
            deadline=b.deadline,
            roles=("review",),
        ),
        config=config,
        documents=docs,
        budget=budget,
    )

    def wrong_binding(body):
        import json

        from packages.llm_provider.transport import Reply

        value = batch_reply(body)
        if citation_rejection:
            assert partial
            value["items"][0]["observation_citations"] = ["SRC:OUTSIDE_OBSERVATION"]
        else:
            value["scope_checksum"] = digest("wrong-scope")
        return Reply(
            content=json.dumps(value).encode(),
            api_requests=1,
            response_confirmed=True,
            finish_reason="stop",
        )

    wire = SyntheticWire([batch_reply, wrong_binding if partial else batch_reply])
    if deadline_partial:
        assert partial and not corrected
        response = batch_reply
        if reasoned:
            from tests.evaluation_preparation.test_directional_review import reply

            response = reply
        wire = SyntheticWire([response])
    transport = wire
    if corrected:
        from packages.llm_provider.transport import HTTPTransport
        from tests.agent_provider.test_integrated_review import malformed

        response = batch_reply
        if reasoned:
            from tests.evaluation_preparation.test_directional_review import reply

            response = reply
        wire = SyntheticWire([malformed, response, response])

        class InjectedHTTP(HTTPTransport):
            def send(self, config, body, **kwargs):
                return self._observed(
                    wire.send(config, body, **kwargs),
                    {"http_status": 200},
                    body,
                    "chat_completions",
                )

        transport = InjectedHTTP()
    provider = StructuredProvider(config, live_opt_in=True, transport=transport)
    agent = StructuredRoleAgent("review", provider=provider, budget=budget, materials=owner)
    with docs.session():
        docs.write("live-stage-intent:review", (dict(event=dict(code="synthetic-event")),))
        if raw_only_deadline:
            assert partial and deadline_partial and not corrected
            docs.write("dynamic-phase-input", (state,))
    if corrected_partial:
        import pytest

        from packages.evidence_details.review_batches import collect_partial
        from packages.llm_provider.provider import ProviderError

        assert corrected and reasoned and partial

        def stop_after_one_confirmed():
            if docs.read("review-batches:synthetic-event:A:batch:000:result"):
                raise ProviderError("budget")

        with pytest.raises(ProviderError, match="budget"):
            run_batches(
                agent,
                state=state,
                documents=docs,
                key="synthetic-event:A",
                round_id="A",
                before_request=stop_after_one_confirmed,
            )
        submission = collect_partial(state, docs, "synthetic-event:A")
    elif partial and not corrected:
        import pytest

        from packages.evidence_details.review_batches import collect_partial
        from packages.llm_provider.provider import ProviderError

        write = docs.write

        def stop_after_saved_result(key, values):
            if raw_only_deadline and key.endswith(":batch:000:result"):
                budget.clock = lambda: b.deadline
                raise ProviderError("deadline")
            result = write(key, values)
            if deadline_partial and key.endswith(":batch:000:result"):
                budget.clock = lambda: b.deadline
            return result

        docs.write = stop_after_saved_result
        try:
            with pytest.raises(ProviderError, match="deadline" if deadline_partial else "scope"):
                run_batches(
                    agent, state=state, documents=docs, key="synthetic-event:A", round_id="A"
                )
        finally:
            docs.write = write
        submission = (
            None if raw_only_deadline else collect_partial(state, docs, "synthetic-event:A")
        )
    else:
        submission = run_batches(
            agent, state=state, documents=docs, key="synthetic-event:A", round_id="A"
        )
    planner.step(
        b.run_id,
        Command(
            key="stop",
            expected_revision=planner.read(b.run_id).revision,
            action="cancel" if deadline_cleanup_cancel else "stop",
        ),
    )
    failed = dict(complete=False, phases=[dict(run_id=b.run_id, error="deadline")])
    with docs.session():
        docs.write("dynamic-case-result", (failed,))
        if partial or corrected:
            phase = dict(
                run_id=b.run_id,
                target=None,
                accepted=False,
                error="deadline"
                if corrected or deadline_partial
                else "candidate_failed_or_context_changed",
            )
            failed = dict(complete=False, phases=[phase])
            if corrected:
                failed.update(
                    case_code=b.case_code,
                    task=dict(target_code=state.nodes[0]),
                    new_requests=len(wire.calls),
                    previous_attempts=[],
                )
            docs.write("dynamic-phase-result", (phase,))
            docs.write("multimodal-case-result", (failed,))
    now = b.deadline + timedelta(seconds=1)
    if expired_clock:
        budget.clock = lambda: now
    if corrected:
        from packages.evidence_details.recorded_finalization import RecordedBatchRequests

        requests = RecordedBatchRequests(journal)
    elif partial:
        from packages.evidence_details.partial_finalization import ClosedReviewRequests

        requests = ClosedReviewRequests(journal)
    else:
        requests = RecordedRequests(journal)
    return SimpleNamespace(
        config=config,
        state=state,
        owner=owner,
        docs=docs,
        old=old,
        requests=requests,
        budget=budget,
        wire=wire,
        submission=submission,
        failed=failed,
        now=now,
        planner=planner,
    )


def permission(f):
    return ConfirmedReviewPermission.prepare(
        owner=f.owner, documents=f.docs, requests=f.requests, config=f.config, clock=lambda: f.now
    )
