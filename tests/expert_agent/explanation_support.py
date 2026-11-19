"""Original Planner/HTTP parsing and durable owner with synthetic response only."""

import json
from threading import Event

from apps.vectorengine_live_assembly import run_case_stages
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.recorded_explanation import RecordedBatchRequests
from packages.grounding.models import digest, validate
from packages.llm_provider.transport import HTTPTransport, Reply
from packages.planner_control.models import Command
from tests.evaluation_preparation.test_directional_review import fixture, reply
from tests.expert_agent.test_batch_assembly import assembly


def failed_explanation(monkeypatch, tmp_path, pg=None):
    template = fixture()
    config = template.config.model_copy(
        update={
            "mode": "live",
            "transport": template.config.transport.model_copy(update={"max_retries": 0}),
        }
    )

    # The synthetic dynamic fixture has five obtainable observations; use one
    # selected requirement for this exact one-response recovery contract.
    def initial(binding, _config, count):
        raw = template.owner.current().model_dump(mode="json")
        p, m = raw["policy"], raw["material"]
        item = p["items"][0]
        p["items"] = [item]
        p["groups"] = [g for g in p["groups"] if item["id"] in sum(g["slots"], [])]
        for group in p["groups"]:
            group["slots"] = [[item["id"]]]
        p["requirements"] = [r for r in p["requirements"] if r["id"] == item["id"]]
        m["observations"] = [
            o for o in m["observations"] if o["reference"]["code"] == item["observation_code"]
        ]
        m["scope"].update(
            case_code=binding.case_code,
            run_id=binding.run_id,
            graph=binding.graph.model_dump(mode="json"),
            input_checksum=binding.input_checksum,
            visible_checksum=binding.visible_checksum,
            policy_checksum=digest(p),
        )
        m["policy_ref"]["checksum"] = digest(p)
        m["scope"].pop("material_checksum")
        m["scope"]["material_checksum"] = digest(m)
        raw["config_checksum"] = digest(config)
        if pg is not None:
            from apps.vectorengine_case_materials import ref

            raw["resources"] = [
                ref("resource", pg.root.asset_code, pg.root.checksum).model_dump(mode="json")
            ]
            item["dependency_paths"] = [[pg.root.asset_code]]
            item["quality"]["source"] = raw["resources"][0]
            m["scope"]["policy_checksum"] = m["policy_ref"]["checksum"] = digest(p)
            m["scope"].pop("material_checksum")
            m["scope"]["material_checksum"] = digest(m)
        return validate(DomainState, raw), b"not_an_image_no_vision_dispatch"

    def duplicate(body):
        value = reply(body)
        token = '"next_action": null'
        text = json.dumps(value)
        assert token in text
        return Reply(
            content=text.replace(token, token + ", " + token, 1).encode(),
            api_requests=1,
            response_confirmed=True,
            finish_reason="stop",
        )

    monkeypatch.setattr("tests.expert_agent.test_batch_assembly.initial_state", initial)
    monkeypatch.setattr("tests.expert_agent.test_batch_assembly.load_config", lambda _: config)
    f = (
        assembly(monkeypatch, replies=[duplicate])
        if pg is None
        else pg_assembly(pg, initial, config, duplicate)
    )
    monkeypatch.setattr(
        HTTPTransport,
        "send",
        lambda self, config, body, **kw: self._observed(
            f.wire.send(config, body, **kw), {"http_status": 200}, body, "chat_completions"
        ),
    )
    run_case_stages(
        planner=f.planner,
        owner=f.owner,
        ports=f.ports,
        review=f.review,
        documents=f.documents,
        binding=f.binding,
        config=f.config,
        output=tmp_path,
        signal_cancel=Event(),
        selected_roles=("review",),
    )
    current = f.planner.read(f.binding.run_id)
    if current.phase != "stopped":
        f.planner.step(
            f.binding.run_id,
            Command(key="close-failed", expected_revision=current.revision, action="stop"),
        )
    intent = f.documents.read("live-stage-intent:review")[0]
    raw = f.documents.read("review:" + intent["event"]["code"] + ":intent")[0]["state"]
    f.state = validate(DomainState, raw)
    phase = dict(
        run_id=f.binding.run_id,
        target=None,
        accepted=False,
        error="candidate_failed_or_context_changed",
    )
    f.failed = dict(complete=False, phases=[phase])
    with f.documents.session():
        f.documents.write("dynamic-phase-result", (phase,))
        f.documents.write("multimodal-case-result", (f.failed,))
    f.requests = RecordedBatchRequests(f.ports.budget.journal)
    return f


def pg_assembly(pg, initial, config, duplicate):
    from types import SimpleNamespace

    from apps.vectorengine_live_assembly import assemble, make_binding
    from packages.grounding.models import seal
    from packages.research_runtime.agent_records import RuntimeDocuments
    from packages.research_runtime.models import ResearchRunContext
    from packages.tool_admission.models import ResearchRequest
    from tests.agent_provider.test_bounded_live import SyntheticWire

    contexts, sessions = pg.documents.contexts, pg.documents.sessions
    raw = contexts.read_research_run(pg.binding.run_id).request.model_dump(
        mode="json", exclude={"checksum"}
    )
    raw["code"], raw["lock"]["run_id"] = "RESEARCH:EXPLANATION", "RUN:OID_EXPLANATION"
    request = seal(ResearchRequest, **raw)
    context = seal(
        ResearchRunContext, request=request, synthetic=True, scope_checksum=digest(request.scope)
    )
    contexts.create_research_run(context, "explanation-test-context")
    state, image = initial(pg.binding.model_copy(update={"run_id": context.run_id}), config, 1)
    started, sha = sessions.origin(context.run_id)
    binding, ranking = make_binding(
        initial=state,
        config=config,
        started=started,
        context_checksum=sha,
        catalog=request.lock.catalog_ref,
    )
    docs = RuntimeDocuments(
        repo=pg.repo,
        contexts=contexts,
        sessions=sessions,
        store=pg.store,
        run_id=context.run_id,
        root=pg.root,
    )

    def verify(current):
        assert current.material.sources == state.material.sources and current.policy == state.policy

    evidence = SimpleNamespace(verify=verify, image=lambda _: image)
    planner, owner, ports, review = assemble(
        evidence=evidence,
        initial=state,
        config=config,
        documents=docs,
        binding=binding,
        ranking=ranking,
    )
    planner.begin(binding)
    return SimpleNamespace(
        planner=planner,
        owner=owner,
        ports=ports,
        review=review,
        documents=docs,
        binding=binding,
        config=config,
        wire=SyntheticWire([duplicate]),
    )
