"""Platform Provider -> original budget/transport/admit, no network or real key."""

from datetime import timedelta

import pytest

from apps.api.security import platform_provider
from packages.evidence_details.review_tasks import project_review_tasks, review_payload
from packages.grounding.models import canonical, digest
from packages.llm_provider.config import load_provider_config, prompt
from packages.llm_provider.models import ReviewProposal
from packages.llm_provider.transport import DeterministicFakeTransport, Reply, wire_request
from packages.platform_security import Approval, EgressDenied, EgressGuard, EgressRequest, Policy
from packages.platform_security.egress import BoundEgressOwner
from tests.agent_provider.support import budget_for, materials, ref, review


def setup():
    config = load_provider_config(environ={})
    reader, b = materials(config=config)
    scope = reader.state.material.scope
    payload = review_payload(reader.state.material, project_review_tasks(reader.state), "A")
    setting, text = prompt(config, "review")
    wire = wire_request(
        config.transport, setting, text, payload, ReviewProposal.model_json_schema(), ()
    )
    request = EgressRequest(
        scope=scope,
        role="review",
        config_checksum=digest(config),
        payload_checksum=digest(payload),
        wire_checksum=digest(wire),
        wire_bytes=len(canonical(wire)),
    )
    policy = Policy(policy_id="POLICY:TEST_EGRESS", max_wire_bytes=16777216)
    approval = Approval(
        reference=ref("approval", "SOURCE:EGRESS"),
        request_checksum=digest(request),
        policy_checksum=digest(policy),
        data_class="deidentified",
        visible_projection_approved=True,
        expires_at=b.deadline,
        active=True,
        source_references=tuple(s.reference for s in reader.state.material.sources),
    )
    state = {"policy": policy, "approval": approval, "reads": 0, "run_active": True}

    def read_approval(_):
        state["reads"] += 1
        return state["approval"]

    def check_run(current):
        if not state["run_active"] or current != scope:
            raise EgressDenied()

    owner = BoundEgressOwner(
        read_policy=lambda: state["policy"],
        read_approval=read_approval,
        materials=reader,
        check_run=check_run,
    )
    transport = DeterministicFakeTransport([review(scope).model_dump(mode="json")])
    provider = platform_provider(config, egress_owner=owner, transport=transport)
    budget = budget_for(b, config)
    kwargs = dict(
        role="review",
        input_payload=payload,
        output_schema=ReviewProposal,
        scope=scope,
        budget=budget,
        deadline=b.deadline,
    )
    return provider, kwargs, state, reader, request


def test_approved_exact_projection_uses_three_current_checks():
    p, args, state, reader, _ = setup()
    accepted = []
    assert p.generate_structured(**args, admit=accepted.append) == review(
        reader.state.material.scope
    )
    assert len(accepted) == 1 and state["reads"] == 3
    assert args["budget"].used == 1 and len(p.transport.calls) == 1
    events = [
        e
        for e in args["budget"].journal.read()
        if e.get("schema_version") == "platform-egress-check/1.0.0"
    ]
    assert len(events) == 1 and len(events[0]["approved_checks"]) == 3
    assert events[0]["response_eligible"] and not events[0]["current_authorization"]


@pytest.mark.parametrize(
    "change",
    [
        "private",
        "unknown",
        "expired",
        "withdrawn",
        "input",
        "config",
        "projection",
        "sources",
        "policy",
        "limit",
        "run",
    ],
)
def test_denied_before_reservation_and_transport(change):
    p, args, state, reader, _ = setup()
    updates = {
        "private": {"data_class": "private"},
        "unknown": {"data_class": "unknown"},
        "expired": {"expires_at": args["deadline"] - timedelta(days=2)},
        "withdrawn": {"active": False},
        "input": {"request_checksum": digest("other")},
        "projection": {"visible_projection_approved": False},
        "sources": {"source_references": ()},
    }
    if change in updates:
        state["approval"] = state["approval"].model_copy(update=updates[change])
    elif change == "config":
        p.config = p.config.model_copy(update={"max_requests": p.config.max_requests + 1})
    elif change == "policy":
        state["policy"] = state["policy"].model_copy(update={"active": False})
    elif change == "limit":
        state["policy"] = state["policy"].model_copy(update={"max_wire_bytes": 1})
    else:
        state["run_active"] = False
    with pytest.raises(EgressDenied):
        p.generate_structured(**args)
    assert not p.transport.calls and args["budget"].used == 0


def test_revocation_after_reservation_records_no_send(monkeypatch):
    p, args, state, _, _ = setup()
    original = args["budget"].reserve

    def reserve(*a):
        result = original(*a)
        state["approval"] = state["approval"].model_copy(update={"active": False})
        return result

    monkeypatch.setattr(args["budget"], "reserve", reserve)
    with pytest.raises(EgressDenied):
        p.generate_structured(**args)
    assert not p.transport.calls and args["budget"].used == 1
    assert args["budget"].records()[0].api_requests == 0


@pytest.mark.parametrize("failure", ["approval", "source", "technical"])
def test_return_recheck_accounts_response_without_acceptance_or_retry(failure):
    p, args, state, reader, _ = setup()
    accepted = []

    def reply(_):
        if failure == "approval":
            state["approval"] = state["approval"].model_copy(update={"active": False})
        elif failure == "source":
            reader.withdrawn = True
        else:
            reader.read = lambda _: (_ for _ in ()).throw(OSError("synthetic_owner_failure"))
        return review(reader.state.material.scope).model_dump(mode="json")

    p.transport.replies = [reply]
    error = (
        EgressDenied if failure == "approval" else ValueError if failure == "source" else OSError
    )
    with pytest.raises(error):
        p.generate_structured(**args, admit=accepted.append)
    assert not accepted and len(p.transport.calls) == 1 and args["budget"].used == 1
    assert args["budget"].records()[0].error == "configuration"


def test_each_retry_rechecks_current_approval():
    p, args, state, _, _ = setup()
    p.transport.replies = [Reply(error="rate_limit"), Reply(error="no_result")]
    p.sleep = lambda _: state.update(
        approval=state["approval"].model_copy(update={"active": False})
    )
    with pytest.raises(EgressDenied):
        p.generate_structured(**args)
    assert len(p.transport.calls) == 1 and args["budget"].used == 1


def test_unknown_never_approved_and_private_needs_separate_permission():
    p, _, state, _, request = setup()
    policy = state["policy"].model_copy(update={"allow_private_with_approval": True})
    state["policy"] = policy
    state["approval"] = state["approval"].model_copy(
        update={"policy_checksum": digest(policy), "data_class": "private"}
    )
    with pytest.raises(EgressDenied):
        p.egress.check(request)
    state["approval"] = state["approval"].model_copy(
        update={"independent_private_approval": ref("approval", "SOURCE:PRIVATE_APPROVAL")}
    )
    with pytest.raises(EgressDenied):
        p.egress.check(request)
    p.egress.owner.read_private_approval = lambda reference, request: reference
    p.egress.check(request)
    state["approval"] = state["approval"].model_copy(update={"data_class": "unknown"})
    with pytest.raises(EgressDenied):
        p.egress.check(request)


def test_missing_owner_fails_closed():
    with pytest.raises(ValueError, match="owner_required"):
        platform_provider(load_provider_config(environ={}), egress_owner=None)
    with pytest.raises(ValueError, match="owner_required"):
        EgressGuard(None)


def test_scope_and_images_cannot_reuse_approval():
    p, _, _, _, request = setup()
    for update in (
        {"scope": request.scope.model_copy(update={"case_code": "CASE:OTHER"})},
        {"image_checksums": (digest("other_image"),)},
    ):
        with pytest.raises(EgressDenied):
            p.egress.check(request.model_copy(update=update))
