"""Explicit successor after a stopped public-search failure; never replay that query.

The original case, confirmed native artifacts, absolute deadline and wire budget
are retained. Only an otherwise unstarted acquisition can use this narrow port.
"""

import fcntl
import json
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest

ROUTED_BACKLOG = "routed_root_source_backlog_no_source_replay"
PUBLIC_WIRE_POLICY = "closed-public-wire-backlog/1.0.0"


def public_wire_reference(reference, *, routed):
    """One version dispatch shared by assembly, native reuse and continuation."""
    if "public_wire_policy" not in reference:
        return {}
    if not routed or reference["public_wire_policy"] != PUBLIC_WIRE_POLICY:
        raise ValueError("definition_successor_public_wire_policy")
    return dict(public_wire_policy=PUBLIC_WIRE_POLICY)


def closed_wire_absence(docs, operation, saved, protocol):
    """Validate the original connector's finally receipt, not remote success.

    Missing receipts remain UNKNOWN. Complete no-response receipts are retained
    as unavailable material, with the original reservation charged. Neither
    condition authorizes replay or use as medical source material.
    """
    from datetime import datetime

    from pydantic import ValidationError

    from packages.knowledge_connectors.ports import WireReceipt

    if not saved or len(saved) != 1:
        raise ValueError("definition_successor_wire_unknown")
    try:
        receipt = WireReceipt.model_validate_json(json.dumps(saved[0]))
    except ValidationError as exc:
        raise ValueError("definition_successor_wire_receipt") from exc
    reservation = docs.read(operation["key"] + ":result")
    prefix, separator, index = operation["key"].rpartition(":")
    if prefix != "definition:wire" or not separator or not index.isdecimal():
        raise ValueError("definition_successor_wire_operation_key")
    operation_key = "definition:operation:" + index
    original = docs.read(operation_key)
    if not original or len(original) != 1:
        raise ValueError("definition_successor_wire_operation_missing")
    original = original[0]
    binding = original.get("binding", {})
    # Failure snapshots intentionally project only key/kind/confirmed. Read
    # request kind and binding from the original persisted reservation owner.
    if (
        original.get("key") != operation["key"]
        or original.get("kind") != "wire"
        or binding.get("campaign_id") != protocol.get("campaign_id")
        or binding.get("input_checksum") != protocol["input"]
        or binding.get("protocol_checksum") != digest(protocol)
        or binding.get("policy_checksum") != digest(protocol["definition_expansion"])
        or binding.get("target") != protocol["definition_expansion"]["public_target"]
    ):
        raise ValueError("definition_successor_wire_operation_binding")
    if (
        receipt.http_status is not None
        or receipt.response_checksum is not None
        or receipt.operation not in {"search", "fetch"}
        or receipt.operation != original.get("wire_kind")
        or reservation != (dict(provider=receipt.provider_code, reservation_only=True),)
        or receipt.requested_at.tzinfo is None
        or not datetime.fromisoformat(protocol["started_at"])
        <= receipt.requested_at
        <= datetime.fromisoformat(protocol["deadline"])
    ):
        raise ValueError("definition_successor_wire_receipt_binding")
    return [
        (
            operation["key"] + ":result",
            (dict(provider=receipt.provider_code, reservation_only=True),),
        ),
        (operation_key, (json.loads(json.dumps(original)),)),
    ]


def definition_provider_transition(previous, following_provider, following_policy):
    """Derive a narrow successor binding, not permission to replay any request.

    The original owner below must still prove terminal Source accounting, no
    acquisition/Review starts, current facts, and the unique successor claim.
    Old references remain byte-for-byte unchanged when the provider is unchanged.
    """
    from packages.llm_provider.config import read_provider_config
    from packages.llm_provider.review_rating import apply, check_entry_binding

    check_entry_binding(previous)
    old = read_provider_config(previous["provider"])
    new = read_provider_config(following_provider)
    prior_policy = previous.get("review_contract")
    if old == new and prior_policy == following_policy:
        return None
    if (
        previous.get("schema_version") != "multimodal-live-entry/1.29.0"
        or previous.get("execution_profile") != "relay-root-question-v1"
        or previous.get("definition_failure_of") is not None
        or previous.get("network_route") != old.model_dump(mode="json").get("network_route")
        or old.schema_version != "agent-provider-config/9.1.0"
        or new.schema_version != old.schema_version
        or following_policy != "rating-presence/1.1.0"
        or prior_policy not in {None, "rating-presence/1.0.0", "rating-presence/1.1.0"}
        or not old.network_route
        or not new.network_route
    ):
        raise ValueError("definition_successor_provider_transition_scope")
    revised = old if prior_policy == following_policy else apply(old, following_policy)
    expected = revised.model_dump(mode="json")
    expected["network_route"] = new.network_route.model_dump(mode="json")
    if read_provider_config(expected) != new:
        raise ValueError("definition_successor_provider_transition_delta")
    return dict(
        schema_version="source-fallback-provider-transition/1.0.0",
        previous_provider=digest(old),
        following_provider=digest(new),
        previous_review_contract=prior_policy,
        following_review_contract=following_policy,
        previous_route=digest(old.network_route),
        following_route=digest(new.network_route),
        source_and_vision_roles_unchanged=True,
        model_and_budgets_unchanged=True,
        original_source_replay_authorized=False,
        requires_original_owner_no_acquisition_or_review_started=True,
    )


def fallback_window(parent, *, seconds, authorization):
    """An explicit new bounded successor, never extend the stopped parent."""
    from datetime import UTC, datetime, timedelta

    if (
        type(seconds) is not int
        or not 600 <= seconds <= 5400
        or not isinstance(authorization, str)
        or not 1 <= len(authorization) <= 256
        or any(ord(c) < 32 for c in authorization)
    ):
        raise ValueError("definition_fallback_window_authorization")
    now = datetime.now(UTC)
    return dict(
        schema_version="definition-fallback-window/1.0.0",
        authorization=authorization,
        parent_deadline=parent["deadline"],
        started_at=now.isoformat(),
        deadline=(now + timedelta(seconds=seconds)).isoformat(),
        seconds=seconds,
        cumulative_request_limit=40,
        source_replay_authorized=False,
        native_execution_authorized=False,
    )


def check_fallback_window(value, parent):
    """Validate the saved window independently of its display or caller flags."""
    from datetime import datetime

    keys = {
        "schema_version",
        "authorization",
        "parent_deadline",
        "started_at",
        "deadline",
        "seconds",
        "cumulative_request_limit",
        "source_replay_authorized",
        "native_execution_authorized",
    }
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value["schema_version"] != "definition-fallback-window/1.0.0"
        or not isinstance(value["authorization"], str)
        or not 1 <= len(value["authorization"]) <= 256
        or any(ord(c) < 32 for c in value["authorization"])
        or value["parent_deadline"] != parent["deadline"]
        or type(value["seconds"]) is not int
        or not 600 <= value["seconds"] <= 5400
        or type(value["cumulative_request_limit"]) is not int
        or value["cumulative_request_limit"] != 40
        or value["source_replay_authorized"] is not False
        or value["native_execution_authorized"] is not False
    ):
        raise ValueError("definition_fallback_window_binding")
    try:
        start, deadline = (datetime.fromisoformat(value[k]) for k in ("started_at", "deadline"))
        valid = (
            start.utcoffset() is not None
            and deadline.utcoffset() is not None
            and (deadline - start).total_seconds() == value["seconds"]
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("definition_fallback_window_clock")
    return {k: value[k] for k in ("started_at", "deadline")}


def definition_parent(
    path,
    *,
    checksum,
    intent,
    defer=False,
    confirmed_backlog=False,
    root_backlog=False,
    root_confirmed=False,
    routed_backlog=False,
    history_base=None,
):
    """Read a terminal, confirmed Source-schema failure, not an UNKNOWN replay."""
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("definition_successor_symlink")
    spec = json.loads((path / "entry.json").read_bytes())
    if history_base is None:
        from .execution_lease import historical_entry_probe

        with historical_entry_probe(path, spec):
            pass
    protocol = json.loads((path / "protocol.json").read_bytes())
    failure = json.loads((path / "stage-failure.json").read_bytes())
    from packages.llm_provider.model_selection import check_entry_binding
    from packages.llm_provider.vision_correction import check_entry_binding as check_vision

    check_entry_binding(spec, protocol)
    check_vision(spec, protocol)
    from packages.llm_provider.review_rating import check_entry_binding as check_review

    check_review(spec, protocol)
    from packages.llm_provider.client_selection import check_entry_binding as check_client

    check_client(spec, protocol)
    if (
        (confirmed_backlog or root_backlog)
        and not defer
        or confirmed_backlog
        and root_backlog
        or root_confirmed
        and not root_backlog
        or routed_backlog
        and (not root_backlog or root_confirmed or confirmed_backlog)
    ):
        raise ValueError("definition_successor_backlog_mode")
    confirmed_backlog = confirmed_backlog or root_confirmed
    if routed_backlog and (
        spec.get("network_route") is None
        or spec.get("provider", {}).get("schema_version")
        not in {"agent-provider-config/9.1.0", "agent-provider-config/9.2.0"}
        or spec["provider"].get("network_route") != spec["network_route"]
        or failure.get("error_code") not in {"schema", "timeout", "transport"}
    ):
        raise ValueError("definition_successor_routed_parent_binding")
    if root_backlog:
        from packages.grounding.models import validate
        from packages.research_runtime.task_context import TaskIntent

        from .single_proposition import root_direction_spec

        question = root_direction_spec(validate(TaskIntent, intent))
        if (
            spec.get("execution_profile") != "relay-root-question-v1"
            or protocol.get("execution_profile") != "relay-root-question-v1"
            or protocol.get("schema_version") != "multimodal-live-campaign/1.16.0"
            or spec.get("root_direction_review") != question
            or protocol.get("root_direction_review") != question
            or type(spec.get("review_batch_size")) is not int
            or spec["review_batch_size"] not in (2, 3)
            or protocol.get("review_batch_size") != spec["review_batch_size"]
            or spec.get("component_consumer") is not None
            or spec.get("definition_failure_of") is not None
            or protocol.get("provider") != digest(spec["provider"])
        ):
            raise ValueError("definition_successor_root_binding")
    unknown = (
        failure["error_code"] in {"timeout", "transport"}
        if routed_backlog
        else defer and not confirmed_backlog
    )
    if (
        spec["schema_version"]
        != (
            "multimodal-live-entry/1.29.0"
            if routed_backlog
            else "multimodal-live-entry/1.26.0"
            if root_backlog
            else "multimodal-live-entry/1.23.0"
            if confirmed_backlog
            else "multimodal-live-entry/1.21.0"
            if defer
            else "multimodal-live-entry/1.20.0"
        )
        or spec["input_checksum"] != checksum
        or protocol["input"] != checksum
        or spec["intent"] != intent
        or protocol["task"] != intent["checksum"]
        or spec["implementation"] != protocol["implementation"]
        or (not (confirmed_backlog or root_backlog) and not spec.get("component_consumer"))
        or (
            confirmed_backlog
            and not root_confirmed
            and (
                spec.get("component_consumer") is not None
                or spec.get("input_semantics") != "grape-v1"
                or spec.get("definition_failure_of") is not None
            )
        )
        or (path / "cancel").exists()
        or (path / "result.json").exists()
        or failure["stage"] != "definitions"
        or (not routed_backlog and failure["error_code"] != ("timeout" if unknown else "schema"))
        or failure["failure_kind"]
        != ("dispatched_unknown" if unknown else "confirmed_response_or_domain_refusal")
        or failure["protocol_checksum"] != digest(protocol)
        or failure["input_checksum"] != checksum
        or failure["campaign_id"] != path.name
        or failure["run_id"]
        != identity(
            "RUN", "task-research:mm-live:" + digest((path.name, "definition-expansion"))[7:55]
        )
        or failure["deadline"] != protocol["deadline"]
        or failure["previous_attempts"] != protocol["previous_attempts"]
        or failure["definition_revision_saved"]
        or failure["confirmed_proposal_saved"]
        or failure["transport"]["remote_response_unknown"] != (1 if unknown else 0)
        or failure["transport"]["dispatch_unknown"]
        or failure["new_requests"] < 1
    ):
        raise ValueError("definition_successor_parent_binding")
    if history_base is not None:
        historical_definition_owner(history_base, path, spec, protocol, failure)
    return spec, protocol, failure


def historical_definition_owner(base, path, spec, protocol, failure):
    """Read an already claimed immutable Source history through original owners.

    The Source owner predates Planner Control. Its durable failure, journal and
    exact existing successor claim are required instead of a filesystem liveness
    probe. This port never creates a claim, changes Control or permits dispatch.
    Current source/tool qualification is still checked by the calling owner.
    """
    from db.runtime.repository.planner import PlannerRepository
    from db.runtime.repository.research import ResearchRunRepository
    from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments

    target = base.runtime._target
    contexts, sessions = ResearchRunRepository(target), PlannerRepository(target)
    context = contexts.read_research_run(failure["run_id"])
    if (
        context.schema_version != "research-run/2.0.0"
        or context.request.scope.case_code != base.binding["case_code"]
        or context.request.input_checksum != spec["input_checksum"]
        or context.request.task != base.task
        or context.request.task.model_dump(mode="json") != spec["intent"]
    ):
        raise ValueError("definition_history_context_binding")
    docs = RuntimeDocuments(
        repo=base.runtime,
        contexts=contexts,
        sessions=sessions,
        store=base.objects,
        run_id=failure["run_id"],
        root=base.root,
    )
    local_path = base.output / "definition-continuation.json"
    if any(p.is_symlink() for p in (local_path, *local_path.parents)):
        raise ValueError("definition_history_claim_symlink")
    local = json.loads(local_path.read_bytes())
    with docs.session():
        saved = docs.read("definition:confirmed-failure-successor")
        if (
            saved != (local,)
            or local.get("output") != str(base.output)
            or docs.read("multimodal-stage-failure") != (failure,)
            or docs.read("definition:revision") is not None
            or sessions.read(docs.run_id) is not None
        ):
            raise ValueError("definition_history_owner_binding")
        reference = local.get("reference", {})
        proof = local.get("proof", {})
        if (
            reference.get("parent") != str(path)
            or reference.get("entry") != digest(spec)
            or reference.get("protocol") != digest(protocol)
            or reference.get("failure") != digest(failure)
            or proof.get("failure") != digest(failure)
            or proof.get("journal") != digest(RuntimeAgentJournal(docs).read())
        ):
            raise ValueError("definition_history_snapshot_binding")


def definition_successor_policy(
    spec, profile, *, defer=False, root_confirmed=False, routed_backlog=False
):
    """Only the declared dependency-encoding revision, not a silent plan reset."""
    old = json.loads(Path(spec["active_acquisition"]["path"]).read_bytes())
    if digest(old) != spec["active_acquisition"]["checksum"]:
        raise ValueError("definition_successor_profile_changed")
    if routed_backlog:
        policy = old["definition_expansion"]
        if (
            not defer
            or spec["schema_version"] != "multimodal-live-entry/1.29.0"
            or old != profile
            or (policy["schema_version"], policy["prompt_version"])
            not in {
                ("campaign-definition-expansion/1.27.0", "definition-draft-6.3.0"),
                ("campaign-definition-expansion/1.28.0", "definition-draft-6.4.0"),
                ("campaign-definition-expansion/1.29.0", "definition-draft-6.5.0"),
            }
        ):
            raise ValueError("definition_successor_plan_delta")
        return
    if defer:
        policy = old["definition_expansion"]
        allowed = (
            (policy["schema_version"], policy["prompt_version"])
            in (
                {
                    ("campaign-definition-expansion/1.27.0", "definition-draft-6.3.0"),
                    ("campaign-definition-expansion/1.28.0", "definition-draft-6.4.0"),
                    ("campaign-definition-expansion/1.29.0", "definition-draft-6.5.0"),
                }
                if root_confirmed
                else {("campaign-definition-expansion/1.26.0", "definition-draft-6.2.0")}
            )
            if spec["schema_version"] == "multimodal-live-entry/1.26.0"
            else (
                (policy["schema_version"], policy["prompt_version"])
                in {
                    ("campaign-definition-expansion/1.24.0", "definition-draft-6.1.0"),
                    ("campaign-definition-expansion/1.27.0", "definition-draft-6.3.0"),
                }
                if spec["schema_version"] == "multimodal-live-entry/1.23.0"
                else policy["prompt_version"] == "definition-draft-6.3.0"
            )
        )
        if old != profile or not allowed:
            raise ValueError("definition_successor_plan_delta")
        return
    expected = json.loads(json.dumps(old))
    expected["id"] = "glaucoma-structural-context-acquisition-v2"
    expected["definition_expansion"]["schema_version"] = "campaign-definition-expansion/1.27.0"
    expected["definition_expansion"]["prompt_version"] = "definition-draft-6.3.0"
    if (
        old["definition_expansion"]["prompt_version"] != "definition-draft-6.2.0"
        or profile != expected
    ):
        raise ValueError("definition_successor_plan_delta")


class ConfirmedDefinitionContinuation:
    """Same-window full-case successor; original Runtime owns claim and accounting."""

    def __init__(self, campaign, reference, *, read_only=False, history_base=None):
        from packages.object_store import checksum
        from packages.research_runtime.agent_records import RuntimeAgentJournal

        if history_base is not None and (not read_only or history_base is not campaign.base):
            raise ValueError("definition_history_read_only")
        self.path = Path(reference["parent"])
        self.routed_backlog = reference.get("mode") == ROUTED_BACKLOG
        self.root_confirmed = (
            reference.get("mode") == "root_confirmed_schema_backlog_no_source_replay"
        )
        self.confirmed_backlog = (
            reference.get("mode") == "confirmed_schema_backlog_no_source_replay"
        ) or self.root_confirmed
        self.root_backlog = (
            (reference.get("mode") == "root_technical_backlog_no_source_replay")
            or self.root_confirmed
            or self.routed_backlog
        )
        self.defer = (
            self.confirmed_backlog
            or self.root_backlog
            or reference.get("mode") == "technical_backlog_no_source_replay"
        )
        spec, protocol, failure = definition_parent(
            self.path,
            checksum=campaign.base.manifest.checksum,
            intent=campaign.base.task.model_dump(mode="json"),
            defer=self.defer,
            confirmed_backlog=self.confirmed_backlog and not self.root_confirmed,
            root_backlog=self.root_backlog,
            root_confirmed=self.root_confirmed,
            routed_backlog=self.routed_backlog,
            history_base=history_base,
        )
        expected_reference = dict(
            parent=str(self.path),
            entry=digest(spec),
            protocol=digest(protocol),
            failure=digest(failure),
        )
        if self.defer:
            expected_reference["mode"] = (
                ROUTED_BACKLOG
                if self.routed_backlog
                else "root_confirmed_schema_backlog_no_source_replay"
                if self.root_confirmed
                else "root_technical_backlog_no_source_replay"
                if self.root_backlog
                else "confirmed_schema_backlog_no_source_replay"
                if self.confirmed_backlog
                else "technical_backlog_no_source_replay"
            )
        transition = (
            definition_provider_transition(
                spec, campaign.config, getattr(campaign.base, "review_contract", None)
            )
            if self.routed_backlog
            else None
        )
        if transition is not None:
            expected_reference["provider_transition"] = transition
        from .resolved_source_decisions import policy_reference

        merge_policy = policy_reference(reference, routed=self.routed_backlog)
        expected_reference.update(merge_policy)
        public_policy = public_wire_reference(reference, routed=self.routed_backlog)
        expected_reference.update(public_policy)
        public_wire_policy = public_policy.get("public_wire_policy")
        self.resolved_merges = []
        if reference != expected_reference:
            raise ValueError("definition_successor_reference_changed")
        if (
            transition is None and spec["provider"] != campaign.config.model_dump(mode="json")
        ) or spec.get("component_consumer") != getattr(
            campaign.base, "component_consumer_selection", None
        ):
            raise ValueError("definition_successor_provider_or_component")
        if self.routed_backlog:
            window = getattr(campaign.base, "source_fallback_window", None)
            if window is not None:
                clock = check_fallback_window(window, protocol)
                if any(campaign.base.active_start[k] != v for k, v in clock.items()):
                    raise ValueError("definition_fallback_window_clock")
        docs = campaign.documents(failure["run_id"])
        events = RuntimeAgentJournal(docs).read()
        records = [e["record"] for e in events if e.get("phase") == "completed"]
        reserved = [e for e in events if e.get("phase") == "reserved"]
        if (
            docs.read("multimodal-stage-failure") != (failure,)
            or campaign.sessions.read(docs.run_id) is not None
            or docs.read("definition:revision") is not None
            or len(reserved) != len(records)
            or not records
            or any(
                r["role"] != "source"
                or r["error"]
                not in (
                    {None, "schema", "timeout", "transport"}
                    if self.routed_backlog
                    else {None, "schema"}
                    if self.confirmed_backlog
                    else {None, "timeout"}
                    if self.defer
                    else {"schema"}
                )
                or r["simulated"]
                for r in records
            )
            or (
                self.defer
                and not self.routed_backlog
                and not self.confirmed_backlog
                and sum(r["error"] == "timeout" for r in records) != 1
            )
            or (self.confirmed_backlog and records[-1]["error"] != "schema")
            or (
                self.root_backlog
                and not self.root_confirmed
                and not self.routed_backlog
                and records[-1]["error"] != "timeout"
            )
            or (
                self.routed_backlog
                and (
                    records[-1]["error"] != failure["error_code"]
                    or any(r["error"] in {"timeout", "transport"} for r in records[:-1])
                    or sum(r["error"] in {"timeout", "transport"} for r in records)
                    != failure["transport"]["remote_response_unknown"]
                )
            )
        ):
            raise ValueError("definition_successor_original_owner")
        for index, r in enumerate(records):
            raw = docs.read("definition:response:" + r["request_id"])
            if not raw or raw[0]["record"] != r:
                raise ValueError("definition_successor_response_binding")
            text = raw[0]["raw_text"]
            if r["error"] is None:
                from packages.evidence_reuse.definitions import OwnerBoundDefinitionProposal
                from packages.grounding.models import validate
                from packages.llm_provider.provider import strict_json

                parsed = validate(OwnerBoundDefinitionProposal, strict_json(text.encode()))
                expected_output = digest(parsed)
                if self.confirmed_backlog or self.root_backlog:
                    # A completed empty first round followed by a schema refusal
                    # is not an UNKNOWN retry. Only an explicitly resolved empty
                    # proposal may be deferred; never discard an unprocessed draft.
                    decision = docs.read(f"definition:gap:{index}:decision")
                    if parsed.definitions and merge_policy and decision:
                        from .resolved_source_decisions import resolved_merge

                        self.resolved_merges.append(
                            resolved_merge(
                                decision[0],
                                parsed,
                                documents=docs,
                                parent=self.path,
                                protocol=protocol,
                                spec=spec,
                                index=index,
                            )
                        )
                    elif (
                        parsed.definitions
                        or not decision
                        or decision[0]["candidate"]["added"]
                        or decision[0]["candidate"]["replaced"]
                        or (self.root_backlog and decision[0]["candidate"].get("merged"))
                    ):
                        raise ValueError("definition_successor_uncommitted_increment")
            else:
                expected_output = digest(text.encode().hex()) if text else None
            if expected_output != r["output_checksum"] or (
                (
                    not self.defer
                    or self.confirmed_backlog
                    or self.routed_backlog
                    and r["error"] == "schema"
                )
                and not text
            ):
                raise ValueError("definition_successor_response_binding")
        wire_proofs = []
        closed_wire_absences, wire_reservations = [], []
        if self.confirmed_backlog or self.root_backlog:
            # Count original public-network reservations too, not only LLM usage.
            # Never replay an unknown wire or erase an uncommitted definition.
            for operation in failure["operations"]:
                if operation["kind"] == "wire":
                    saved = docs.read(operation["key"] + ":outcome")
                    if (
                        not saved
                        or not saved[0].get("response_checksum")
                        or type(saved[0].get("http_status")) is not int
                        or not 100 <= saved[0]["http_status"] <= 599
                    ):
                        if public_wire_policy is None:
                            raise ValueError("definition_successor_wire_unknown")
                        wire_reservations.extend(
                            closed_wire_absence(docs, operation, saved, protocol)
                        )
                        closed_wire_absences.append(operation["key"])
                    wire_proofs.append((operation["key"] + ":outcome", saved))
            # A current local source pool can legitimately require no new HTTP
            # fetch. Account for the actual wire operations, not a mandatory one.
            if len({key for key, _ in wire_proofs}) != len(wire_proofs):
                raise ValueError("definition_successor_wire_accounting")
        if (
            sum(r["api_requests"] for r in records) != failure["transport"]["dispatched_observed"]
            or failure["new_requests"]
            != failure["transport"]["dispatched_observed"] + len(wire_proofs)
            or any(
                o["kind"]
                not in (
                    {
                        "local_search",
                        "local_pool_reconciliation",
                        "llm",
                        "search_group",
                        "fetch_group",
                        "fetch_order",
                        "wire",
                        "local_index_plan",
                        "local_index_window",
                    }
                    if self.confirmed_backlog or self.root_backlog
                    else {"local_search", "local_pool_reconciliation", "llm"}
                )
                for o in failure["operations"]
            )
        ):
            raise ValueError("definition_successor_request_accounting")
        for target in (*protocol["targets"], None):
            rid = identity(
                "RUN", "task-research:mm-live:" + digest((self.path.name, target or "review"))[7:55]
            )
            try:
                campaign.contexts.read_research_run(rid)
            except ValueError as exc:
                if str(exc) != "research_context_required":
                    raise
            else:
                raise ValueError("definition_successor_acquisition_started")
        self.reference = dict(reference)
        self.attempts = [
            *protocol["previous_attempts"],
            dict(
                directory=str(self.path),
                requests=failure["new_requests"],
                checksum=digest(failure),
                kind="routed_source_failure_deferred"
                if self.routed_backlog
                else "confirmed_source_schema_deferred"
                if self.confirmed_backlog
                else "terminal_source_unknown_deferred"
                if self.defer
                else "confirmed_source_schema_failure",
            ),
        ]
        if len(self.attempts) > 8 or sum(a["requests"] for a in self.attempts) >= 40:
            raise ValueError("definition_successor_budget")
        self.proof = dict(journal=digest(events), failure=digest(failure))
        if merge_policy:
            self.proof["resolved_merges"] = self.resolved_merges
        self.wire_proofs = wire_proofs
        self.wire_reservations = wire_reservations
        if public_wire_policy is not None:
            self.proof.update(
                public_wire_policy=public_wire_policy,
                closed_public_wire_absences=closed_wire_absences,
                wire_reservations=digest(wire_reservations),
            )
        if self.confirmed_backlog or self.root_backlog:
            self.proof["wire_outcomes"] = digest(wire_proofs)
        if self.defer:
            decisions = []
            for index in range(2):
                value = docs.read(f"definition:gap:{index}:decision")
                if value:
                    candidate = value[0]["candidate"]
                    if candidate["added"] or candidate["replaced"]:
                        raise ValueError("definition_successor_uncommitted_increment")
                    decisions.append(value[0])
            self.proof["unadopted_decisions"] = digest(decisions)
        self.docs = docs
        claim = dict(output=str(campaign.output), reference=reference, proof=self.proof)
        if self.routed_backlog and getattr(campaign.base, "source_fallback_window", None):
            claim["successor_window"] = campaign.base.source_fallback_window
        with docs.session():
            old = docs.read("definition:confirmed-failure-successor")
            if old is not None and old != (claim,):
                raise ValueError("definition_successor_already_claimed")
            if old is None:
                if read_only:
                    raise ValueError("definition_successor_claim_missing")
                docs.write("definition:confirmed-failure-successor", (claim,))
        self.claim = claim
        self.files = {
            n: checksum((self.path / n).read_bytes())
            for n in ("entry.json", "protocol.json", "stage-failure.json")
        }
        if read_only:
            if json.loads((campaign.output / "definition-continuation.json").read_bytes()) != claim:
                raise ValueError("definition_successor_local_claim_changed")
        else:
            write_once(campaign.output / "definition-continuation.json", claim)

    def check_attempts(self):
        from packages.object_store import checksum
        from packages.research_runtime.agent_records import RuntimeAgentJournal

        for resolved in self.resolved_merges:
            for receipt in resolved["original_acceptances"]:
                saved = self.docs.read("definition:" + receipt["key"])
                if not saved or digest(saved[0]) != receipt["checksum"]:
                    raise ValueError("definition_successor_history_changed")
        if self.defer:
            decisions = [
                value[0]
                for index in range(2)
                if (value := self.docs.read(f"definition:gap:{index}:decision"))
            ]
            if digest(decisions) != self.proof["unadopted_decisions"]:
                raise ValueError("definition_successor_history_changed")
        if (
            any(self.docs.read(key) != value for key, value in self.wire_proofs)
            or any(self.docs.read(key) != value for key, value in self.wire_reservations)
            or any(checksum((self.path / n).read_bytes()) != sha for n, sha in self.files.items())
            or (self.path / "cancel").exists()
            or digest(RuntimeAgentJournal(self.docs).read()) != self.proof["journal"]
            or digest(self.docs.read("multimodal-stage-failure")[0]) != self.proof["failure"]
            or self.docs.read("definition:confirmed-failure-successor") != (self.claim,)
        ):
            raise ValueError("definition_successor_history_changed")
        return self.attempts

    def defer_expansion(self, expansion, until):
        """Original owner commits baseline + explicit technical backlog, no Source call."""
        from .gap_expansion import empty, finish

        if not self.defer:
            raise ValueError("definition_successor_deferral_not_authorized")
        self.check_attempts()
        expansion.control()
        candidate = empty(expansion)
        for resolved in self.resolved_merges:
            if digest(candidate["rows"]) != resolved["baseline_requirements_checksum"]:
                raise ValueError("definition_resolved_merge_current_baseline_changed")
            candidate["backlog"].append(
                dict(
                    reason="original_resolved_definition_merge_retained",
                    original_resolved_decision=resolved,
                )
            )
        candidate["backlog"].append(
            dict(
                reason="source_failure_quarantined_not_replayed"
                if self.routed_backlog
                else "source_schema_confirmed_not_replayed"
                if self.confirmed_backlog
                else "source_timeout_unknown_not_replayed",
                original_failure=self.reference,
                new_source_requests=0,
                medical_no_addition_claimed=False,
            )
        )
        summary = dict(
            schema_version="definition-gap-loop/1.0.0",
            binding=expansion.binding,
            attempts=[],
            stop_reason="terminal_source_technical_backlog",
            next_action="continue_existing_qualified_requirements",
            shared_budget=True,
            reset_deadline=False,
            **(
                dict(successor_window=self.claim.get("successor_window"))
                if self.routed_backlog
                else {}
            ),
            clinical_success_claimed=False,
            mandatory_increment=False,
            terminal_source_deferral=self.reference,
        )
        if "resolved_merges" in self.proof:
            summary["resolved_nonadopted_definitions"] = self.resolved_merges
        return finish(expansion, candidate, summary, until)


def load_parent(path, *, checksum, intent):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("discovery_continuation_symlink")
    spec = json.loads((path / "entry.json").read_bytes())
    protocol = json.loads((path / "protocol.json").read_bytes())
    if (
        spec["schema_version"] != "multimodal-live-entry/1.11.0"
        or spec["input_checksum"] != checksum
        or spec["intent"] != intent
        or not (path / "cancel").exists()
        or (path / "result.json").exists()
        or protocol["input"] != checksum
        or protocol["task"] != intent["checksum"]
        or protocol["implementation"] != spec["implementation"]
        or protocol["previous_attempts"]
        or spec.get("active_continuation_of")
    ):
        raise ValueError("discovery_continuation_parent_binding")
    with (path / "lock").open("r") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return spec, protocol


class DiscoveryContinuation:
    def __init__(self, campaign, path):
        from packages.research_runtime.agent_records import RuntimeAgentJournal

        self.campaign, self.path = campaign, Path(path).absolute()
        self.spec, self.protocol = load_parent(
            self.path,
            checksum=campaign.base.manifest.checksum,
            intent=campaign.base.task.model_dump(mode="json"),
        )
        if self.spec["provider"] != campaign.config.model_dump(mode="json"):
            raise ValueError("discovery_continuation_provider_changed")
        rid = identity(
            "RUN", "task-research:mm-live:" + digest((self.path.name, "definition-expansion"))[7:55]
        )
        docs = campaign.documents(rid)
        events = RuntimeAgentJournal(docs).read()
        if any(e.get("phase") in {"reserved", "completed"} for e in events):
            raise ValueError("discovery_continuation_agent_already_started")
        if campaign.sessions.read(rid) is not None or docs.read("definition:revision"):
            raise ValueError("discovery_continuation_control_or_revision")
        ops = []
        for n in range(12):
            saved = docs.read(f"definition:operation:{n}")
            if saved:
                ops.append(saved[0])
        if not ops or any(
            o["kind"] not in {"local_search", "search_group", "local_repair", "wire"} for o in ops
        ):
            raise ValueError("discovery_continuation_operation_scope")
        wires = sum(o["kind"] == "wire" for o in ops)
        if not 1 <= wires < 40:
            raise ValueError("discovery_continuation_wire_budget")
        for t in (*self.protocol["targets"], None):
            other = identity(
                "RUN", "task-research:mm-live:" + digest((self.path.name, t or "review"))[7:55]
            )
            try:
                campaign.contexts.read_research_run(other)
            except ValueError as exc:
                if str(exc) != "research_context_required":
                    raise
            else:
                raise ValueError("discovery_continuation_acquisition_started")
            if campaign.sessions.read(other) is not None:
                raise ValueError("discovery_continuation_control_present")
        old_profile = json.loads(Path(self.spec["active_acquisition"]["path"]).read_bytes())
        if digest(old_profile) != self.spec["active_acquisition"]["checksum"]:
            raise ValueError("discovery_continuation_profile_changed")
        indices = {int(o["key"].rsplit(":", 1)[1]) for o in ops if o["kind"] == "local_search"}
        self.failed_queries = tuple(
            old_profile["definition_expansion"]["public_queries"][i] for i in sorted(indices)
        )
        self.reference = dict(
            parent=str(self.path),
            run_id=rid,
            operations_checksum=digest(ops),
            requests=wires,
            unknown_requests_not_replayed=True,
            deadline=self.protocol["deadline"],
        )
        # An append-only successor claim prevents two new directories each spending
        # the same remainder. No old command, response, score or freeze is changed.
        claim = dict(campaign_id=campaign.campaign_id, reference=self.reference)
        with docs.session():
            existing = docs.read("definition:successor")
            if existing is not None and existing != (claim,):
                raise ValueError("discovery_continuation_successor_already_claimed")
            if existing is None:
                docs.write("definition:successor", (claim,))
        self.attempts = [
            dict(
                directory=str(self.path),
                requests=wires,
                kind="stopped_discovery_no_replay",
                checksum=digest(self.reference),
            )
        ]
        write_once(campaign.output / "discovery-continuation.json", self.reference)

    def check_attempts(self):
        spec, protocol = load_parent(
            self.path, checksum=self.spec["input_checksum"], intent=self.spec["intent"]
        )
        if spec != self.spec or protocol != self.protocol:
            raise ValueError("discovery_continuation_parent_changed")
        return self.attempts


def reuse_native(base, path):
    from .native_fundus import recheck

    spec, protocol = load_parent(
        path, checksum=base.manifest.checksum, intent=base.task.model_dump(mode="json")
    )
    start = json.loads((Path(path) / "active-start.json").read_bytes())
    if (
        start["deadline"] != protocol["deadline"]
        or start["input_checksum"] != base.manifest.checksum
    ):
        raise ValueError("discovery_continuation_deadline")
    write_once(base.output / "active-start.json", start)
    native = json.loads((Path(path) / "native-fundus/accepted.json").read_bytes())
    recheck(base, native)

    def runner(current, requirement, *, before_dispatch):
        if (
            requirement["id"] != native["plan"]["target"]
            or requirement["component"] != native["plan"]["component"]
        ):
            raise ValueError("discovery_continuation_native_binding")
        recheck(current, native)
        write_once(
            base.output / "native-reuse.json",
            dict(parent=str(Path(path).absolute()), checksum=digest(native), new_model_runs=0),
        )
        return native

    return runner


def reuse_failed_native(
    base,
    reference,
    *,
    read_only=False,
    following_provider=None,
    following_review_contract=None,
    history_base=None,
):
    """Only exact confirmed artifacts from the bound Source-failed parent.

    This is a read/revalidation port, not permission to launch a tool. The
    original ConfirmedDefinitionContinuation still owns the single successor
    claim, shared budget and dispatch control before any subsequent API work.
    """
    from .native_fundus import recheck

    if history_base is not None and (not read_only or history_base is not base):
        raise ValueError("definition_history_read_only")
    routed = reference.get("mode") == ROUTED_BACKLOG
    if reference.get("mode") not in {
        "root_confirmed_schema_backlog_no_source_replay",
        ROUTED_BACKLOG,
    }:
        raise ValueError("root_backlog_native_reuse_required")
    parent = Path(reference["parent"]).absolute()

    def check_parent():
        spec, protocol, failure = definition_parent(
            parent,
            checksum=base.manifest.checksum,
            intent=base.task.model_dump(mode="json"),
            defer=True,
            root_backlog=True,
            root_confirmed=not routed,
            routed_backlog=routed,
            history_base=history_base,
        )
        expected = dict(
            parent=str(parent),
            entry=digest(spec),
            protocol=digest(protocol),
            failure=digest(failure),
            mode=ROUTED_BACKLOG if routed else "root_confirmed_schema_backlog_no_source_replay",
        )
        from .resolved_source_decisions import policy_reference

        expected.update(policy_reference(reference, routed=routed))
        expected.update(public_wire_reference(reference, routed=routed))
        if "provider_transition" in reference:
            if not routed or following_provider is None:
                raise ValueError("definition_successor_native_provider_binding_required")
            transition = definition_provider_transition(
                spec, following_provider, following_review_contract
            )
            if transition is not None:
                expected["provider_transition"] = transition
        if reference != expected:
            raise ValueError("definition_successor_native_parent_changed")

    check_parent()

    def runner(current, requirement, *, before_dispatch, component_scoped=False):
        if current is not base:
            raise ValueError("definition_successor_native_owner_changed")
        check_parent()
        directory = parent / "native-fundus"
        suffix = digest((requirement["id"], requirement["component"]))[7:39]
        if component_scoped:
            directory /= suffix
        path = directory / "accepted.json"
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("definition_successor_native_symlink")
        native = json.loads(path.read_bytes())
        if (
            native.get("schema_version")
            != (
                "native-fundus-acceptance/1.1.0"
                if component_scoped
                else "native-fundus-acceptance/1.0.0"
            )
            or native["plan"]["target"] != requirement["id"]
            or native["plan"]["component"] != requirement["component"]
        ):
            raise ValueError("definition_successor_native_binding")
        recheck(current, native)
        receipt = dict(
            schema_version="source-failure-native-reuse/1.0.0",
            original_failure=reference,
            original_native_checksum=digest(native),
            bundle_key=native["bundle_key"],
            target=requirement["id"],
            component=requirement["component"],
            new_model_runs=0,
        )
        output = base.output / ("native-failure-reuse-" + suffix + ".json")
        if read_only:
            if json.loads(output.read_bytes()) != receipt:
                raise ValueError("definition_successor_native_receipt_changed")
        else:
            write_once(output, receipt)
        return native

    return runner
