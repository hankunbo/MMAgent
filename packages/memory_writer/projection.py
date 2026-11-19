"""Pure projection of already saved facts. No owner calls, scoring or history inference."""

from packages.grounding.models import canonical, digest
from packages.memory.models import Dependency, Record, validate_relationships
from packages.memory.writer_models import (
    MAX_EVENTS,
    SEGMENT_EVENTS,
    EpisodeData,
    FieldRef,
    Segment,
    Snapshot,
    Step,
    Terminal,
    WriterRecord,
)
from packages.planner_control.models import Graph, Reference


class NotReady(ValueError):
    pass


def missing(reason):
    return FieldRef(missing_reason=reason)


def locate(value, path):
    for part in path.strip("/").split("/"):
        value = value[int(part)] if isinstance(value, (list, tuple)) else value[part]
    return value


def ref(snapshot, source, path):
    return FieldRef(source=source.checksum, path=path, checksum=digest(locate(snapshot, path)))


def terminal(snapshot, source):
    r, c = snapshot["runtime"], snapshot["control"]
    report = r["report"]
    if any(a["status"] != "CONFIRMED" for a in r["attempts"]):
        raise NotReady("memory_execution_unconfirmed")
    if r["lifecycle_phase"] == "UNKNOWN" or (c and c["pending"] is not None):
        raise NotReady("memory_execution_unconfirmed")
    if c and c["phase"] != "stopped":
        raise NotReady("memory_control_not_terminal")
    if r["lifecycle_phase"] not in {None, "FINISHED"}:
        raise NotReady("memory_lifecycle_not_terminal")
    if r["lifecycle_phase"] == "FINISHED" and report is None:
        raise NotReady("memory_terminal_report_missing")
    if report:
        if not report["terminal"] or report["technical_status"] == "UNKNOWN_JOB":
            raise NotReady("memory_execution_unconfirmed")
        if report["run_status"] != r["runtime_status"] or r["runtime_status"] not in {
            "COMPLETED_SUPPORTED",
            "COMPLETED_REFUTED",
            "COMPLETED_INSUFFICIENT_EVIDENCE",
            "COMPLETED_UNCERTAIN",
            "FAILED_TECHNICAL",
            "STOPPED_UNSAFE",
        }:
            raise ValueError("memory_terminal_inconsistent")
        scope = "preparation_rejection" if r["preparation_rejected"] else "runtime_lifecycle"
        if scope == "preparation_rejection" and (c or r["attempts"]):
            raise ValueError("memory_terminal_inconsistent")
        reason = report["reason"]
        evidence = ref(snapshot, source, "/runtime/report")
    elif c:
        # Planner owns Control termination, not Runtime lifecycle completion.
        if r["runtime_status"] != "CREATED":
            raise NotReady("memory_runtime_terminal_evidence_missing")
        scope, reason = "planner_control", c["reason"]
        evidence = ref(snapshot, source, "/control_report")
    else:
        raise NotReady("memory_terminal_evidence_missing")
    return Terminal(
        scope=scope,
        runtime_status=r["runtime_status"],
        control_phase=c["phase"] if c else None,
        lifecycle_phase=r["lifecycle_phase"],
        technical_status=report["technical_status"] if report else None,
        reason=reason or "owner_reason_not_recorded",
        cancelled=c["cancelled"]
        if c
        else bool(report and report["reason"] in {"cancelled_before_launch", "tool_cancelled"}),
        evidence=evidence,
    )


def step(snapshot, source, index):
    e = snapshot["control"]["events"][index]
    p = f"/control/events/{index}"
    selection = e["outcome"].get("decision")
    refs = e["outcome"]["refs"]
    review = snapshot.get("reviews", {}).get(str(index))
    decision = review["decision"] if review else None
    acceptance = next(
        (i for i, v in enumerate(refs) if v["version"] == "research-acceptance/1.0.0"), None
    )
    return Step(
        index=index,
        command_key=e["command"]["key"],
        original_revision=e["command"]["expected_revision"],
        action=e["command"]["action"],
        before=e["before"],
        after=e["after"],
        event=ref(snapshot, source, p),
        state=ref(snapshot, source, p + "/outcome/decision/action_set/state_checksum")
        if selection
        else missing("historical_state_not_recorded"),
        legal_actions=ref(snapshot, source, p + "/outcome/decision/action_set")
        if selection
        else missing("historical_legal_actions_not_recorded"),
        selection=ref(snapshot, source, p + "/outcome/decision")
        if selection
        else missing("selection_decision_not_recorded"),
        execution=ref(snapshot, source, p + "/outcome")
        if e["command"]["action"] in {"execute", "recover", "verify"}
        else missing("not_execution_event"),
        observation=ref(snapshot, source, p + "/outcome/refs")
        if refs
        else missing("no_observation_reference"),
        feedback=ref(snapshot, source, p + "/outcome"),
        expert_advice=ref(snapshot, source, f"/reviews/{index}/decision/suggestion")
        if decision and decision["suggestion"] is not None
        else missing("expert_advice_not_recorded"),
        expert_decision=ref(snapshot, source, f"/reviews/{index}/decision")
        if decision
        else missing("expert_decision_not_recorded"),
        review_result=ref(snapshot, source, f"/reviews/{index}/result")
        if review
        else missing("review_result_not_recorded"),
        controlled_decision=ref(snapshot, source, p + "/outcome"),
        domain_acceptance=ref(snapshot, source, f"/reviews/{index}/result/domain_status")
        if review
        else ref(snapshot, source, p + f"/outcome/refs/{acceptance}")
        if acceptance is not None
        else missing("domain_receipt_not_in_event"),
        versions=ref(snapshot, source, "/control/binding/locks"),
        status=e["outcome"]["status"],
        reason=e["outcome"]["reason"],
    )


def project(snapshot, source, *, request_key, fact_sources=(), supplements=()):
    """Bound before hashing; 16 events/segment, <=256 events and <=16 supplements."""
    t = terminal(snapshot, source)
    c = snapshot["control"]
    events = c["events"] if c else []
    if len(events) > MAX_EVENTS or len(supplements) > 16:
        raise ValueError("memory_trajectory_bound")
    if len({e["command"]["key"] for e in events}) != len(events) or any(
        a["command"]["expected_revision"] >= b["command"]["expected_revision"]
        or a["after"] != b["before"]
        for a, b in zip(events, events[1:], strict=False)
    ):
        raise ValueError("memory_event_order")
    prefix = "writer:" + digest([source.run_id, request_key])[7:]
    common = dict(
        case_code=source.case_code,
        run_id=source.run_id,
        applicability=source.applicability,
        nature="source_fact",
    )
    records, segments, graphs, supplemental = [], [], [], []
    for start in range(0, len(events), SEGMENT_EVENTS):
        key = prefix + ":segment:" + str(start // SEGMENT_EVENTS)
        segments.append(key)
        records.append(
            WriterRecord(
                **common,
                key=key,
                kind="trajectory_step",
                sources=(source,),
                detail=Segment(
                    start=start,
                    steps=tuple(
                        step(snapshot, source, i)
                        for i in range(start, min(start + SEGMENT_EVENTS, len(events)))
                    ),
                ),
            )
        )
    for role, path in (
        ("initial_historical", "/control/binding/graph"),
        ("final_current", "/control/graph"),
    ):
        if c:
            key = prefix + ":" + role
            graphs.append(key)
            records.append(
                WriterRecord(
                    **common,
                    key=key,
                    kind="graph_snapshot",
                    sources=(source, *fact_sources),
                    detail=Snapshot(
                        role=role,
                        graph=Graph.model_validate(locate(snapshot, path)),
                        evidence=ref(snapshot, source, path),
                    ),
                )
            )
    for n, s in enumerate(supplements):
        kind = {"simulation": "counterfactual_pair", "recovery": "recovery_experience"}.get(s.owner)
        if kind is None:
            raise ValueError("memory_supplement_kind")
        key = prefix + ":source:" + str(n)
        supplemental.append(key)
        records.append(Record(**{**common, "nature": s.nature}, key=key, kind=kind, sources=(s,)))
    score = next((s for s in fact_sources if s.kind == "scores"), None)
    detail = EpisodeData(
        request_key=request_key,
        input_checksum=digest(dict(snapshot=snapshot, facts=fact_sources, supplements=supplements)),
        terminal=t,
        initial_graph=ref(snapshot, source, "/control/binding/graph")
        if c
        else missing("no_graph_before_failure"),
        final_graph=ref(snapshot, source, "/control/graph")
        if c
        else missing("no_research_graph_recorded"),
        scores=FieldRef(source=score.checksum, path="/value", checksum=score.checksum)
        if score
        else missing("no_accepted_research_scores"),
        versions=tuple(Reference.model_validate(v) for v in c["binding"]["locks"]) if c else (),
        runtime_versions=ref(snapshot, source, "/runtime/versions"),
        execution_attempts=ref(snapshot, source, "/runtime/attempts"),
        runtime_report=ref(snapshot, source, "/runtime/report")
        if snapshot["runtime"]["report"]
        else missing("runtime_lifecycle_report_not_recorded"),
        trajectory_count=len(events),
        trajectory_missing_reason=None if events else "no_planner_events_recorded",
        segments=tuple(segments),
        graph_members=tuple(graphs),
        supplemental_members=tuple(supplemental),
    )
    records.append(
        WriterRecord(
            **common,
            key=prefix + ":episode",
            kind="episode",
            sources=(source, *fact_sources, *supplements),
            parents=tuple(
                Dependency(key=r.key, checksum=digest(r), relation="episode_member")
                for r in records
            ),
            detail=detail,
        )
    )
    validate_aggregate(request_key, tuple(records))
    return tuple(records)


def validate_aggregate(key, records):
    if not records or len(records) > 35 or not isinstance(records[-1], WriterRecord):
        raise ValueError("memory_aggregate_shape")
    episode = records[-1]
    if not isinstance(episode.detail, EpisodeData) or episode.detail.request_key != key:
        raise ValueError("memory_request_binding")
    parents = {r.key: r for r in records[:-1]}
    if len(parents) != len(records) - 1 or set(parents) != {p.key for p in episode.parents}:
        raise ValueError("memory_episode_members")
    if any(r.parents for r in records[:-1]):
        raise ValueError("memory_member_parents")
    validate_relationships(episode, parents)
    steps = []
    for k in episode.detail.segments:
        member = parents[k]
        if not isinstance(member, WriterRecord) or not isinstance(member.detail, Segment):
            raise ValueError("memory_segment_kind")
        steps.extend(member.detail.steps)
    if [s.index for s in steps] != list(range(episode.detail.trajectory_count)):
        raise ValueError("memory_event_order")
    if len({s.command_key for s in steps}) != len(steps):
        raise ValueError("memory_event_duplicate")
    if any(len(canonical(r)) > 48000 for r in records):
        # Leave room for PostgreSQL jsonb textual whitespace within its original 64KiB cap.
        raise ValueError("memory_record_bound")
