"""Additive reports and 08 restricted review over original Runtime/Audit storage."""

import html
import json

from agents.expert_agent import ResearchReview
from agents.expert_agent.models import ExpertDecision, ReviewFacts, read_result
from packages.evidence_details.current_rejection import CurrentResearchRejected, current_rejection
from packages.grounding.models import digest, validate
from packages.llm_provider.models import Scope
from packages.planner_control.models import Graph
from packages.rule_engine import SnapshotBinding

from .models import PathGraphReview, PathGraphSummary


def checked_summary(value):
    summary = validate(PathGraphSummary, value)
    raw = summary.model_dump(mode="json", exclude={"checksum"})
    if digest(raw) != summary.checksum:
        raise ValueError("path_graph_summary_checksum")
    if (
        summary.original_report is not None
        and digest(summary.original_report) != summary.original_report_checksum
    ):
        raise ValueError("path_graph_report_binding")
    if (summary.domain == "research") != (summary.purpose == "research_offline"):
        raise ValueError("path_graph_domain_dispatch")
    if summary.domain == "research":
        from packages.evidence_details.acceptance_models import DomainState
        from packages.evidence_scoring.api import validate as validate_scores

        from .research import project_research

        parent = validate(DomainState, summary.binding["parent_state"])
        report = summary.original_report
        if report is None:
            raise ValueError("path_graph_research_report_required")
        validate_scores(
            dict(state=parent, candidate=report["records"]["candidate"]), report["scores"]
        )
        rebuilt = project_research(parent, report, summary.binding["current_scope"])
        if rebuilt != summary:
            raise ValueError("path_graph_projection_binding")
    return summary


def render_path_graph_html(value):
    summary = checked_summary(value)
    body = html.escape(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
    )
    return (
        '<!doctype html><html lang="zh"><meta charset="utf-8">'
        "<title>路径与整图验证</title><h1>研究离线路径与整图验证</h1>"
        "<p>非临床结论；未知不等于阴性；下一动作须核当前Control。</p>"
        f"<p>摘要：{summary.checksum}</p><pre>{body}</pre></html>"
    )


class PathGraphValidation:
    def __init__(self, *, reader, documents):
        self.reader, self.documents = reader, documents

    def _validate_storage(self, summary, documents=None):
        """Ask the destination owner, including on historical reads (no source gate)."""
        documents = self.documents if documents is None else documents
        if summary.domain == "formal_synthetic":
            documents.validate_snapshot(
                case_code=summary.case_code,
                run_id=summary.run_id,
                snapshot=SnapshotBinding(**summary.graph),
            )
            return
        parent = validate(Scope, summary.binding["parent_state"]["material"]["scope"])
        current = validate(Scope, summary.binding["current_scope"])
        graph = Graph(
            **summary.graph, kind="research_artifact", parent=summary.binding["parent_graph"]
        )
        if (
            (summary.case_code, summary.run_id) != (parent.case_code, parent.run_id)
            or (current.case_code, current.run_id) != (parent.case_code, parent.run_id)
            or current.graph not in (parent.graph, graph)
        ):
            raise ValueError("path_graph_storage_scope")
        # Both the accepted object and its root/parent must belong to this writer.
        for scope in (parent, current, parent.model_copy(update={"graph": graph})):
            documents.validate_scope(scope)

    def inspect(self, *, key):
        """Read-check-write-read using the existing immutable bundle transaction."""
        try:
            with self.documents.session():
                summary = checked_summary(self.reader())
                self._validate_storage(summary)
                value = summary.model_dump(mode="json")
                rendered = dict(
                    schema_version="path-graph-html/1.0.0",
                    summary_checksum=summary.checksum,
                    html=render_path_graph_html(summary),
                )
                prior = self.documents.read("path-graph:" + key)
                if prior is not None and prior != (value, rendered):
                    raise ValueError("path_graph_idempotency_conflict")
                if checked_summary(self.reader()) != summary:
                    raise CurrentResearchRejected("path_graph_read_changed")
                self._validate_storage(summary)
                self.documents.write("path-graph:" + key, (value, rendered))
                stored = self.documents.read("path-graph:" + key)
                if stored != (value, rendered):
                    raise ValueError("path_graph_persistence_binding")
                if checked_summary(self.reader()) != summary:
                    raise CurrentResearchRejected("path_graph_read_changed")
                return value
        except Exception as error:
            reason = (
                "source_revalidation_failed"
                if current_rejection(error)
                else "object_or_transaction_failure"
            )
            self.documents.failure(key, digest(key), reason)
            raise

    def read(self, *, key, historical=False):
        with self.documents.session():
            saved = self.documents.read("path-graph:" + key)
            if saved is None or len(saved) != 2:
                raise ValueError("path_graph_report_missing")
            summary = checked_summary(saved[0])
            self._validate_storage(summary)
            if saved[1] != dict(
                schema_version="path-graph-html/1.0.0",
                summary_checksum=summary.checksum,
                html=render_path_graph_html(summary),
            ):
                raise ValueError("path_graph_html_binding")
            if not historical and checked_summary(self.reader()) != summary:
                raise CurrentResearchRejected("path_graph_read_changed")
            return saved[0]

    def review(self, *, key, review, event, proposed_action_id, cancel=None):
        """Only role=None advisory review. Original 08 controls requests and recovery.

        Caller supplies the already-persisted summary key and the original pending
        Planner event; no second run, new candidate, tool dispatch or graph commit.
        """
        if (
            event.role != "review_control"
            or event.version != "planner-research-review/1.0.0"
            or event.code != "review:" + event.checksum[7:]
        ):
            raise ValueError("review_control_binding")
        # Validate the original event/run before observing a caller's signal.
        review.context.control(event)
        try:
            review.context.control(event, cancel)
            record = self._review(
                key=key,
                review=review,
                event=event,
                proposed_action_id=proposed_action_id,
                cancel=cancel,
            )
            # The original result and outer bundle are now recoverable. Observe
            # cancellation before the original immutable terminal checkpoint.
            review.context.control(event, cancel)
            review.context.checkpoint(event, record["result"], cancel=cancel)
        except BaseException as error:
            try:
                review.context.control(event, cancel)
            except Exception as control_error:
                # Keep the I/O failure primary and the unconfirmed cleanup as its cause.
                raise error from control_error
            raise
        return record

    def _review(self, *, key, review, event, proposed_action_id, cancel):
        summary = checked_summary(self.read(key=key, historical=True))
        if summary.domain != "research":
            raise ValueError("formal_expert_requires_original_v1_port")
        self._validate_storage(summary, review.acceptance.documents)
        context = PathReviewContext(review.context, self.reader, summary)
        advisory = ResearchReview(
            acceptance=review.acceptance,
            ports=review.ports,
            context=context,
            protocol=review.protocol,
        )
        binding_key = event.code + ":path-graph-binding"
        if advisory.store.read(event.code + ":result") and not advisory.store.read(binding_key):
            raise ValueError("path_graph_review_event_already_used")
        advisory.store.claim(
            binding_key,
            dict(
                summary_checksum=summary.checksum,
                event=event.model_dump(mode="json"),
                proposed_action_id=proposed_action_id,
            ),
        )
        result = advisory.run(event=event, proposed_action_id=proposed_action_id, cancel=cancel)
        decision = advisory.store.read(event.code + ":decision")
        record = dict(
            schema_version="path-graph-review/1.0.0",
            summary_checksum=summary.checksum,
            summary=summary.model_dump(mode="json"),
            event=event.model_dump(mode="json"),
            result=result,
            expert_decision=decision[0] if decision else None,
            disposition="candidate_only_no_tool_dispatch",
        )
        record["checksum"] = digest(record)
        rendered = dict(
            schema_version="path-graph-review-html/1.0.0",
            report_checksum=record["checksum"],
            html=render_review_html(record),
        )
        storage_key = "path-graph-review:" + event.code
        try:
            with self.documents.session():
                self._validate_storage(summary)
                self.documents.write(storage_key, (record, rendered))
                if self.documents.read(storage_key) != (record, rendered):
                    raise ValueError("path_graph_review_report_binding")
        except Exception:
            self.documents.failure(storage_key, digest(record), "object_or_transaction_failure")
            raise
        return record


class PathReviewContext:
    """Re-read the original context and 09B sources on every 08 recheck."""

    def __init__(self, original, reader, summary):
        self.original, self.reader, self.summary = original, reader, summary

    def read(self, event):
        frame = self.original.read(event)
        try:
            current = checked_summary(self.reader())
        except ValueError as error:
            reason = current_rejection(error)
            if reason is None:
                raise
            raise CurrentResearchRejected(reason) from error
        if (
            current != self.summary
            or frame.scope.model_dump(mode="json") != current.binding["current_scope"]
        ):
            raise CurrentResearchRejected("path_graph_review_scope_changed")
        facts = validate(ReviewFacts, current.review_facts)
        old = frame.facts
        merged = ReviewFacts(
            conflict=old.conflict or facts.conflict,
            uncertain=old.uncertain or facts.uncertain,
            high_risk=old.high_risk or facts.high_risk,
            required=old.required or facts.required,
            no_progress=max(old.no_progress, facts.no_progress),
        )
        return frame.model_copy(update={"facts": merged})

    def control(self, event, cancel=None):
        return self.original.control(event, cancel)

    def counters(self):
        return self.original.counters()

    def checkpoint(self, event, result, *, cancel=None):
        # 08 already saved the result. Keep the original pending review until
        # 09B's extra I/O finishes; otherwise STOPPED cannot record a late signal.
        # The outer caller invokes the original checkpoint with this exact result.
        return self.original.control(event, cancel)

    def install(self, event, *, parent, successor):
        raise ValueError("path_graph_advice_cannot_install")


def render_review_html(record):
    """Advisory snapshot, including a preserved decision when current Control stopped."""
    if record.get("schema_version") != "path-graph-review/1.0.0":
        raise ValueError("path_graph_review_version")
    validate(PathGraphReview, record)
    raw = {k: v for k, v in record.items() if k != "checksum"}
    summary = checked_summary(record["summary"])
    result = read_result(record["result"])
    if (
        digest(raw) != record["checksum"]
        or summary.checksum != record["summary_checksum"]
        or result["event"] != record["event"]
        or result["run_id"] != summary.run_id
    ):
        raise ValueError("path_graph_review_report_binding")
    decision = record["expert_decision"]
    if decision is not None:
        decision = validate(ExpertDecision, decision)
        if (
            decision.event.model_dump(mode="json") != record["event"]
            or decision.scope.model_dump(mode="json") != summary.binding["current_scope"]
            or (
                result["decision_checksum"] is not None
                and result["decision_checksum"] != digest(decision)
            )
        ):
            raise ValueError("path_graph_expert_report_binding")
    elif result["decision_checksum"] is not None:
        raise ValueError("path_graph_expert_decision_missing")
    body = html.escape(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2))
    return (
        '<!doctype html><html lang="zh"><meta charset="utf-8">'
        "<title>路径与整图审查记录</title><h1>研究离线验证与受限Expert</h1>"
        "<p>建议、采纳理由和停止事实；不是下一动作授权或临床结论。</p>"
        f"<p>摘要：{record['checksum']}</p><pre>{body}</pre></html>"
    )
