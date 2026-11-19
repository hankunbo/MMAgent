"""Versioned adapters for existing report/graph owners. No reconstruction by API."""

import html
import json

from apps.api.errors import Fault
from packages.grounding.models import digest, validate
from packages.object_store import checksum


class ConfirmedExecutionReader:
    """Installation binds one original 06 plan/key; HTTP may only read its facts."""

    def __init__(self, executor, plan, key):
        self.executor, self.plan, self.key = executor, plan, key

    def read(self, run_id):
        from packages.tool_execution.models import WorkflowResult

        if self.plan.document.run_id != run_id:
            raise ValueError("execution_reader_scope")
        record = self.executor.recover(run_id, self.key)
        if record is None:
            raise Fault("DATA_NOT_READY")
        if record["status"] != "CONFIRMED" or record["result"] is None:
            raise Fault("COMMIT_UNKNOWN", commit_state="unconfirmed")
        return self.executor.confirmed_result(
            self.plan, self.key, validate(WorkflowResult, record["result"])
        )


class ResearchReportReader:
    def __init__(self, runtime, scores):
        self.runtime, self.scores = runtime, scores

    def read(self, run_id):
        state = self.runtime.lifecycle(run_id)
        if state is None or state["report"] is None:
            raise Fault("DATA_NOT_READY")
        report = state["report"]
        if report.get("version") != "research-case-report/1.0.0":
            raise Fault("VERSION_UNSUPPORTED")
        context = self.runtime.read_research_run(run_id)
        if state["binding"]["context_checksum"] != context.checksum or report["run_id"] != run_id:
            raise ValueError("research_report_scope")
        ledger = self.scores.read(run_id, state["binding"]["score_key"])
        if report["score_checksum"] != ledger.checksum or report["checksum"] != digest(
            {k: v for k, v in report.items() if k != "checksum"}
        ):
            raise ValueError("research_report_source_binding")
        return report


class SavedGraphReader:
    """All versions supported by the original saved-view owner, selected explicitly."""

    def __init__(self, repository, objects):
        self.repository, self.objects = repository, objects

    def read(self, case_code, asset_code, version):
        from apps.glaucoma_case_storage import read_view

        case = self.repository.get_case_bundle(case_code)
        a = next((a for a in case.artifacts if a.asset_code == asset_code), None)
        if a is None:
            raise Fault("OBJECT_NOT_FOUND")
        if a.scope.case_code != case_code or a.artifact_type_code != "ARTIFACT:CASE_EVIDENCE_GRAPH":
            raise Fault("SCOPE_FORBIDDEN")
        raw = self.objects.read_bounded(a.uri, 4194304)
        if checksum(raw) != a.checksum:
            raise ValueError("graph_object_checksum")
        doc = json.loads(raw)
        if doc.get("schema") != version:
            raise Fault("VERSION_CONFLICT")
        view = read_view(case_code, asset_code, repo=self.repository, store=self.objects)
        return dict(
            graph=doc,
            view=view,
            source_checksum=a.checksum,
            records=view.get("evidence_records"),
            scores=next(
                (
                    view[k]
                    for k in (
                        "current_scoring",
                        "interpretation_scoring",
                        "participation_scoring",
                        "continuous_scoring",
                        "research_scoring",
                    )
                    if k in view
                ),
                None,
            ),
        )


class AcceptedGraphReader:
    """08/09 saved graph 2.0, records 2.0/2.1 and scores 6.0/6.1 retain own versions."""

    def __init__(self, acceptance, key):
        self.acceptance, self.key = acceptance, key

    def read(self, case_code, asset_code, version):
        from packages.evidence_details.acceptance_models import (
            AcceptanceReceipt,
            AcceptedRecords,
            AcceptedRecordsV21,
            ResearchGraphValue,
        )

        saved = self.acceptance.read(key=self.key)
        if saved is None:
            raise Fault("OBJECT_NOT_FOUND")
        records, graph, scores, receipt = saved
        graph = validate(ResearchGraphValue, graph)
        receipt = validate(AcceptanceReceipt, receipt)
        types = {
            "case-observation-interpretation/2.0.0": AcceptedRecords,
            "case-observation-interpretation/2.1.0": AcceptedRecordsV21,
        }
        model = types.get(records.get("schema_version"))
        if model is None:
            raise Fault("VERSION_UNSUPPORTED")
        records = validate(model, records)
        if (
            graph.scope.case_code != case_code
            or receipt.graph is None
            or receipt.graph.code != asset_code
            or receipt.graph.version != version
        ):
            raise Fault("VERSION_CONFLICT")
        self.acceptance.successor_material(key=self.key)  # exact original source/lineage read
        if graph.records_checksum != digest(records) or graph.scores_checksum != (
            digest(scores["scores"]) if scores["scores"] is not None else None
        ):
            raise ValueError("accepted_graph_binding")
        return dict(
            graph=graph.model_dump(mode="json"),
            view=None,
            records=records.model_dump(mode="json"),
            scores=scores["scores"],
            source_checksum=receipt.graph.checksum,
        )


def render_bound_json(value):
    """Self-contained escaped HTML, generated from the exact verified JSON response."""
    sha = digest(value)
    body = html.escape(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    return (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<meta name="source-checksum" content="{sha}"><title>Research record</title>'
        f"</head><body><p>Historical research record · clinical acceptance disabled</p>"
        f"<pre>{body}</pre></body></html>"
    ), sha
