"""Historical aggregate snapshots, rebuilt from original accepted Review owners.

No new judgments, tool executions, publication or current-use permissions. The
unchanged archive source limit is checked after reconstructing the full lineage.
"""

import json
from types import SimpleNamespace

from agents.expert_agent.reporting import read_research_report
from db.runtime.repository.case_pipeline import identity
from db.runtime.repository.planner import PlannerRepository
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.accepted_aggregation import (
    ACCEPTANCE_KEY,
    GRANT_KEY,
    AcceptedOriginReader,
    AggregatedSubmission,
    AggregationGrant,
    AliasAggregatedSubmission,
    HistoricalRequests,
)
from packages.evidence_details.review_batches import read_submission
from packages.grounding.models import digest, validate
from packages.llm_provider.config import read_provider_config
from packages.multimodal_research.report_storage import unpack
from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments

from .archive_models import archive_record_class

REPORT_KEY = "accepted-aggregation-report"


class Documents:
    """Narrow read adapter; all artifacts go through the original ArchiveSources."""

    envelope = RuntimeDocuments.envelope
    reference = RuntimeDocuments.reference

    def __init__(self, owner, case, run, root, refs, sessions, acceptance_key):
        self.owner, self.case, self.run_id = owner, case, run
        self.root = SimpleNamespace(checksum=root)
        self.refs, self.sessions, self.acceptance_key = refs, sessions, acceptance_key

    def artifact(self, code):
        a, raw = self.owner.artifact(self.case, code)
        self.refs[a.asset_code] = a
        return a, raw

    def read(self, key):
        if self.owner.read_run_case(self.run_id) != self.case:
            raise ValueError("memory_aggregation_source_case")
        artifacts = self.owner.runtime.read_observation_bundle(
            identity("TASK", self.run_id + ":agent:" + key)
        )
        if not artifacts:
            return None
        values = []
        for index, artifact in enumerate(artifacts):
            _, raw = self.artifact(artifact.asset_code)
            d = json.loads(raw)
            if (d.get("run_id"), d.get("root_checksum"), d.get("key_checksum"), d.get("index")) != (
                self.run_id,
                self.root.checksum,
                digest(key),
                index,
            ):
                raise ValueError("memory_aggregation_document_binding")
            values.append(d["value"])
        return tuple(values)

    def validate_scope(self, scope):
        if (scope.case_code, scope.run_id) != (self.case, self.run_id):
            raise ValueError("memory_aggregation_scope")
        a, raw = self.artifact(scope.graph.code)
        value = json.loads(raw)
        if a.checksum != scope.graph.checksum:
            raise ValueError("memory_aggregation_graph")
        if a.checksum != self.root.checksum:
            if (value.get("run_id"), value.get("root_checksum")) != (
                self.run_id,
                self.root.checksum,
            ):
                raise ValueError("memory_aggregation_graph_lineage")
            value = value["value"]
        if value.get("schema_version", value.get("schema")) != scope.graph.version:
            raise ValueError("memory_aggregation_graph_version")

    def consumed(self, graph):
        values = self.read("accepted:" + self.acceptance_key)
        return bool(
            values
            and values[-1].get("status") == "accepted"
            and values[-1].get("parent_graph", {}).get("code") == graph.code
            and values[-1].get("parent_graph", {}).get("checksum") == graph.checksum
        )

    def state(self):
        values = self.read("dynamic-phase-input")
        if not values:
            raise ValueError("memory_aggregation_source_state_missing")
        state = validate(DomainState, values[0])
        self.validate_scope(state.material.scope)
        for resource in state.resources:
            a, _ = self.artifact(resource.code)
            if a.checksum != resource.checksum:
                raise ValueError("memory_aggregation_resource")
        return state


def original_origins(documents, state, submission, *, plan=None):
    """Revalidate unchanged items and exact request/permission provenance."""
    reasoned_partition = False
    alias_projection = isinstance(submission, AliasAggregatedSubmission)
    if alias_projection:
        from packages.multimodal_research.accepted_aggregation import (
            AliasAggregationPlan,
            read_plan,
        )

        if not plan or not isinstance(read_plan(plan), AliasAggregationPlan):
            raise ValueError("memory_aggregation_alias_plan_required")
    if plan and plan.get("schema_version") == "accepted-review-aggregation-plan/1.4.0":
        from packages.multimodal_research.accepted_aggregation import read_plan

        read_plan(plan)  # Only the explicit persisted partition-only plan opts in.
        reasoned_partition = True
    readers, configs = [], {}
    for expected in submission.origins:
        old = Documents(
            documents.owner,
            documents.case,
            expected.run_id,
            documents.root.checksum,
            documents.refs,
            documents.sessions,
            expected.acceptance_key,
        )
        parent = old.state()
        source = read_submission(old.read("submission:" + expected.acceptance_key)[0]["submission"])
        for batch in source.batches:
            values = old.read(
                "review-batches:" + source.plan.key + ":" + batch.batch_id + ":response"
            )
            if values and "provider_config" in values[0]:
                conf = read_provider_config(values[0]["provider_config"])
                configs[digest(conf)] = conf
        readers.append((expected, old, parent))
    items = []
    for expected, old, parent in readers:
        config = configs.get(state.config_checksum)
        if parent.config_checksum != state.config_checksum and config is None:
            raise ValueError("memory_aggregation_configuration_lineage")

        def aliases(old=old):
            previous = old.read("generic-review-source-aliases")
            following = documents.read("generic-review-source-aliases")
            if previous is None or following is None or len(previous) != 1 or len(following) != 1:
                raise ValueError("memory_aggregation_alias_manifest_missing")
            return previous[0], following[0]

        reader = AcceptedOriginReader(
            documents=old,
            requests=HistoricalRequests(RuntimeAgentJournal(old)),
            state_reader=lambda parent=parent: parent,
            acceptance_key=expected.acceptance_key,
            current_config=config,
            reasoned_batch_resize=reasoned_partition,
            alias_manifests=aliases if alias_projection else None,
        )
        proof, original_items, _ = reader.read(state)
        if proof != expected:
            raise ValueError("memory_aggregation_origin_changed")
        items.extend(original_items)
    if tuple(sorted(items, key=lambda i: i.assessment_id)) != submission.candidate.items:
        raise ValueError("memory_aggregation_items_changed")


def prepare(owner, *, key, case_code, run_id, origin, envelope, kind, target):
    if (
        kind != "graph_snapshot"
        or envelope.get("run_id") != run_id
        or envelope.get("key_checksum") != digest(REPORT_KEY)
        or envelope.get("index") != 0
        or owner.read_run_case(run_id) != case_code
    ):
        raise ValueError("memory_aggregation_snapshot_binding")
    bundle = owner.runtime.read_observation_bundle(
        identity("TASK", run_id + ":agent:" + REPORT_KEY)
    )
    if not bundle or bundle[0] != origin:
        raise ValueError("memory_aggregation_origin_binding")
    refs = {origin.asset_code: origin}
    docs = Documents(
        owner,
        case_code,
        run_id,
        envelope["root_checksum"],
        refs,
        PlannerRepository(owner.runtime._target),
        ACCEPTANCE_KEY,
    )
    report, _ = unpack(docs.read(REPORT_KEY), key=REPORT_KEY)
    result = report["result"]
    state = docs.state()
    submission = read_submission(docs.read("submission:" + ACCEPTANCE_KEY)[0]["submission"])
    grant = validate(AggregationGrant, docs.read(GRANT_KEY)[0])
    manifest = docs.read("accepted-aggregation-manifest")[0]
    control = docs.sessions.read(run_id)
    if (
        not isinstance(submission, AggregatedSubmission)
        or result.get("schema_version")
        not in {
            "generic-accepted-aggregation-result/1.0.0",
            "multimodal-accepted-aggregation-result/1.0.0",
        }
        or result.get("complete") is not True
        or result.get("new_requests") != 0
        or result.get("case_code") != case_code
        or target != state.material.goal_code
        or docs.read("multimodal-case-result") != (result,)
        or docs.read("dynamic-phase-result") != (result["phases"][-1],)
        or result["phases"][-1]["run_id"] != run_id
        or result["phases"][-1]["acceptance_key"] != ACCEPTANCE_KEY
        or result["phases"][-1].get("accepted") is not True
        or result["phases"][-1].get("target") is not None
        or control is None
        or control.phase != "stopped"
        or control.cancelled
        or control.pending
        or RuntimeAgentJournal(docs).read()
        or grant.run_id != run_id
        or grant.control_checksum != digest(control)
        or grant.state_checksum != digest(state)
        or grant.submission_checksum != digest(submission)
        or grant.origins_checksum != digest(submission.origins)
        or manifest.get("run_id") != run_id
        or manifest.get("state_checksum") != digest(state)
        or manifest.get("submission_checksum") != digest(submission)
        or manifest.get("origins_checksum") != digest(submission.origins)
        or manifest.get("original_result_checksum") != result.get("original_result_checksum")
        or result["origins"] != [v.model_dump(mode="json") for v in submission.origins]
    ):
        raise ValueError("memory_aggregation_committed_binding")
    original_origins(docs, state, submission, plan=manifest.get("plan"))
    checked = read_research_report(
        ResearchAcceptance(owner=None, documents=docs, requests=None),
        key=ACCEPTANCE_KEY,
        parent=state,
    )
    if (
        checked != report["original_review"]
        or checked["scores"] != report["scores"]
        or checked["records"]["candidate"] != submission.candidate.model_dump(mode="json")
        or checked["records"]["origins"] != result["origins"]
        or checked["receipt"]["submission_checksum"] != digest(submission)
        or checked["receipt"].get("permission_checksum") != digest(grant)
    ):
        raise ValueError("memory_aggregation_report_binding")
    sources = tuple(
        owner.reference(case_code=case_code, run_id=run_id, code=k) for k in sorted(refs)
    )
    modalities = sorted({m for r in state.policy.requirements for m in r.input_modalities})
    return archive_record_class(sources)(
        key=key,
        kind=kind,
        case_code=case_code,
        run_id=run_id,
        applicability=("USE:RESEARCH_WORKFLOW",),
        nature="source_fact",
        sources=sources,
        detail=dict(
            type="case_snapshot",
            target=target,
            modality="+".join(modalities),
            modality_source="original_accepted_aggregation_report",
            origin_code=origin.asset_code,
            steps=(),
            outcome="recorded",
        ),
    )
