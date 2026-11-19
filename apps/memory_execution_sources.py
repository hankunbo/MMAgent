"""Installed original Runtime/Planner owners for 10B, including pre-graph failures."""

from packages.evidence_details.current_rejection import CurrentResearchRejected
from packages.grounding.models import digest, validate
from packages.memory.writer_models import ExecutionSource


class ExecutionSources:
    def __init__(
        self,
        *,
        runtime,
        run_id,
        applicability,
        planner=None,
        research=None,
        documents=None,
        task_completion=None,
    ):
        self.runtime, self.run_id, self.planner = runtime, run_id, planner
        self.task_completion = task_completion
        self.applicability, self.research = tuple(applicability), research
        self.documents = documents or (research.documents if research else None)

    def read_task_completion(self):
        if self.task_completion is None:
            raise CurrentResearchRejected("task_completion_not_installed")
        proof = self.task_completion.read_task_completion(self.run_id)
        snapshot = self.snapshot()
        if proof.lifecycle_checksum != snapshot["runtime"]["lifecycle_checksum"]:
            raise CurrentResearchRejected("task_completion_source_changed")
        return proof, self.snapshot_reference(snapshot)

    def snapshot(self):
        runtime = self.runtime.read_memory_execution(self.run_id)
        if not runtime["synthetic"]:
            raise ValueError("live_domain_acceptance_not_enabled")
        if (
            self.task_completion is not None
            and runtime["lifecycle_phase"] == "FINISHED"
            and runtime["runtime_status"] == "COMPLETED_INSUFFICIENT_EVIDENCE"
        ):
            # Opted-in task sources must be current even at Writer/direct Repository
            # create/read/synchronize, before any qualification or publication exists.
            proof = self.task_completion.read_task_completion(self.run_id)
            if proof.lifecycle_checksum != runtime["lifecycle_checksum"]:
                raise CurrentResearchRejected("task_completion_source_changed")
        state = self.planner.replay(self.run_id) if self.planner else None
        if state and (
            state.binding.case_code != runtime["case_code"]
            or state.binding.run_id != self.run_id
            or state.binding.run_context_checksum != runtime["context_checksum"]
        ):
            raise ValueError("memory_scope")
        # Replay is historical. Current graph authority still belongs to the installed owner.
        if state and self.planner.owner.current(state.binding) != state.graph:
            raise CurrentResearchRejected("memory_source_changed")
        report = self.planner.report(self.run_id) if state else None
        if state and report.control_checksum != digest(state):
            raise CurrentResearchRejected("memory_source_changed")
        if self.runtime.read_memory_execution(self.run_id) != runtime:
            raise CurrentResearchRejected("memory_source_changed")
        return dict(
            schema_version="research-memory-execution/1.0.0",
            runtime=runtime,
            control=state.model_dump(mode="json") if state else None,
            control_report=report.model_dump(mode="json") if report else None,
            reviews=self._reviews(state) if state else {},
        )

    def _reviews(self, state):
        """Rebuild saved advice/decision/result separately; never call Expert again."""
        from agents.expert_agent.models import ExpertDecision, ReviewResultV2

        reviews = {}
        for n, event in enumerate(state.events):
            refs = [r for r in event.outcome.refs if r.role == "research_review"]
            if not refs or self.documents is None:
                continue
            if len(refs) != 1 or refs[0].version != "research-review-result/2.0.0":
                raise ValueError("memory_source_version")
            reference = refs[0]
            key = "review:" + reference.code
            saved = self.documents.read(key + ":result")
            if not saved:
                raise ValueError("memory_source_missing")
            result = validate(ReviewResultV2, saved[0])
            if (
                digest(result) != reference.checksum
                or result.run_id != self.run_id
                or result.event.code != reference.code
                or result.event.checksum != event.request_checksum
            ):
                raise ValueError("memory_source_binding")
            decision = None
            if result.decision_checksum is not None:
                values = self.documents.read(key + ":decision")
                if not values:
                    raise ValueError("memory_source_missing")
                decision = validate(ExpertDecision, values[0])
                if (
                    digest(decision) != result.decision_checksum
                    or decision.event != result.event
                    or decision.scope.run_id != self.run_id
                    or decision.scope.case_code != state.binding.case_code
                ):
                    raise ValueError("memory_source_binding")
            reviews[str(n)] = dict(
                result=result.model_dump(mode="json"),
                decision=decision.model_dump(mode="json") if decision else None,
                result_artifact=self.documents.reference(
                    key + ":result",
                    0,
                    saved[0],
                    role="review_result",
                    version=result.schema_version,
                ).model_dump(mode="json"),
            )
        return reviews

    def reference(self, owner, key, kind):
        if owner != "runtime_execution":
            if self.research is None:
                raise ValueError("memory_source_port_unavailable")
            return self.research.reference_for_memory(owner, key, kind)
        if key != self.run_id or kind != "execution_snapshot":
            raise ValueError("memory_source_binding")
        return self.snapshot_reference(self.snapshot())

    def snapshot_reference(self, snapshot):
        return ExecutionSource(
            case_code=snapshot["runtime"]["case_code"],
            run_id=self.run_id,
            key=self.run_id,
            object_code=self.run_id,
            checksum=digest(snapshot),
            object_checksum=digest(snapshot),
            dependencies=(
                digest(["runtime_execution", self.run_id, self.run_id, "execution_snapshot"]),
            ),
            applicability=self.applicability,
            nature="source_fact",
        )

    def verify(self, source):
        if source.owner != "runtime_execution":
            if self.research is None:
                raise ValueError("memory_source_port_unavailable")
            if source.owner == "simulation":
                actual = self.reference(source.owner, source.key, source.kind)
                if source.version != actual.version:
                    raise ValueError("memory_source_version")
                if actual != source:
                    raise CurrentResearchRejected("memory_source_changed")
                return actual
            return self.research.verify(source)
        source = validate(ExecutionSource, source)
        if source.run_id != self.run_id:
            raise ValueError("memory_scope")
        actual = self.reference(source.owner, source.key, source.kind)
        if source != actual:
            raise CurrentResearchRejected("memory_source_changed")
        return actual

    def required_fact_sources(self, snapshot):
        """Owner-defined completeness, independent of submitted Memory sources."""
        control = snapshot["control"]
        if control and control["graph"]["kind"] != "research_artifact":
            raise ValueError("memory_research_graph_required")
        if not control or control["graph"]["version"] != "case-evidence-graph/2.0.0":
            return ()
        if self.research is None or self.research.fact_reader is None:
            raise ValueError("memory_fact_reader_required")
        return self.research.accepted_sources(
            case_code=snapshot["runtime"]["case_code"],
            run_id=self.run_id,
            graph=control["graph"],
        )

    def verify_record(self, record):
        from packages.memory.writer_models import EpisodeData, Segment, Snapshot, WriterRecord
        from packages.memory_writer.projection import project, ref, step, terminal

        if not isinstance(record, WriterRecord):
            return
        snapshot = self.snapshot()
        source = self.snapshot_reference(snapshot)
        if source not in record.sources:
            raise CurrentResearchRejected("memory_source_changed")
        facts = ()
        if isinstance(record.detail, (Snapshot, EpisodeData)):
            facts = self.required_fact_sources(snapshot)
            supplied = tuple(s for s in record.sources if s.owner == "research_fact")
            if {s.kind for s in supplied} != {s.kind for s in facts}:
                raise ValueError("memory_accepted_sources_incomplete")
            if supplied != facts:
                raise ValueError("memory_source_binding")
        if isinstance(record.detail, Segment):
            expected = tuple(step(snapshot, source, s.index) for s in record.detail.steps)
            if record.detail.steps != expected:
                raise ValueError("memory_projection_binding")
        elif isinstance(record.detail, Snapshot):
            path = (
                "/control/binding/graph"
                if record.detail.role == "initial_historical"
                else "/control/graph"
            )
            if record.detail.evidence != ref(snapshot, source, path):
                raise ValueError("memory_projection_binding")
            if digest(record.detail.graph) != record.detail.evidence.checksum:
                raise ValueError("memory_projection_binding")
        elif isinstance(record.detail, EpisodeData):
            if record.detail.terminal != terminal(snapshot, source):
                raise ValueError("memory_projection_binding")
            supplements = tuple(s for s in record.sources if s.owner in {"simulation", "recovery"})
            expected = project(
                snapshot,
                source,
                request_key=record.detail.request_key,
                fact_sources=facts,
                supplements=supplements,
            )[-1]
            if expected != record:
                raise ValueError("memory_projection_binding")
