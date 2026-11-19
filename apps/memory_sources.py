"""Installed owner adapters: no caller accepted flags or export-file authorization."""

from agents.expert_agent.models import ReviewResultV2
from packages.evidence_details.current_rejection import CurrentResearchRejected
from packages.grounding.models import digest, validate
from packages.memory.models import Source
from packages.validation_recovery.models import Failure, RecoveryFallbackReport, RecoveryStopReport


class ResearchSources:
    """Per-run installation; factories and original owners are trusted assembly dependencies.

    Current fact rebuilding is mandatory even for stored historical Control/documents.
    No model calls, simulation runs, recovery.resolve or new actions occur here.
    """

    def __init__(
        self, *, fact_reader, applicability, simulation=None, documents=None, recovery=None
    ):
        self.fact_reader = fact_reader
        self.planner, self.run_id = fact_reader.planner, fact_reader.run_id
        self.applicability = tuple(applicability)
        self.simulation, self.documents = simulation, documents
        self.recovery = recovery

    def reference(self, owner, key, kind):
        return self._reference(owner, key, kind, evidence_only=False)

    def reference_for_memory(self, owner, key, kind):
        """Stopped-run simulation evidence, still rebuilt by its original owner."""
        return self._reference(owner, key, kind, evidence_only=owner == "simulation")

    def _reference(self, owner, key, kind, *, evidence_only):
        fact = self.fact_reader()
        return self._from_fact(fact, owner, key, kind, evidence_only=evidence_only)

    def _from_fact(self, fact, owner, key, kind, *, evidence_only=False, documents=None):
        b = fact.binding
        value, version, code, nature = self._resolve(
            fact, owner, key, kind, evidence_only=evidence_only
        )
        document_key = None
        index = 0
        if owner == "research_fact":
            document_key = "accepted:" + key
            index = {"records": 0, "graph": 1, "scores": 2, "receipt": 3}[kind]
        elif owner == "simulation":
            document_key = "simulation:result:" + key
        elif owner == "recovery":
            suffix = "report" if kind == "recovery-stop-report" else "fallback-report"
            document_key = "recovery:" + key + ":" + suffix
        elif owner == "agent_review":
            document_key = "review:" + key
        object_checksum = digest(value)
        if document_key is not None:
            documents = (
                documents or self.documents or self.fact_reader.path_reader.acceptance.documents
            )
            values = documents.read(document_key)
            if values is None:
                raise ValueError("memory_source_missing")
            reference = documents.reference(
                document_key, index, values[index], role=kind, version=version
            )
            code, object_checksum = reference.code, reference.checksum
        return Source(
            owner=owner,
            case_code=b.case_code,
            run_id=b.run_id,
            key=key,
            object_code=code,
            kind=kind,
            version=version,
            checksum=digest(value),
            object_checksum=object_checksum,
            dependencies=tuple(
                digest(["research_fact", b.run_id, self.fact_reader.path_reader.key, part])
                for part in ("records", "graph", "scores", "receipt")
            ),
            semantics_version="counterfactual-semantics/1.1.0" if owner == "simulation" else None,
            applicability=self.applicability,
            nature=nature,
        )

    def accepted_sources(self, *, case_code, run_id, graph):
        """Rebuild one exact acceptance batch through FactReader and original objects."""
        from packages.planner_control.models import Graph

        reader = self.fact_reader
        if reader is None:
            raise ValueError("memory_fact_reader_required")
        path = reader.path_reader
        if path is None or path.acceptance is None or path.acceptance.documents is None:
            raise ValueError("memory_source_port_unavailable")
        fact = reader()
        receipt = fact.summary.original_report["receipt"]
        accepted_graph = Graph(
            **fact.summary.graph, kind="research_artifact", parent=receipt["parent_graph"]
        )
        if (
            (fact.binding.case_code, fact.binding.run_id) != (case_code, run_id)
            or accepted_graph.model_dump(mode="json") != graph
            or fact.summary.binding["current_scope"]["graph"] != graph
        ):
            raise ValueError("memory_source_binding")
        key = reader.path_reader.key
        sources = tuple(
            self._from_fact(fact, "research_fact", key, kind, documents=path.acceptance.documents)
            for kind in ("records", "graph", "scores", "receipt")
        )
        # The owner and objects have no shared global snapshot. Detect drift across
        # this bounded read, and keep the existing post-save/current revalidation.
        if reader.path_reader.key != key or reader().checksum != fact.checksum:
            raise CurrentResearchRejected("memory_source_changed")
        return sources

    def verify(self, source):
        source = validate(Source, source)
        if source.run_id != self.run_id:
            raise ValueError("memory_scope")
        actual = self.reference(source.owner, source.key, source.kind)
        if actual.version != source.version or actual.semantics_version != source.semantics_version:
            raise ValueError("memory_source_version")
        if actual != source:
            raise CurrentResearchRejected("memory_source_changed")
        return actual

    def _resolve(self, fact, owner, key, kind, *, evidence_only=False):
        if owner == "research_fact":
            if key != self.fact_reader.path_reader.key or kind not in {
                "records",
                "graph",
                "scores",
                "receipt",
            }:
                raise ValueError("memory_source_binding")
            value = fact.summary.original_report[kind]
            versions = {
                "records": "case-observation-interpretation/2.1.0",
                "graph": "case-evidence-graph/2.0.0",
                "scores": "case-research-scoring/6.1.0",
                "receipt": "research-acceptance/1.0.0",
            }
            if value["schema_version"] != versions[kind]:
                raise ValueError("memory_source_version")
            return value, value["schema_version"], fact.summary.graph["code"], "source_fact"
        if owner == "planner_event":
            state = self.planner.replay(self.run_id)
            events = [e for e in state.events if e.command.key == key]
            if len(events) != 1 or kind not in {"control_event", "tool_execution_event"}:
                raise ValueError("memory_source_binding")
            e = events[0]
            if kind == "tool_execution_event" and e.command.action not in {
                "execute",
                "recover",
                "verify",
            }:
                raise ValueError("memory_source_binding")
            return e, e.schema_version, e.command.key, "source_fact"
        if owner == "simulation":
            if self.simulation is None or kind != "counterfactual_report":
                raise ValueError("memory_source_port_unavailable")
            value = (
                self.simulation.read_evidence(candidate_id=key)
                if evidence_only
                else self.simulation.read(candidate_id=key, historical=False)
            )
            if value is None or value["schema_version"] != "counterfactual-report/1.1.0":
                raise ValueError("memory_source_version")
            return value, value["schema_version"], value["simulation_id"], "simulation"
        if self.documents is None or self.documents.run_id != self.run_id:
            raise ValueError("memory_source_port_unavailable")
        if owner == "recovery":
            if kind not in {"recovery-stop-report", "recovery-fallback-report"}:
                raise ValueError("memory_source_binding")
            suffix = "report" if kind == "recovery-stop-report" else "fallback-report"
            values = self.documents.read("recovery:" + key + ":" + suffix)
            if not values:
                raise ValueError("memory_source_missing")
            cls = RecoveryStopReport if suffix == "report" else RecoveryFallbackReport
            report = validate(cls, values[0])
            incident = report.incident
            current = self.planner.replay(self.run_id)
            if (
                incident.key != key
                or incident.control.binding != current.binding
                or current.events[: len(incident.control.events)] != incident.control.events
                or report.control_observed.binding != current.binding
            ):
                raise ValueError("memory_source_binding")
            if self.recovery is None or self.recovery.planner is not self.planner:
                raise ValueError("memory_source_port_unavailable")
            from packages.validation_recovery.service import render

            stored_incident = self.documents.read("recovery:" + key + ":incident")
            stored_policy = self.documents.read("recovery:policy")
            if (
                stored_incident != (incident.model_dump(mode="json"),)
                or stored_policy
                != (
                    dict(
                        binding_checksum=digest(current.binding),
                        policy=self.recovery.policy.model_dump(mode="json"),
                    ),
                )
                or incident.policy_checksum != digest(self.recovery.policy)
                or values != (report.model_dump(mode="json"), render(report))
                or report.control_observed.events
                != current.events[: len(report.control_observed.events)]
                or not any(e.command == incident.command for e in current.events)
            ):
                raise ValueError("memory_source_binding")
            rebuilt = validate(Failure, self.recovery.failure_reader(incident.control))
            if rebuilt != incident.failure:
                raise CurrentResearchRejected("memory_recovery_source_changed")
            nature = (
                "missing_evidence"
                if incident.failure.category in {"missing_input", "product_demand"}
                else "source_fact"
                if incident.failure.category in {"medical_negative", "valid_refutation"}
                else "technical_failure"
            )
            return report, report.schema_version, incident.command.key, nature
        if owner == "agent_review":
            if kind != "review_result":
                raise ValueError("memory_source_binding")
            values = self.documents.read("review:" + key)
            if not values:
                raise ValueError("memory_source_missing")
            result = validate(ReviewResultV2, values[0])
            current = self.planner.replay(self.run_id)
            if result.run_id != self.run_id or not any(
                e.request_checksum == result.event.checksum
                and any(r.checksum == digest(result) for r in e.outcome.refs)
                for e in current.events
            ):
                raise ValueError("memory_source_binding")
            return result, result.schema_version, result.event.code, "agent_opinion"
        raise ValueError("memory_source_port_unavailable")


class SourceRegistry:
    def __init__(self, owners):
        self.owners = dict(owners)

    def read_task_completion(self, run_id):
        if run_id not in self.owners:
            raise ValueError("memory_source_port_unavailable")
        return self.owners[run_id].read_task_completion()

    def verify(self, source):
        if source.run_id not in self.owners:
            raise ValueError("memory_source_port_unavailable")
        return self.owners[source.run_id].verify(source)

    def verify_record(self, record):
        if record.run_id not in self.owners:
            raise ValueError("memory_source_port_unavailable")
        return self.owners[record.run_id].verify_record(record)
