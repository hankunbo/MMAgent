"""Two-case bounded composition of original Runtime/Provider/domain owners.

Each observation and final review has an original run with its own immutable
policy. The campaign is an orchestration manifest, not another execution system.
All phase reservations count against the same case cap, including old attempts.
"""

import importlib.util
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

from apps.vectorengine_live_assembly import assemble, make_binding, run_case_stages, stop_failed_run
from apps.vectorengine_single_case import load_config, write_once
from db.runtime.repository.planner import PlannerRepository
from db.runtime.repository.research import ResearchRunRepository
from packages.evidence_details.acceptance_models import DomainState, read_receipt, read_records
from packages.grounding.models import canonical, digest, seal, validate
from packages.llm_provider.bounded_live import BoundedLiveBudget
from packages.llm_provider.provider import ProviderError
from packages.planner_control.models import Command
from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments
from packages.research_runtime.models import ResearchRunContext

from .case_live import requests_for
from .dynamic_materials import DynamicCaseEvidence
from .dynamic_stages import VISUAL_IDS, PhaseEvidence

CONFIG = "configs/agents/vectorengine_dynamic_evidence_v1.json"
VERSION = "13-dynamic-case-v1"


def bounded_phase_binding(binding, deadline, existing=None):
    """Narrow a new original Planner profile, preserving its exact clock contract."""
    if existing is not None:
        recovered = validate(
            type(binding),
            binding.model_copy(
                update=dict(
                    profile=binding.profile.model_copy(
                        update={"seconds": existing.profile.seconds}
                    ),
                    deadline=existing.deadline,
                )
            ),
        )
        if recovered != existing:
            raise ValueError("dynamic_phase_existing_binding_changed")
        return existing
    seconds = min(binding.profile.seconds, int((deadline - binding.started_at).total_seconds()))
    if seconds < 1:
        raise ProviderError("deadline")
    return validate(
        type(binding),
        binding.model_copy(
            update=dict(
                profile=binding.profile.model_copy(update={"seconds": seconds}),
                deadline=binding.started_at + timedelta(seconds=seconds),
            )
        ),
    )


class Campaign:
    version = VERSION
    evidence_class = DynamicCaseEvidence
    phase_class = PhaseEvidence

    def read_model(self):
        from .dynamic_model import read_model

        return read_model(self.base, self.target)

    def __init__(self, sample):
        self.base = self.evidence_class(sample)
        self.config = load_config(CONFIG)
        self.requests, self.target = requests_for(self.base)
        self.contexts, self.sessions = (
            ResearchRunRepository(self.target),
            PlannerRepository(self.target),
        )
        self.output = self.base.directory / "dynamic-v1"
        self.output.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.sample = sample

    def documents(self, run_id, root=None):
        return RuntimeDocuments(
            repo=self.base.runtime,
            contexts=self.contexts,
            sessions=self.sessions,
            store=self.base.objects,
            run_id=run_id,
            root=root or self.base.root,
        )

    def prior(self):
        old = validate(
            DomainState, json.loads((self.base.directory / "graph/domain-state.json").read_bytes())
        )
        root = self.base.runtime.read_case_artifact(
            self.base.binding["case_code"], old.material.scope.graph.code
        )
        docs = self.documents(old.material.scope.run_id, root)
        events = RuntimeAgentJournal(docs).read()
        return dict(
            run_id=docs.run_id,
            reservations=sum(e.get("phase") == "reserved" for e in events),
            journal_checksum=digest(events),
        )

    def paths(self):
        return [self.output / ("vision-" + digest(t)[7:19]) for t in VISUAL_IDS] + [
            self.output / "review"
        ]

    def usage(self):
        used = 0
        for path in self.paths():
            if (path / "request.json").exists():
                request = json.loads((path / "request.json").read_bytes())
                run = request["lock"]["run_id"]
                if self.contexts.read_research_run(run) is not None:
                    used += sum(
                        e.get("phase") == "reserved"
                        for e in RuntimeAgentJournal(self.documents(run)).read()
                    )
        return used

    def freeze(self):
        from packages.object_store import checksum

        path = self.output / "protocol.json"
        files = [
            Path(CONFIG),
            *Path("packages/evaluation_preparation").glob("dynamic*.py"),
            Path("apps/vectorengine_live_assembly.py"),
            Path("agents/expert_agent/service.py"),
            Path("configs/prompts/agents/review-batch-1.5.0.txt"),
            Path("packages/llm_provider/transport.py"),
            Path("packages/llm_provider/review_citations.py"),
        ]
        implementation = {str(p): checksum(p.read_bytes()) for p in files}
        expected = dict(
            schema_version="dynamic-case-campaign/1.0.0",
            sample=self.sample,
            case_code=self.base.binding["case_code"],
            image_checksum=self.base.image_artifact.checksum,
            provider_checksum=digest(self.config),
            visual_requirements=list(VISUAL_IDS),
            previous=self.prior(),
            max_cumulative_graph_requests=40,
            max_new_requests=10,
            implementation=implementation,
            seconds=2700,
            clinical_acceptance=False,
        )
        if path.exists():
            value = json.loads(path.read_bytes())
            if {k: v for k, v in value.items() if k not in {"started_at", "deadline"}} != expected:
                self.check_continuation(value, expected)
        else:
            now = datetime.now(UTC)
            value = dict(
                **expected,
                started_at=now.isoformat(),
                deadline=(now + timedelta(seconds=2700)).isoformat(),
            )
            write_once(path, value)
        self.protocol = value
        return value

    def check_continuation(self, previous, expected):
        """Explicit operator-reviewed code-only handoff; never reset input or budget."""
        before = {
            k: v
            for k, v in previous.items()
            if k not in {"started_at", "deadline", "implementation"}
        }
        after = {k: v for k, v in expected.items() if k != "implementation"}
        proof_path = self.output / "review-implementation-handoff.json"
        if before != after or not proof_path.is_file():
            raise ValueError("dynamic_campaign_changed")
        proof = json.loads(proof_path.read_bytes())
        changed = {
            k
            for k in previous["implementation"].keys() | expected["implementation"].keys()
            if previous["implementation"].get(k) != expected["implementation"].get(k)
        }
        if (
            proof.get("schema_version") != "dynamic-code-read-port-handoff/1.0.0"
            or proof.get("original_protocol_checksum") != digest(previous)
            or proof.get("current_implementation") != expected["implementation"]
            or not changed
            <= {
                "packages/evaluation_preparation/dynamic_model.py",
                "packages/evaluation_preparation/dynamic_campaign.py",
                "packages/evaluation_preparation/dynamic_report.py",
                "packages/evaluation_preparation/dynamic_stages.py",
            }
        ):
            raise ValueError("dynamic_unreviewed_implementation_change")
        gate = Path(proof["verification_path"])
        if not gate.is_relative_to(Path("docs/acceptance/13-dynamic-campaign-v1")):
            raise ValueError("dynamic_continuation_gate_path")
        verified = json.loads(gate.read_bytes())
        if (
            digest(verified) != proof["verification_checksum"]
            or verified.get("passed") is not True
            or verified.get("implementation") != expected["implementation"]
        ):
            raise ValueError("dynamic_continuation_gate")
        expected_paths = {
            str(p / "result.json") for p in self.paths()[:-1] if (p / "result.json").exists()
        }
        if set(proof["acquisition_results"]) != expected_paths:
            raise ValueError("dynamic_continuation_acquisition_inventory")
        # All already completed acquisitions must still be the exact owner records.
        for path, expected_checksum in proof["acquisition_results"].items():
            result = json.loads(Path(path).read_bytes())
            if digest(result) != expected_checksum or self.documents(result["run_id"]).read(
                "dynamic-phase-result"
            ) != (result,):
                raise ValueError("dynamic_continuation_acquisition_changed")

    def guard(self, *, dispatch=False):
        if self.prior() != self.protocol["previous"]:
            raise ProviderError("scope", fields=("prior_case_attempts",))
        if datetime.now(UTC) >= datetime.fromisoformat(self.protocol["deadline"]):
            raise ProviderError("deadline")
        used = self.usage()
        # before_wire includes the reservation just committed by RequestBudget.
        limit = min(
            self.protocol["max_new_requests"], 40 - self.protocol["previous"]["reservations"]
        )
        if used > limit or (not dispatch and used >= limit):
            raise ProviderError("budget")

    def budget_factory(self, **kwargs):
        campaign = self
        # execute_phase narrows the original Planner binding to the campaign's
        # remaining window. Pass that explicit contract to the original budget;
        # do not require a fresh 2700 seconds after local preparation took time.
        if kwargs.get("allocation") is None and 0 < kwargs["binding"].profile.seconds < 2700:
            kwargs.setdefault("bounded_phase", True)

        class SharedCaseBudget(BoundedLiveBudget):
            @property
            def effective_deadline(self):
                return min(self.deadline, datetime.fromisoformat(campaign.protocol["deadline"]))

            def before_reserve(self, checksum):
                campaign.guard()
                super().before_reserve(checksum)

            def before_wire(self, *args):
                campaign.guard(dispatch=True)
                super().before_wire(*args)

        return SharedCaseBudget(**kwargs)

    def accepted_observations(self):
        observations = {}
        for target, path in zip(VISUAL_IDS, self.paths()[:-1], strict=True):
            if not (path / "result.json").exists():
                continue
            result = json.loads((path / "result.json").read_bytes())
            docs = self.documents(result["run_id"])
            saved_result = docs.read("dynamic-phase-result")
            if saved_result != (result,):
                raise ValueError("dynamic_phase_result_owner_binding")
            if not result["accepted"]:
                continue
            initial = validate(DomainState, docs.read("dynamic-phase-input")[0])
            expected, _ = self.base.project(run_id=result["run_id"], config=self.config)
            from .dynamic_stages import acquisition_state

            if acquisition_state(expected, target, self.base.image_artifact) != initial:
                raise ValueError("dynamic_current_observation_source_changed")
            accepted = docs.read("accepted:" + result["acceptance_key"])
            if accepted is None or len(accepted) != 4:
                raise ValueError("dynamic_observation_acceptance_missing")
            records, receipt = read_records(accepted[0]), read_receipt(accepted[3])
            if (
                receipt.status != "accepted"
                or receipt.records_checksum != digest(records)
                or records.scope != initial.material.scope
                or len(records.observations) != 1
            ):
                raise ValueError("dynamic_observation_receipt_binding")
            observation = records.observations[0]
            if (
                observation.reference.code != "OBS:DYNAMIC:" + target
                or observation.image.reference.checksum != self.base.image_artifact.checksum
            ):
                raise ValueError("dynamic_observation_case_binding")
            observations[target] = observation
        return observations

    def phase_path(self, target):
        return self.paths()[VISUAL_IDS.index(target)] if target else self.paths()[-1]

    def request_key(self, target):
        return self.version + ":" + self.sample + ":" + (target or "review")

    def guard_window(self):
        """Fast campaign boundary; the full guard remains at every wire boundary."""
        if datetime.now(UTC) >= datetime.fromisoformat(self.protocol["deadline"]):
            raise ProviderError("deadline")

    def cancellation_signal(self):
        return Event()

    def register_phase(self, evidence, key):
        """Original v1 composition; task-v2 entries override only identity creation."""
        request = self.requests.register(self.base.visible_value, key)
        state, _ = evidence.project(run_id=request.lock.run_id, config=self.config)
        context = seal(
            ResearchRunContext,
            request=request,
            synthetic=False,
            scope_checksum=digest(request.scope),
        )
        self.contexts.create_research_run(context, key)
        return context, state

    def execute_phase(self, target=None):
        role = "vision" if target else "review"
        path = self.phase_path(target)
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        evidence = self.phase_class(
            self.base,
            target=target,
            observation_reader=self.accepted_observations,
            model_reader=self.read_model,
        )
        key = self.request_key(target)
        context, state = self.register_phase(evidence, key)
        request = context.request
        docs = self.documents(context.run_id)
        if docs.read("dynamic-phase-result"):
            result = docs.read("dynamic-phase-result")[0]
            write_once(path / "result.json", result)
            return result
        prior = self.sessions.read(context.run_id)
        if prior is not None and (prior.phase == "stopped" or prior.cancelled):
            raise ValueError("dynamic_stopped_phase_requires_report_recovery_not_execution")
        self.guard()
        definition_projection = None
        if hasattr(evidence, "definition_revision_manifest"):
            from .generic_definition_projection import projection_record

            # Current source/definition I/O precedes the short Runtime write.
            definition_projection = projection_record(evidence, state, context.run_id)
        capability_projection = None
        if hasattr(evidence, "capability_projection"):
            from .generic_capability_projection import projection_record as capability_record

            capability_projection = capability_record(evidence, state, context.run_id)
        with docs.session():
            docs.write("dynamic-phase-input", (state,))
            if capability_projection is not None:
                docs.write("generic-capability-demand-projection", (capability_projection,))
            if definition_projection is not None:
                docs.write("generic-definition-review-projection", (definition_projection,))
            if hasattr(evidence, "source_alias_manifest"):
                docs.write("generic-review-source-aliases", (evidence.source_alias_manifest,))
            docs.write("dynamic-campaign", (self.protocol,))
        write_once(path / "request.json", request)
        started, context_checksum = self.sessions.origin(context.run_id)
        binding, ranking = make_binding(
            initial=state,
            config=self.config,
            started=started,
            context_checksum=context_checksum,
            catalog=request.lock.catalog_ref,
        )
        # Only a new Planner binding is narrowed. A persisted binding is
        # never rewritten; the campaign guard still limits its later work.
        deadline = (
            self.work_deadline()
            if hasattr(self, "work_deadline")
            else datetime.fromisoformat(self.protocol["deadline"])
        )
        binding = bounded_phase_binding(binding, deadline, prior.binding if prior else None)
        planner, owner, ports, review = assemble(
            evidence=evidence,
            initial=state,
            config=self.config,
            documents=docs,
            binding=binding,
            ranking=ranking,
            budget_factory=self.budget_factory,
            source_quote_policy="research-quote-policy/layout-whitespace/1.0.0",
        )
        cancel = self.cancellation_signal()
        ports.budget.cancel_signal = cancel
        if self.sessions.read(binding.run_id) is None:
            planner.begin(binding)
        completed, error = [], None
        try:
            if role == "review" and not state.policy.items:
                from packages.evidence_scoring.accepted import build_accepted

                with docs.session():
                    docs.write(
                        "dynamic-zero-evidence",
                        (
                            dict(
                                status="evidence_limited",
                                scores=build_accepted(state, None),
                                requirements=state.policy.requirements,
                                no_review_requests=True,
                            ),
                        ),
                    )
            else:
                completed, _ = run_case_stages(
                    planner=planner,
                    owner=owner,
                    ports=ports,
                    review=review,
                    documents=docs,
                    binding=binding,
                    config=self.config,
                    output=path,
                    signal_cancel=cancel,
                    selected_roles=(role,),
                    before_step=self.guard_window,
                )
        except (ProviderError, ValueError, OSError) as exc:
            error = (
                str(exc)[:160]
                if isinstance(exc, (ProviderError, ValueError))
                else type(exc).__name__
            )
            stop_failed_run(
                planner=planner,
                sessions=self.sessions,
                binding=binding,
                budget=ports.budget,
                error=exc,
                cancel=cancel,
            )
        current = planner.read(binding.run_id)
        if current.phase != "stopped" and current.pending is None:
            planner.step(
                binding.run_id,
                Command(
                    key="dynamic-phase-stop", expected_revision=current.revision, action="stop"
                ),
                cancel=cancel,
            )
        last = completed[-1] if completed else {}
        result = dict(
            schema_version="dynamic-phase-result/1.0.0",
            run_id=binding.run_id,
            target=target,
            accepted=last.get("domain_status") == "accepted",
            acceptance_key=last.get("acceptance_key"),
            error=error
            or (last.get("reason") if last.get("domain_status") != "accepted" else None),
            evidence_limited=role == "review" and not state.policy.items,
            requests=ports.budget.summary(),
            transport=ports.budget.transport_counts(),
            planner=planner.report(binding.run_id),
            clinical_acceptance=False,
        )
        result = json.loads(canonical(result))
        with docs.session():
            docs.write("dynamic-phase-result", (result,))
        if target is None:
            from packages.multimodal_research.report_io import publish_json

            publish_json(path / "result.json", result)
        else:
            write_once(path / "result.json", result)
        return result

    def execute(self, *, acquire_only=False):
        import fcntl

        with (self.output / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.freeze()
            if (self.output / "result.json").exists():
                saved = json.loads((self.output / "result.json").read_bytes())
                docs = self.documents(saved["phases"][-1]["run_id"])
                if docs.read("dynamic-case-result") != (saved,):
                    raise ValueError("dynamic_case_result_readback")
                from .dynamic_report import export

                export(self, saved)
                return saved
            self.guard()
            spec = importlib.util.spec_from_file_location(
                "private_launcher", ".local/vectorengine/run.py"
            )
            launcher = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(launcher)
            name = self.config.transport.api_key_env
            previous = os.environ.get(name)
            try:
                os.environ[name] = launcher.load_secret()
                phases = [self.execute_phase(target) for target in VISUAL_IDS]
                if acquire_only:
                    return dict(
                        stage="acquisition",
                        phases=phases,
                        complete=all(p["accepted"] for p in phases),
                        new_requests=self.usage(),
                        review_not_started=True,
                    )
                phases.append(self.execute_phase())
            finally:
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous
            result = dict(
                schema_version="dynamic-case-result/1.0.0",
                sample=self.sample,
                case_code=self.base.binding["case_code"],
                phases=phases,
                complete=phases[-1]["accepted"] or phases[-1]["evidence_limited"],
                new_requests=self.usage(),
                prior_graph_requests=self.protocol["previous"]["reservations"],
                clinical_acceptance=False,
                hidden_labels_used=False,
            )
            final_docs = self.documents(phases[-1]["run_id"])
            with final_docs.session():
                final_docs.write("dynamic-case-result", (result,))
            write_once(self.output / "result.json", result)
            from .dynamic_report import export

            export(self, result)
            return result
