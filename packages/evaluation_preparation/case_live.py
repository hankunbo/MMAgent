"""New case composition using the original request/run/Planner/review owners."""

import importlib.util
import json
import os
import signal
from pathlib import Path
from threading import Event

from apps.research_single_case import credential_boundary
from apps.vectorengine_case_preflight import check_case_admissibility
from apps.vectorengine_live_assembly import assemble, make_binding, run_case_stages, stop_failed_run
from apps.vectorengine_single_case import load_config, write_once
from db.runtime.repository.planner import PlannerRepository
from db.runtime.repository.research import ResearchRunRepository
from packages.grounding.models import digest, seal, validate
from packages.planner_control.models import Command
from packages.research_runtime.agent_records import RuntimeDocuments
from packages.research_runtime.models import ResearchRunContext

from .case_materials import NewCaseEvidence
from .direct_baseline import CONFIG


def capability_basis(config):
    path = Path("docs/acceptance/12-live-active-acquisition-r02-v1/verification.json")
    value = json.loads(path.read_bytes())
    public = json.loads(
        Path("configs/research/vectorengine_active_acquisition_r02_public.json").read_bytes()
    )
    if value.get("complete") is not True or value["captures"]["public"]["exit_code"] != 0:
        raise ValueError("new_case_public_capability_unqualified")
    if public["provider_checksum"] != digest(config):
        raise ValueError("new_case_provider_capability_changed")
    return dict(evidence=str(path), checksum=digest(value), public_requests_repeated=0)


def requests_for(evidence):
    # Install just the original visible-request/current-catalog port, not the 06 executor.
    # Later platform migrations are not dependencies of this direct research entry.
    from packages.research_runtime import deployment as d

    boundary = credential_boundary()
    target = boundary.resolve_target("runtime")
    config = d.load_configuration(Path("configs/research/fairclip_single_case.json"))
    installed = d.read_installed_bundle(config["installed_bundle"])
    adapter = validate(d.AdapterConfig, installed["adapter_config"])
    resources = d.LocalResourcePolicy(
        allowed_roots=tuple(installed["resource_roots"]), max_bytes=2147483648, timeout_seconds=60
    )
    d.read_record(installed["projection_policy"])
    projection = d.load_projection_policy(Path(installed["projection_policy"]["path"]), d.ROOT)
    from .scoped_catalog import SingleToolSource

    declaration = json.loads(d.read_record(installed["research_review"]))["deployment_declaration"]
    source = SingleToolSource(
        adapter,
        identity=tuple(declaration[k] for k in ("tool_code", "version", "implementation_key")),
        dsn_resolver=lambda: d.medgraph_readonly_dsn(boundary),
    )
    catalog = d.ReviewedCatalog(
        d.MedgraphCatalogReader(
            source, d.SnapshotStore(Path(installed["snapshot_root"])), adapter, projection
        ),
        validate(d.ReviewedFile, installed["research_review"]),
        resources,
    )
    if catalog.current_safety(catalog.catalog.reference()).disabled_implementations:
        raise ValueError("new_case_catalog_currently_disabled")
    requests = d.VisibleResearchRequests(
        runtime=d.ResearchLifecycleRepository(target),
        visible_reader=evidence.visible,
        objects=evidence.objects,
        candidate=validate(d.Candidate, installed["candidate"]),
        invocation=validate(d.InvocationInput, installed["invocation"]),
        catalog_ref=catalog.catalog.reference(),
        policy=validate(d.Policy, installed["compatibility_policy"]),
        input_binding=validate(d.ArtifactTypeBinding, installed["input_type_binding"]),
        modality=validate(d.Term, installed["input_modality"]),
        scope_terms=installed["input_scope_terms"],
        config_checksum=config["checksum"],
    )
    return requests, target


def prepare(sample_id):
    evidence = NewCaseEvidence(sample_id)
    config = load_config(CONFIG)
    capability_basis(config)
    requests, target = requests_for(evidence)
    key = "13-new-case-graph-v1:" + sample_id
    request = requests.register(evidence.visible_value, key)
    if requests.for_key(key) != request or request.scope.case_code != evidence.binding["case_code"]:
        raise ValueError("new_case_request_scope")
    initial, manifest = evidence.project(run_id=request.lock.run_id, config=config)
    # The batch checker excludes fixed slots; check those before the first paid request.
    from packages.evidence_scoring.accepted import check_fixed_model

    observations = {o.reference.code: o for o in initial.material.observations}
    for item in initial.policy.items:
        if item.fixed_model_row is not None:
            check_fixed_model(item, observations[item.observation_code])
    admissibility = check_case_admissibility(initial, config)
    evidence.verify(initial)
    output = evidence.directory / "graph"
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_once(output / "domain-state.json", initial)
    write_once(
        output / "manifest.json",
        dict(
            **manifest,
            admissibility=admissibility,
            config_checksum=digest(config),
            max_requests=40,
            max_extra_requests=3,
            absolute_seconds=2700,
            direct_baseline_budget_separate=1,
            sample_count=2,
            label_access=False,
            frozen_template_complete=True,
            executed_model_components=["SegMAN-B"],
            classification_and_vessel_tools="not_executed_missing_not_old_case_reuse",
        ),
    )
    write_once(output / "request.json", request)
    return evidence, config, request, target, initial, output


def execute(sample_id):
    import fcntl

    evidence, config, request, target, initial, output = prepare(sample_id)
    with (output / "lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "run-result.json").exists():
            return json.loads((output / "run-result.json").read_bytes())
        contexts, sessions = ResearchRunRepository(target), PlannerRepository(target)
        previous = sessions.read(request.lock.run_id)
        if previous is not None and (previous.phase == "stopped" or previous.cancelled):
            raise ValueError("new_case_stopped_no_restart")
        context = seal(
            ResearchRunContext,
            request=request,
            synthetic=False,
            scope_checksum=digest(request.scope),
        )
        contexts.create_research_run(context, "13-new-case-graph-v1:" + sample_id)
        started, context_checksum = sessions.origin(context.run_id)
        binding, ranking = make_binding(
            initial=initial,
            config=config,
            started=started,
            context_checksum=context_checksum,
            catalog=request.lock.catalog_ref,
        )
        documents = RuntimeDocuments(
            repo=evidence.runtime,
            contexts=contexts,
            sessions=sessions,
            store=evidence.objects,
            run_id=context.run_id,
            root=evidence.root,
        )
        planner, owner, ports, review = assemble(
            evidence=evidence,
            initial=initial,
            config=config,
            documents=documents,
            binding=binding,
            ranking=ranking,
            source_quote_policy="research-quote-policy/layout-whitespace/1.0.0",
        )
        cancel = Event()
        ports.budget.cancel_signal = cancel
        if sessions.read(binding.run_id) is None:
            planner.begin(binding)
        planner.require_agent_work(binding)
        # Public compatibility is reused only as capability evidence. New case permission,
        # budget and scope come from this new original Runtime binding, never an old gate.
        write_once(
            output / "capability-basis.json",
            dict(
                **capability_basis(config),
                new_case_scope=initial.material.scope,
            ),
        )
        spec = importlib.util.spec_from_file_location(
            "private_launcher", ".local/vectorengine/run.py"
        )
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        key_name = config.transport.api_key_env
        previous_key = os.environ.get(key_name)
        handlers = {
            s: signal.signal(s, lambda *_: cancel.set()) for s in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            os.environ[key_name] = launcher.load_secret()
            completed, roles = run_case_stages(
                planner=planner,
                owner=owner,
                ports=ports,
                review=review,
                documents=documents,
                binding=binding,
                config=config,
                output=output,
                signal_cancel=cancel,
            )
            current = planner.read(binding.run_id)
            if current.phase != "stopped" and current.pending is None:
                planner.step(
                    binding.run_id,
                    Command(
                        key="new-case-bounded-stop",
                        expected_revision=current.revision,
                        action="stop",
                    ),
                    cancel=cancel,
                )
            result = dict(
                schema_version="new-case-graph-result/1.0.0",
                run_id=binding.run_id,
                case_code=binding.case_code,
                complete=len(completed) == len(roles)
                and all(r["status"] == "accepted" for r in completed),
                stages=completed,
                request_summary=ports.budget.summary(),
                transport_counts=ports.budget.transport_counts(),
                planner=planner.report(binding.run_id),
                new_tool_executions=1,
                clinical_acceptance=False,
            )
            write_once(output / "run-result.json", result)
            return result
        except Exception as error:
            failure = stop_failed_run(
                planner=planner,
                sessions=sessions,
                binding=binding,
                budget=ports.budget,
                error=error,
                cancel=cancel,
            )
            write_once(output / "failure.json", failure)
            raise
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
            if previous_key is None:
                os.environ.pop(key_name, None)
            else:
                os.environ[key_name] = previous_key
