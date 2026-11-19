"""Stage commands composing original campaign methods; no second executor/ledger.

Local inspections never assemble owners. Receipts are immutable Runtime mirrors;
only original owners authorize consumption. Review includes atomic acceptance.
"""

import fcntl
import importlib.util
import json
import os
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.grounding.models import canonical, digest, validate

from .implementation import freeze
from .report_io import mirror_state, publish_json
from .stage_models import LocalReviewClosure, StagePlan, StageReceipt

STAGES = ("sources", "definitions", "requirements", "acquisition", "review", "acceptance", "export")
KEY = "multimodal-stage-operation:"


def read(path):
    return json.loads(path.read_bytes()) if path.is_file() else None


def inspect_local(output):
    out = Path(output).absolute()
    errors = []

    def diagnostic(path, required, model=None):
        if not path.is_file():
            return None
        raw = path.read_bytes()
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            reason = "invalid_local_json"
        else:
            if isinstance(value, dict) and all(k in value for k in required):
                try:
                    if model is not None:
                        validate(model, value)
                except ValueError:
                    pass
                else:
                    return value
            reason = "invalid_local_structure"
        from packages.object_store import checksum

        errors.append(
            dict(
                file=str(path.relative_to(out)),
                bytes=len(raw),
                checksum=checksum(raw),
                reason=reason,
            )
        )
        return None

    protocol = read(out / "protocol.json")
    if protocol is None:
        raise ValueError("stage_protocol_missing")
    deadline = datetime.fromisoformat(protocol["deadline"])
    plan = diagnostic(out / "stage-plan.json", ("protocol_checksum", "implementation"), StagePlan)
    receipts = {
        s: diagnostic(out / "stages" / (s + ".json"), ("run_id",), StageReceipt) for s in STAGES
    }
    failure = diagnostic(
        out / "stage-failure.json",
        ("stage", "failure_kind", "error_code", "control", "automatic_replay_authorized"),
    )
    closure = diagnostic(out / "stage-closure-error.json", ("stage",))
    return dict(
        schema_version="multimodal-stage-status/1.1.0",
        read_only=True,
        execution_authorized=False,
        control="not_checked_local_view",
        protocol_checksum=digest(protocol),
        plan_matches=bool(
            plan
            and plan["protocol_checksum"] == digest(protocol)
            and plan["implementation"] == freeze()
        ),
        deadline=protocol["deadline"],
        expired=datetime.now(UTC) >= deadline,
        cancelled=(out / "cancel").exists(),
        stages={
            s: (dict(run_id=r["run_id"], checksum=digest(r)) if r else None)
            for s, r in receipts.items()
        },
        failure=(
            {
                k: failure[k]
                for k in (
                    "stage",
                    "failure_kind",
                    "error_code",
                    "control",
                    "automatic_replay_authorized",
                )
            }
            if failure
            else None
        ),
        local_closure_error=closure,
        local_mirror_errors=errors,
        automatic_replay_authorized=False,
        next_stage=None if errors else next((s for s, r in receipts.items() if r is None), None),
        note="Local mirrors are not current source, Control or execution authorization",
    )


def prepare(output):
    out = Path(output).absolute()
    status = inspect_local(out)
    protocol = read(out / "protocol.json")
    if (
        status["expired"]
        or status["cancelled"]
        or status["failure"]
        or status["local_closure_error"]
        or status["local_mirror_errors"]
    ):
        raise ValueError("stage_previous_failure_read_only")
    if protocol.get("implementation") != freeze():
        raise ValueError("stage_implementation_changed")
    value = dict(
        schema_version="multimodal-stage-plan/1.0.0",
        protocol_checksum=digest(protocol),
        implementation=freeze(),
        stages=list(STAGES),
        deadline=protocol["deadline"],
        max_new_requests=protocol["max_new_requests"],
        budget_owner="original_campaign_and_RuntimeAgentJournal",
        acceptance_is_inside_review=True,
    )
    value = validate(StagePlan, value).model_dump(mode="json")
    write_once(out / "stage-plan.json", value)
    return value


def check_plan(campaign):
    value = read(campaign.output / "stage-plan.json")
    if value:
        validate(StagePlan, value)
    expected = freeze()
    if (campaign.output / "execution-lease.json").exists():
        from .execution_lease import read as read_lease

        amendment = read_lease(campaign)
        if amendment["implementation_after"] != expected:
            raise ValueError("stage_lease_implementation_changed")
        expected = amendment["implementation_before"]
    if (
        not value
        or value["protocol_checksum"] != digest(campaign.protocol)
        or value["implementation"] != expected
    ):
        raise ValueError("stage_plan_binding")
    if hasattr(campaign, "propositions"):
        raise ValueError("stage_requires_one_explicit_proposition")


def verify(campaign, stage):
    """Read-only durable receipt verification, not current consumption authorization."""
    check_plan(campaign)
    value = durable_receipt(campaign, stage)
    if value:
        # Historical confirmation only: a proven prefix is not a new work grant.
        mirror_state(
            campaign.output / "stages" / (stage + ".json"),
            canonical(value) + b"\n",
            kind="json",
        )
    if not value or value["binding"] != digest(campaign.protocol):
        raise ValueError("stage_precondition")
    if campaign.documents(value["run_id"]).read(KEY + stage) != (value,):
        raise ValueError("stage_owner_binding")
    return value


def durable_receipt(campaign, stage):
    from db.runtime.repository.case_pipeline import identity

    target = "definition-expansion" if stage in STAGES[:4] else None
    run_id = identity("RUN", "task-research:" + campaign.request_key(target))
    if campaign.contexts.read_research_run(run_id, missing_ok=True) is None:
        return None
    values = campaign.documents(run_id).read(KEY + stage)
    if not values:
        return None
    value = validate(StageReceipt, values[0]).model_dump(mode="json")
    if (
        value["binding"] != digest(campaign.protocol)
        or value["stage"] != stage
        or value["run_id"] != run_id
    ):
        raise ValueError("stage_owner_binding")
    return value


@contextmanager
def credentials(campaign, needed):
    if not needed or campaign.synthetic:
        yield
        return
    campaign.guard()
    spec = importlib.util.spec_from_file_location("private_launcher", ".local/vectorengine/run.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    name = campaign.config.transport.api_key_env
    previous = os.environ.get(name)
    try:
        os.environ[name] = launcher.load_secret()
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def seal(campaign, stage, run_id, payload):
    docs = campaign.documents(run_id)
    value = json.loads(
        canonical(
            dict(
                schema_version="multimodal-stage-receipt/1.0.0",
                binding=digest(campaign.protocol),
                stage=stage,
                run_id=run_id,
                payload=payload,
            )
        )
    )
    value = validate(StageReceipt, value).model_dump(mode="json")
    with docs.session():
        docs.write(KEY + stage, (value,))
    if docs.read(KEY + stage) != (value,):
        raise ValueError("stage_owner_binding")
    directory = campaign.output / "stages"
    directory.mkdir(exist_ok=True)
    publish_json(directory / (stage + ".json"), value)
    return value


def run(campaign, stage, *, definition_ports=None):
    context = getattr(campaign, "phase_context", lambda target: nullcontext())
    with context(None if stage == "review" else stage):
        return _run(campaign, stage, definition_ports=definition_ports)


def confirmed_phase(campaign, target):
    """Read the original committed phase, never register or dispatch a request."""
    from db.runtime.repository.case_pipeline import identity

    run_id = identity("RUN", "task-research:" + campaign.request_key(target))
    if campaign.contexts.read_research_run(run_id, missing_ok=True) is None:
        return None
    docs = campaign.documents(run_id)
    values = docs.read("dynamic-phase-result")
    if not values:
        return None
    value = values[0]
    if (
        len(values) != 1
        or value["run_id"] != run_id
        or value["target"] != target
        or docs.read("dynamic-campaign") != (campaign.protocol,)
    ):
        raise ValueError("stage_phase_binding")
    try:
        mirror_state(
            campaign.phase_path(target) / "result.json", canonical(value) + b"\n", kind="json"
        )
    except ValueError as error:
        if str(error) == "local_report_content_conflict":
            raise ValueError("stage_phase_binding") from error
        raise
    return value


def acceptance_payload(campaign, review):
    docs = campaign.documents(review["run_id"])
    if review.get("schema_version") == "dynamic-phase-local-closure/1.0.0":
        if review_for_export(campaign) != review:
            raise ValueError("stage_owner_binding")
    elif docs.read("dynamic-phase-result") != (review,):
        raise ValueError("stage_owner_binding")
    accepted = docs.read("accepted:" + review["acceptance_key"]) if review["accepted"] else None
    if review["accepted"] and not accepted:
        raise ValueError("stage_acceptance_missing")
    return dict(
        review_checksum=digest(review),
        accepted_checksum=digest(accepted),
        score_computed=accepted is not None,
        atomic_owner="ResearchAcceptance inside original Review",
    )


def review_for_export(campaign):
    """Historical commit projection; never overwrites a failed phase summary."""
    from agents.expert_agent.committed_review import read_committed_review
    from db.runtime.repository.case_pipeline import identity
    from packages.evidence_details.acceptance_models import DomainState

    original = confirmed_phase(campaign, None)
    if original and (original["accepted"] or original["evidence_limited"]):
        return original
    rid = identity("RUN", "task-research:" + campaign.request_key(None))
    if campaign.contexts.read_research_run(rid, missing_ok=True) is None:
        return original
    docs = campaign.documents(rid)
    parents = docs.read("dynamic-phase-input")
    if not parents:
        return original
    if docs.read("dynamic-campaign") != (campaign.protocol,):
        raise ValueError("stage_phase_binding")
    proof = read_committed_review(
        docs,
        parent=validate(DomainState, parents[0]),
        config=campaign.config,
        confirmed_review_reader=getattr(campaign.base, "confirmed_review_reader", None),
    )
    if proof is None:
        return original
    return dict(
        original
        or dict(
            run_id=rid,
            target=None,
            requests=None,
            transport=None,
            error="local_phase_summary_missing",
            evidence_limited=False,
            clinical_acceptance=False,
        ),
        schema_version="dynamic-phase-local-closure/1.0.0",
        accepted=True,
        acceptance_key=proof["acceptance_key"],
        closure=LocalReviewClosure(
            original_phase=original, original_phase_checksum=digest(original), confirmation=proof
        ).model_dump(mode="json"),
    )


@contextmanager
def recovered_report(campaign, review):
    """Separate names for an old failed phase/result; always restore attributes."""
    values = dict(local_export_recovery=True)
    if review.get("schema_version") == "dynamic-phase-local-closure/1.0.0":
        values.update(
            result_key="multimodal-stage-recovered-case-result",
            result_filename="stage-recovered-result.json",
            report_key="multimodal-stage-recovered-case-report",
            report_name=campaign.output.name + "_local-recovery",
            local_review_closure=review["closure"],
        )
    missing = object()
    previous = {key: getattr(campaign, key, missing) for key in values}
    try:
        for key, value in values.items():
            setattr(campaign, key, value)
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                delattr(campaign, key)
            else:
                setattr(campaign, key, value)


def local_current(campaign, definition_ports):
    """No deadline extension or dispatch: recheck inputs, sources and saved facts."""
    campaign.expand_definitions(restore_only=True, report_only=True, **(definition_ports or {}))
    campaign.base.check()
    if hasattr(campaign, "procedure_reuse"):
        campaign.procedure_reuse.check()
    campaign.accepted_observations()
    review = review_for_export(campaign)
    if review:
        from packages.evidence_details.acceptance_models import DomainState

        rid = review["run_id"]
        campaign.base.contexts[rid] = campaign.contexts.read_research_run(rid)
        parent = validate(DomainState, campaign.documents(rid).read("dynamic-phase-input")[0])
        phase = campaign.phase_class(
            campaign.base,
            target=None,
            observation_reader=campaign.accepted_observations,
            model_reader=campaign.read_model,
        )
        expected, _ = phase.project(run_id=rid, config=campaign.config)
        if expected != parent:
            raise ValueError("stage_recovery_current_parent_changed")
        phase.verify(parent)
        control = campaign.sessions.read(rid)
        if control is None or control.pending is not None or control.phase != "stopped":
            raise ValueError("stage_recovery_control_unconfirmed")


def recover_export(campaign, *, definition_ports=None):
    """Explicit zero-request closure; immutable failure remains beside recovery."""
    from .stage_failure import read_saved_failure

    failure = read_saved_failure(campaign)
    closure = None if failure else read(campaign.output / "stage-closure-error.json")
    origin = failure or closure
    if not origin or origin["stage"] not in {"review", "acceptance", "export"}:
        raise ValueError("stage_previous_failure_read_only")
    if failure:
        if failure["protocol_checksum"] != digest(campaign.protocol) or (
            failure["run_id"]
            and campaign.documents(failure["run_id"]).read("multimodal-stage-failure") != (failure,)
        ):
            raise ValueError("stage_failure_owner_binding")
    local_current(campaign, definition_ports)
    review = review_for_export(campaign)
    if not review or not (review["accepted"] or review["evidence_limited"]):
        raise ValueError("stage_recovery_requires_confirmed_review")
    acquisition = verify(campaign, "acquisition")["payload"]["phases"]
    for phase in acquisition:
        if confirmed_phase(campaign, phase["target"]) != phase:
            raise ValueError("stage_phase_binding")
    accepted = acceptance_payload(campaign, review)
    if review.get("closure"):
        docs = campaign.documents(review["run_id"])
        with docs.session():
            docs.write("multimodal-review-local-closure", (review["closure"],))
    # Missing stage acknowledgements do not mean missing phase/score commits.
    for stage, payload in (("review", review), ("acceptance", accepted)):
        prior = durable_receipt(campaign, stage)
        if prior:
            if prior["payload"] != payload:
                raise ValueError("stage_owner_binding")
        else:
            seal(campaign, stage, review["run_id"], payload)
    before = campaign.usage()
    with recovered_report(campaign, review):
        payload = campaign.finish(acquisition + [review])
    if campaign.usage() != before:
        raise ValueError("stage_local_recovery_request_changed")
    payload = dict(
        payload,
        recovery=dict(
            schema_version="multimodal-stage-export-recovery/1.0.0",
            failure_checksum=digest(origin),
            score_checksum=accepted["accepted_checksum"],
            added_requests=0,
            historical_failure_preserved=True,
        ),
    )
    return seal(campaign, "export", review["run_id"], payload)


@contextmanager
def stage_execution_session(campaign):
    """Use the protocol's original case lease, not an extra unsupported NFS lock."""
    if getattr(campaign, "protocol", {}).get("schema_version") in {
        "multimodal-campaign-renewal/1.79.0",
        "multimodal-campaign-renewal/1.80.0",
    } or (
        getattr(campaign, "protocol", {}).get("schema_version")
        in {"multimodal-live-campaign/1.20.0", "multimodal-live-campaign/1.21.0"}
        and campaign.protocol.get("entry_execution_lock") == "multimodal-runtime-entry/1.0.0"
    ):
        with campaign.execution_session():
            yield
    else:
        with (campaign.output / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield


def _run(campaign, stage, *, definition_ports=None):
    """One stage, same original guards and irreversible operation keys. No retries."""
    if stage not in STAGES:
        raise ValueError("stage_unknown")
    with stage_execution_session(campaign):
        campaign.freeze()
        check_plan(campaign)
        # A missing local mirror is not a lost commit. Check the original owner
        # before any guard/provider work; returned receipt is historical, not a grant.
        committed = durable_receipt(campaign, stage)
        if committed:
            return verify(campaign, stage)
        if (campaign.output / "stage-failure.json").exists() or (
            campaign.output / "stage-closure-error.json"
        ).exists():
            if stage == "export":
                return recover_export(campaign, definition_ports=definition_ports)
            raise ValueError("stage_previous_failure_read_only")
        prior = read(campaign.output / "stages" / (stage + ".json"))
        if prior:
            return verify(campaign, stage)
        if stage != "sources":
            verify(campaign, STAGES[STAGES.index(stage) - 1])
        # Guard BEFORE loading secrets and before registering even the definition run.
        saved_review = confirmed_phase(campaign, None) if stage == "review" else None
        try:
            if (
                not saved_review
                and stage not in {"acceptance", "export"}
                and (campaign.output / "cancel").exists()
            ):
                raise ValueError("cancelled")
            if not saved_review and stage not in {"acceptance", "export"}:
                campaign.guard()
        except Exception as error:
            from .stage_failure import close

            try:
                close(campaign, error, stage=stage)
            except Exception as closure_error:
                # A local reporting failure must not replace the dispatch refusal.
                # Retain the secondary cause without pretending closure was saved.
                raise error from closure_error
            raise  # Preserve the original refusal and nonzero CLI exit.
        try:
            if getattr(campaign, "scope_targets", ()):
                from .input_scope import execute as execute_input_scope
                from .input_scope import read_results as read_input_scopes

                if stage == "sources":
                    pending_scope = any(
                        read(campaign.phase_path(t) / "result.json") is None
                        for t in campaign.scope_targets
                    )
                    with credentials(campaign, pending_scope):
                        execute_input_scope(campaign)
                else:
                    # Later stage commands cannot silently skip the same intake
                    # that the full-run entry performs before source expansion.
                    read_input_scopes(campaign, require_all=True)
            from .scope_selection import ensure

            ensure(campaign, allow_native=stage in {"sources", "acquisition"})
            if stage in STAGES[:3]:
                with credentials(campaign, stage == "definitions"):
                    payload = campaign.expand_definitions(until=stage, **(definition_ports or {}))
                rid = campaign.definition_expansion.documents.run_id
            else:
                # Reinstall the exact persisted revision, no search/model call.
                campaign.expand_definitions(
                    restore_only=True,
                    report_only=bool(saved_review) or stage in {"acceptance", "export"},
                    **(definition_ports or {}),
                )
                if stage == "acquisition":
                    phases = []
                    for target in campaign.targets:
                        if campaign.acquisition_exhausted() or campaign.active_review_reserved():
                            break
                        saved = read(campaign.phase_path(target) / "result.json")
                        with credentials(campaign, saved is None):
                            phase = campaign.execute_phase(target)
                        phases.append(phase)
                        if phase["error"] and not phase["accepted"]:
                            from .observation_closure import close_observation

                            if not close_observation(
                                campaign, phase
                            ) and not campaign.local_acquisition_gap(phase):
                                raise ValueError("stage_acquisition_not_confirmed")
                            if hasattr(campaign, "observation_closure"):
                                break
                    payload = dict(phases=phases)
                    rid = campaign.definition_expansion.documents.run_id
                elif stage == "review":
                    if saved_review:
                        local_current(campaign, definition_ports)
                        payload = saved_review
                    else:
                        with credentials(campaign, True):
                            payload = campaign.execute_phase()
                    if not (payload["accepted"] or payload["evidence_limited"]):
                        raise ValueError("stage_review_not_accepted")
                    rid = payload["run_id"]
                elif stage == "acceptance":
                    review = verify(campaign, "review")["payload"]
                    payload = acceptance_payload(campaign, review)
                    rid = review["run_id"]
                else:
                    phases = verify(campaign, "acquisition")["payload"]["phases"] + [
                        verify(campaign, "review")["payload"]
                    ]
                    payload = campaign.finish(phases)
                    rid = phases[-1]["run_id"]
            return seal(campaign, stage, rid, payload)
        except Exception as error:
            try:
                committed = durable_receipt(campaign, stage)
            except Exception:
                committed = None  # Uncertain commit: technical closure, never resend.
            if committed:
                return committed
            from .stage_failure import close

            return close(campaign, error, stage=stage)


def assemble_for_stage(output):
    """No initial native tool acquisition as a side effect of a stage command.

    A uses existing owner-authorized successor preparation. Initial native/tool
    preparation remains its original explicit command, not hidden in status.
    """
    out = Path(output).absolute()
    protocol = read(out / "protocol.json") or {}
    if protocol.get("schema_version") == "multimodal-live-campaign/1.21.0":
        from .entry import assemble_entry

        if not scoped_entry(out):
            raise ValueError("stage_scope_entry_binding")
        campaign = assemble_entry(out)
        campaign.freeze()  # Revalidate/load the exact existing protocol before verify/export.
        return campaign
    if protocol.get("schema_version") in {
        "multimodal-live-campaign/1.5.0",
        "multimodal-live-campaign/1.6.0",
        "multimodal-live-campaign/1.7.0",
        "multimodal-live-campaign/1.8.0",
        "multimodal-live-campaign/1.9.0",
    }:
        from .definition_trial import cold

        return cold(out)
    if not protocol.get("schema_version", "").startswith("multimodal-campaign-renewal/"):
        raise ValueError("stage_cli_requires_prepared_renewal")
    from .campaign_renewal import assemble

    return assemble(out)


def scoped_entry(output):
    """Select an explicit new entry, not authority to execute or skip owners."""
    from .active_requirements import load
    from .execution_lease import ENTRY_PROFILE, validate_entry

    out = Path(output)
    protocol = read(out / "protocol.json")
    if protocol and protocol.get("schema_version") != "multimodal-live-campaign/1.21.0":
        return False  # Legacy stage assembly/verification keeps its own contract.
    spec = read(out / "entry.json") or {}
    ref = spec.get("active_acquisition")
    if not ref:
        return False
    profile = load(ref["path"])
    if not profile.get("scope_selection_policy"):
        return False
    if digest(profile) != ref["checksum"]:
        raise ValueError("stage_scope_profile_changed")
    if not validate_entry(spec) or spec["entry_execution_lock"]["schema_version"] != ENTRY_PROFILE:
        raise ValueError("stage_scope_original_entry_lease_required")
    return True


DEFINITION_REPAIR_KEY = "renewal:definition-review-repair"


def definition_repair_delta(before, after):
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    if not changed or not changed <= {
        "packages/multimodal_research/definition_expansion.py",
        "packages/multimodal_research/campaign_renewal.py",
        "packages/multimodal_research/stages.py",
        "apps/multimodal_case.py",
    }:
        raise ValueError("definition_review_repair_change_scope")
    return sorted(changed)


def read_definition_review_repair(campaign):
    """Exact additive code amendment; never alter the failed window or its budget."""
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    from .stage_failure import read_saved_failure

    value = read(campaign.output / "definition-review-repair.json")
    if not value or value.get("schema_version") not in {
        "definition-review-repair/1.0.0",
        "definition-review-repair/1.1.0",
    }:
        raise ValueError("definition_review_repair_missing")
    docs = campaign.documents(value["run_id"])
    proof = read(Path(value["proof_path"]))
    if (
        value["protocol_checksum"] != digest(campaign.protocol)
        or value["implementation_before"] != campaign.protocol["implementation"]
        or value["deadline"] != campaign.protocol["deadline"]
        or value["max_new_requests"] != campaign.protocol["max_new_requests"]
        or value["failure_checksum"] != digest(read_saved_failure(campaign))
        or docs.read(DEFINITION_REPAIR_KEY) != (value,)
        or digest(docs.read("definition:revision")) != value["revision_checksum"]
        or digest(RuntimeAgentJournal(docs).read()) != value["source_journal_checksum"]
        or not proof
        or proof.get("direct_exit") != 0
        or proof.get("implementation") != value["implementation_after"]
        or digest(proof) != value["proof_checksum"]
    ):
        raise ValueError("definition_review_repair_binding")
    definition_repair_delta(value["implementation_before"], value["implementation_after"])
    return value


def definition_repair_eligible(campaign, failure, revision, *, include_acquisition=False):
    if include_acquisition:
        from .campaign_renewal import CONFIRMED_DEFINITION_FAILURE_VERSION

        selected = {r["id"] for r in revision.get("replaced", [])}
        if (
            campaign.protocol.get("schema_version") != CONFIRMED_DEFINITION_FAILURE_VERSION
            or len(campaign.targets) != 1
            or not set(campaign.targets) <= selected
            or campaign.protocol["max_new_requests"] - campaign.usage() < 2
        ):
            raise ValueError("definition_acquisition_repair_scope")
    if (
        campaign.protocol.get("mode") != "definitions"
        or not campaign.protocol.get("definition_increment_only")
        or not failure
        or failure["stage"] != "definitions"
        or failure["error_code"] != "local_owner_error"
        or not failure["definition_revision_saved"]
        or failure["transport"]["remote_response_unknown"]
        or failure["transport"]["dispatch_unknown"]
        or revision["added"]
        or not revision["replaced"]
        or (campaign.targets and not include_acquisition)
    ):
        raise ValueError("definition_review_repair_scope")


@contextmanager
def definition_repair_session(campaign, *, include_acquisition):
    if include_acquisition:
        # This version already mandates the original Runtime advisory lease.
        # Do not add an unavailable NFS flock or fall back to no locking.
        from .campaign_renewal import CONFIRMED_DEFINITION_FAILURE_VERSION

        if campaign.protocol.get("schema_version") != CONFIRMED_DEFINITION_FAILURE_VERSION:
            raise ValueError("definition_acquisition_repair_lease_version")
        with campaign.execution_session():
            yield
    else:
        with (campaign.output / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield


def resume_definition_review(campaign, proof_path, authorization, *, include_acquisition=False):
    """Explicit single Review continuation after a saved-definition projection fault.

    The original Source result/revision/failure stay immutable. No Source Provider
    is constructed. An intent commits before dispatch; an interrupted/UNKNOWN
    attempt is never automatically repeated. This is not the ordinary resume path.
    """
    from db.runtime.repository.case_pipeline import identity
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    from .stage_failure import read_saved_failure, safe_error

    if not authorization or not authorization.strip():
        raise ValueError("definition_review_repair_authorization")
    with definition_repair_session(campaign, include_acquisition=include_acquisition):
        with campaign.phase_context(None):
            path = campaign.output / "definition-review-repair.json"
            if not path.exists():
                campaign.guard()
                failure = read_saved_failure(campaign)
                before, after = campaign.protocol["implementation"], freeze()
                changed = definition_repair_delta(before, after)
                proof_path = Path(proof_path).absolute()
                proof = read(proof_path)
                if (
                    not proof
                    or proof.get("direct_exit") != 0
                    or proof.get("implementation") != after
                ):
                    raise ValueError("definition_review_repair_test_proof")
                rid = identity("RUN", "task-research:" + campaign.request_key(None))
                if campaign.contexts.read_research_run(rid, missing_ok=True) is not None:
                    raise ValueError("definition_review_repair_review_already_started")
                revision = campaign.expand_definitions(restore_only=True, report_only=True)
                definition_repair_eligible(
                    campaign, failure, revision, include_acquisition=include_acquisition
                )
                docs = campaign.definition_expansion.documents
                if docs.run_id != failure["run_id"]:
                    raise ValueError("definition_review_repair_source_binding")
                value = dict(
                    schema_version="definition-review-repair/1.1.0"
                    if include_acquisition
                    else "definition-review-repair/1.0.0",
                    run_id=docs.run_id,
                    protocol_checksum=digest(campaign.protocol),
                    implementation_before=before,
                    implementation_after=after,
                    changed=changed,
                    proof_path=str(proof_path),
                    proof_checksum=digest(proof),
                    failure_checksum=digest(failure),
                    revision_checksum=digest((revision,)),
                    source_journal_checksum=digest(RuntimeAgentJournal(docs).read()),
                    deadline=campaign.protocol["deadline"],
                    max_new_requests=campaign.protocol["max_new_requests"],
                    authorization=authorization,
                    only_next_operation="original_acquisition_then_review_then_local_finish"
                    if include_acquisition
                    else "original_review_then_local_finish",
                    **(
                        {"acquisition_targets": list(campaign.targets)}
                        if include_acquisition
                        else {}
                    ),
                )
                with docs.session():
                    docs.write(DEFINITION_REPAIR_KEY, (value,))
                write_once(path, value)
            value = read_definition_review_repair(campaign)
            if include_acquisition != (value["schema_version"] == "definition-review-repair/1.1.0"):
                raise ValueError("definition_acquisition_repair_explicit_scope")
            campaign.freeze()
            campaign.expand_definitions(restore_only=True, report_only=True)
            docs = campaign.documents(value["run_id"])
            acquisition_phases = []
            if include_acquisition:
                if list(campaign.targets) != value["acquisition_targets"]:
                    raise ValueError("definition_acquisition_repair_targets_changed")
                for target in campaign.targets:
                    with campaign.phase_context(target):
                        acquisition = confirmed_phase(campaign, target)
                        acquisition_key = DEFINITION_REPAIR_KEY + ":acquisition:" + target
                        if acquisition is None:
                            target_run = identity(
                                "RUN", "task-research:" + campaign.request_key(target)
                            )
                            if (
                                docs.read(acquisition_key) is not None
                                or campaign.contexts.read_research_run(target_run, missing_ok=True)
                                is not None
                            ):
                                raise ValueError(
                                    "definition_acquisition_repair_requires_confirmation"
                                )
                            campaign.guard()
                            with docs.session():
                                docs.write(
                                    acquisition_key, (dict(amendment_checksum=digest(value)),)
                                )
                            with credentials(campaign, True):
                                acquisition = campaign.execute_phase(target)
                        if not acquisition["accepted"]:
                            raise ValueError("definition_acquisition_repair_not_accepted")
                        acquisition_phases.append(acquisition)
            key = DEFINITION_REPAIR_KEY + ":review-intent"
            phase = confirmed_phase(campaign, None)
            if phase is None:
                if docs.read(key) is not None:
                    raise ValueError("definition_review_repair_attempt_requires_confirmation")
                campaign.guard()
                # A code amendment is not permission to reissue an existing run.
                rid = identity("RUN", "task-research:" + campaign.request_key(None))
                if campaign.contexts.read_research_run(rid, missing_ok=True) is not None:
                    raise ValueError("definition_review_repair_review_already_started")
                with docs.session():
                    docs.write(key, (dict(amendment_checksum=digest(value)),))
                try:
                    with credentials(campaign, True):
                        phase = campaign.execute_phase()
                except Exception as error:
                    publish_json(
                        campaign.output / "definition-review-recovery-error.json",
                        dict(
                            error=safe_error(error),
                            original_failure_preserved=True,
                            automatic_replay_authorized=False,
                        ),
                    )
                    raise
            if not (phase["accepted"] or phase["evidence_limited"]):
                raise ValueError("definition_review_repair_not_accepted")
            # Same original scoring/report owners, same budget, no repeated Source.
            return campaign.finish([*acquisition_phases, phase])


def cli(args):
    operation = args.command.removeprefix("stage-")
    if operation in {"status", "check", "preview"}:
        status = inspect_local(args.input)
        if operation == "preview":
            status["operations"] = list(STAGES)
        return status
    if scoped_entry(args.input):
        from .entry import assemble_entry
        from .execution_lease import entry_session

        with entry_session(args.input):
            if operation == "prepare":
                # Initial stage preparation may register the original input and
                # freeze its campaign. It never dispatches a model/native tool.
                if any(
                    (Path(args.input) / name).exists()
                    for name in ("cancel", "stage-failure.json", "stage-closure-error.json")
                ):
                    raise ValueError("stage_previous_failure_read_only")
                if (Path(args.input) / "protocol.json").exists():
                    prepare(args.input)  # Refuse expired/drifted windows before assembly.
                assemble_entry(args.input, preflight_only=True)
                campaign = assemble_entry(args.input)
                campaign.freeze()
                if campaign.protocol["schema_version"] != "multimodal-live-campaign/1.21.0":
                    raise ValueError("stage_scope_protocol_binding")
                return prepare(args.input)
            return execute_cli(args, operation)
    return execute_cli(args, operation)


def execute_cli(args, operation):
    if operation == "prepare":
        return prepare(args.input)
    campaign = assemble_for_stage(args.input)
    if operation == "resume" and getattr(args, "definition_repair_proof", None):
        if args.stage != "review":
            raise ValueError("definition_review_repair_requires_review")
        return resume_definition_review(
            campaign,
            args.definition_repair_proof,
            args.authorization,
            include_acquisition=getattr(args, "include_unstarted_acquisition", False),
        )
    if operation == "verify":
        return verify(campaign, args.stage)
    if operation == "export":
        if (campaign.output / "stage-failure.json").exists():
            from .stage_failure import export_saved, read_saved_failure

            failure = read_saved_failure(campaign)
            if failure["stage"] not in {"review", "acceptance", "export"}:
                return export_saved(campaign)
        return run(campaign, "export")
    return run(campaign, args.stage)
