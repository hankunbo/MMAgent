"""Explicit pre-dispatch lease repair over the original Runtime session owner.

No silent flock fallback, new database or restarted deadline. An immutable parent
receipt binds the same prepared campaign to its reviewed implementation delta.
"""

import errno
import fcntl
import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import CasePipelineRepository, identity
from packages.grounding.models import digest

from .implementation import freeze

PATH = "execution-lease.json"
VERSION = "renewal-execution-lease/1.0.0"
PROFILE = "original-runtime-case-pipeline-session/1.0.0"
AGGREGATE_VERSION = "multimodal-campaign-renewal/1.26.0"
ALLOWED = {
    "packages/multimodal_research/campaign.py",
    "packages/multimodal_research/campaign_renewal.py",
    "packages/multimodal_research/execution_lease.py",
    "apps/multimodal_case.py",
}

ENTRY_PROFILE = "multimodal-runtime-entry/1.0.0"
SOURCE_PROFILE = "multimodal-runtime-source-backlog/1.0.0"
ENTRY_FILES = (
    "packages/multimodal_research/execution_lease.py",
    "packages/multimodal_research/entry.py",
    "packages/multimodal_research/entry_preflight.py",
    "packages/multimodal_research/campaign.py",
    "packages/multimodal_research/native_fundus.py",
    "packages/evaluation_preparation/case_execution.py",
    "packages/evaluation_preparation/generic_execution_lock.py",
    "apps/multimodal_case.py",
    "db/runtime/repository/case_pipeline.py",
)
ROOT = Path(__file__).resolve().parents[2]
_ENTRY = ContextVar("multimodal_original_entry_lease", default=None)


def entry_implementation():
    from packages.object_store import checksum

    return {p: checksum((ROOT / p).read_bytes()) for p in ENTRY_FILES}


def validate_entry(spec):
    value = spec.get("entry_execution_lock")
    if value is None:
        return False
    source = value.get("schema_version") == SOURCE_PROFILE
    reference = spec.get("definition_failure_of")
    if (
        spec.get("schema_version")
        != ("multimodal-live-entry/1.30.0" if source else "multimodal-live-entry/1.29.0")
        or spec.get("execution_profile") != "relay-root-question-v1"
        or any(
            spec.get(p)
            for p in (
                "previous",
                "continuation_of",
                "review_from",
                "active_continuation_of",
                "discovery_continuation_of",
            )
        )
        or (
            source
            and (
                not isinstance(reference, dict)
                or reference.get("mode") != "routed_root_source_backlog_no_source_replay"
                or not all(reference.get(k) for k in ("parent", "entry", "protocol", "failure"))
            )
        )
        or (not source and reference is not None)
        or value
        != dict(
            schema_version=SOURCE_PROFILE if source else ENTRY_PROFILE,
            implementation=entry_implementation(),
        )
    ):
        raise ValueError("multimodal_entry_execution_lock_binding")
    return True


@contextmanager
def historical_entry_probe(output, spec):
    """Probe the original parent exclusion, not authorization to replay its run.

    Historical implementation bytes may differ from the repaired implementation.
    Preserve the old binding; subsequent original domain/journal checks still
    prove the terminal failure and unique successor claim.
    """
    from packages.object_store import checksum

    out = Path(output).resolve()
    value = spec.get("entry_execution_lock")
    if value is None:
        with (out / "lock").open("r") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        return
    expected = {
        p: spec.get("implementation", {}).get(p, checksum((ROOT / p).read_bytes()))
        for p in ENTRY_FILES
    }
    if (
        spec.get("schema_version") != "multimodal-live-entry/1.29.0"
        or spec.get("execution_profile") != "relay-root-question-v1"
        or spec.get("definition_failure_of")
        or value != dict(schema_version=ENTRY_PROFILE, implementation=expected)
        or json.loads((out / "entry.json").read_bytes()) != spec
    ):
        raise ValueError("historical_entry_lock_binding")
    repository = CasePipelineRepository(entry_target())
    with repository.guarded_session(
        "CASE:MULTIMODAL_ENTRY", "multimodal-entry:" + digest(str(out))[7:]
    ) as guard:
        guard()
        yield
        guard()
        if json.loads((out / "entry.json").read_bytes()) != spec:
            raise ValueError("historical_entry_lock_input_changed")


def check_current(output=None):
    active = _ENTRY.get()
    if active is None:
        return None
    if output is not None and Path(output).resolve() != active.root:
        raise ValueError("multimodal_entry_execution_lock_scope")
    if json.loads((active.root / "entry.json").read_bytes()) != active.spec:
        raise ValueError("multimodal_entry_execution_lock_input_changed")
    validate_entry(active.spec)
    for guard in active.guards:
        guard()
    return active


def require_entry(output, spec):
    if validate_entry(spec) and check_current(output) is None:
        raise ValueError("multimodal_entry_original_execution_lease_required")


def entry_target():
    from apps.research_single_case import credential_boundary

    return credential_boundary().resolve_target("runtime")


@contextmanager
def entry_session(output):
    out = Path(output).resolve()
    try:
        spec = json.loads((out / "entry.json").read_bytes())
    except (FileNotFoundError, json.JSONDecodeError):
        # Preserve the original command's safe, append-only invalid-input report.
        spec = {}
    if not validate_entry(spec):
        with (out / "assembly.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        return
    if _ENTRY.get() is not None:
        raise ValueError("multimodal_entry_execution_lock_nested")
    repository = CasePipelineRepository(entry_target())
    key = "multimodal-entry:" + digest(str(out))[7:]
    with repository.guarded_session("CASE:MULTIMODAL_ENTRY", key) as guard:
        token = _ENTRY.set(
            SimpleNamespace(root=out, spec=spec, guards=(guard,), repository=repository)
        )
        try:
            check_current(out)
            yield
        finally:
            _ENTRY.reset(token)


@contextmanager
def native_resource_session():
    active = check_current()
    if active is None:
        yield None
        return
    with active.repository.guarded_session(
        "RESOURCE:AUTOMORPH_FUNDUS", "native-inference"
    ) as guard:
        token = _ENTRY.set(SimpleNamespace(**dict(vars(active), guards=active.guards + (guard,))))
        try:
            check_current()
            yield check_current
        finally:
            _ENTRY.reset(token)


def entry_capability():
    repository = CasePipelineRepository(entry_target())
    with repository.guarded_session("RESOURCE:AUTOMORPH_FUNDUS", "native-inference") as guard:
        guard()
        return dict(available=True, profile=ENTRY_PROFILE, runtime_resource_lease=True)


def declare_selected(plan):
    """Bind newly prepared selected acquisitions to the versioned Runtime lease.

    Old plans are never upgraded here; they require their original immutable
    amendment. No exception-driven fallback from a filesystem lock is allowed.
    """
    from .campaign_renewal import RETAINED_ACQUISITION_VERSIONS

    if plan.get("schema_version") not in RETAINED_ACQUISITION_VERSIONS:
        raise ValueError("execution_lease_selected_version")
    if plan.get("mode") != "acquisition" or "execution_lease_profile" in plan:
        raise ValueError("execution_lease_selected_scope")
    plan["execution_lease_profile"] = PROFILE


def aggregate_binding(plan, parent, initial_targets):
    """Exact local binding to an aggregation restored by the original owners.

    This predicate cannot replace assembly, input/source or dispatch guards.
    It only selects the exclusion mechanism for that already verified scope.
    """
    baseline = getattr(parent, "aggregation_baseline", None)
    targets = plan.get("acquisition_targets")
    return (
        plan.get("schema_version")
        in {
            AGGREGATE_VERSION,
            "multimodal-campaign-renewal/1.68.0",
            "multimodal-campaign-renewal/1.73.0",
        }
        and (
            bool(plan.get("alternative_view"))
            == (
                plan.get("schema_version")
                in {"multimodal-campaign-renewal/1.68.0", "multimodal-campaign-renewal/1.73.0"}
            )
        )
        and plan.get("mode") == "acquisition"
        and plan.get("acquisition_increment_only") is True
        and baseline is not None
        and digest(baseline.state) == plan.get("accepted_state_checksum")
        and digest(baseline.manifest) == plan.get("accepted_origin_checksum")
        and isinstance(targets, list)
        and 1 <= len(targets) <= 3
        and all(isinstance(t, str) and t for t in targets)
        and len(set(targets)) == len(targets)
        and isinstance(initial_targets, (tuple, list))
        and tuple(targets) == tuple(initial_targets)
        and targets == plan.get("review_target_ids")
    )


def declare_aggregate(plan, parent):
    """Opt in only while preparing a fresh plan, never mutate an old protocol."""
    if "execution_lease_profile" in plan or not aggregate_binding(
        plan, parent, plan.get("acquisition_targets", ())
    ):
        raise ValueError("execution_lease_aggregate_declaration_binding")
    plan["execution_lease_profile"] = PROFILE


def validate_aggregate(campaign):
    if campaign.protocol.get("execution_lease_profile") != PROFILE or not aggregate_binding(
        campaign.protocol, campaign.parent, campaign.initial_targets
    ):
        raise ValueError("execution_lease_aggregate_binding")


def session(campaign):
    key = "multimodal-execution:" + digest(str(campaign.output.absolute()))[7:]
    return CasePipelineRepository(campaign.target).session(campaign.base.binding["case_code"], key)


def receipt_key(campaign):
    return "renewal:execution-lease:" + digest(str(campaign.output.absolute()))[7:]


def parent_documents(campaign):
    docs = campaign.documents(campaign.protocol["tip"]["run_id"])
    claim = dict(output=str(campaign.output.absolute()), plan_checksum=digest(campaign.protocol))
    if docs.read("authorized-campaign-renewal") != (claim,):
        raise ValueError("execution_lease_original_claim_binding")
    return docs


def read(campaign):
    value = json.loads((campaign.output / PATH).read_bytes())
    docs = parent_documents(campaign)
    if (
        value.get("schema_version") != VERSION
        or value.get("profile") != PROFILE
        or value.get("output") != str(campaign.output.absolute())
        or value.get("protocol") != digest(campaign.protocol)
        or value.get("implementation_before") != campaign.protocol["implementation"]
        or value.get("deadline") != campaign.protocol["deadline"]
        or docs.read(receipt_key(campaign)) != (value,)
        or digest(json.loads(Path(value["proof_path"]).read_bytes())) != value["proof"]
    ):
        raise ValueError("execution_lease_receipt_binding")
    return value


def unstarted(campaign, *, allowed_versions=None):
    from .campaign_renewal import RETAINED_ACQUISITION_VERSION

    if campaign.protocol.get("schema_version") == "multimodal-campaign-renewal/1.79.0":
        if (
            campaign.protocol.get("mode") != "definitions"
            or campaign.protocol.get("execution_lease_profile") != PROFILE
            or not campaign.protocol.get("definition_increment_only")
            or any(
                (campaign.output / name).exists()
                for name in (
                    "result.json",
                    "stage-failure.json",
                    "stage-closure-error.json",
                    "cancel",
                )
            )
            or list(campaign.output.glob("**/request.json"))
        ):
            raise ValueError("execution_lease_unstarted_definition_scope")
        for target in (None, "definition-expansion", *campaign.initial_targets):
            rid = identity("RUN", "task-research:" + campaign.request_key(target))
            if (
                campaign.contexts.read_research_run(rid, missing_ok=True) is not None
                or campaign.sessions.read(rid) is not None
            ):
                raise ValueError("execution_lease_phase_already_registered")
        return
    versions = {RETAINED_ACQUISITION_VERSION} if allowed_versions is None else allowed_versions
    if allowed_versions is None and aggregate_binding(
        campaign.protocol, getattr(campaign, "parent", None), campaign.initial_targets
    ):
        # assemble() has restored the original accepted aggregation and all its
        # origins. Bind that exact baseline again; a plain 1.26 acquisition is
        # not granted these repair semantics. The old plan remains immutable.
        versions = versions | {AGGREGATE_VERSION}
    if (
        campaign.protocol["schema_version"] not in versions
        or campaign.protocol["mode"] != "acquisition"
    ):
        raise ValueError("execution_lease_unstarted_acquisition_scope")
    for name in (
        "result.json",
        "stage-failure.json",
        "stage-closure-error.json",
        "prewire-repair.json",
        "archive-repair.json",
        "definition-review-repair.json",
    ):
        if (campaign.output / name).exists():
            raise ValueError("execution_lease_started_or_other_repair")
    for target in (None, *campaign.initial_targets):
        run_id = identity("RUN", "task-research:" + campaign.request_key(target))
        # A context is created before any phase request, Control or reservation.
        # Check the original repository, not just absence of filesystem mirrors.
        if campaign.contexts.read_research_run(run_id, missing_ok=True) is not None:
            raise ValueError("execution_lease_phase_already_registered")
        if (campaign.phase_path(target) / "request.json").exists():
            raise ValueError("execution_lease_phase_mirror_exists")


def authorize(output, proof, authorization):
    from .campaign_renewal import assemble

    if not authorization.strip():
        raise ValueError("execution_lease_explicit_authorization")
    campaign = assemble(output)
    with session(campaign):
        docs = parent_documents(campaign)
        saved = docs.read(receipt_key(campaign))
        if saved:
            # Recover only the immutable local copy after a lost acknowledgement.
            write_once(campaign.output / PATH, saved[0])
            value = read(campaign)
            if value["implementation_after"] != freeze():
                raise ValueError("execution_lease_implementation_changed")
            return dict(status="already_recorded", requests=0, deadline=value["deadline"])
        unstarted(campaign)
        campaign.guard()
        before, after = campaign.protocol["implementation"], freeze()
        changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
        allowed = (
            {
                "packages/multimodal_research/stages.py",
                "packages/multimodal_research/execution_lease.py",
            }
            if campaign.protocol.get("schema_version") == "multimodal-campaign-renewal/1.79.0"
            else ALLOWED
        )
        if not changed or not changed <= allowed:
            raise ValueError("execution_lease_change_scope")
        evidence_path = Path(proof).absolute()
        evidence = json.loads(evidence_path.read_bytes())
        if evidence.get("direct_exit") != 0 or evidence.get("implementation") != after:
            raise ValueError("execution_lease_test_proof")
        try:
            with (campaign.output / "lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno != errno.ENOLCK:
                raise
        else:
            raise ValueError("execution_lease_filesystem_available")
        value = dict(
            schema_version=VERSION,
            profile=PROFILE,
            output=str(campaign.output.absolute()),
            protocol=digest(campaign.protocol),
            implementation_before=before,
            implementation_after=after,
            deadline=campaign.protocol["deadline"],
            proof_path=str(evidence_path),
            proof=digest(evidence),
            authorization=authorization,
            original_lock_errno=errno.ENOLCK,
            requests=0,
            no_phase_registered=True,
        )
        with docs.session():
            # Other command instances share the same Runtime execution lease.
            if docs.read(receipt_key(campaign)) is not None:
                raise ValueError("execution_lease_receipt_conflict")
            docs.write(receipt_key(campaign), (value,))
        write_once(campaign.output / PATH, value)
        return dict(status="authorized_execution_lease", requests=0, deadline=value["deadline"])
