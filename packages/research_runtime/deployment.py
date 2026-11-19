"""Frozen owner installation to original 07A/06/Runtime/H ports; no fixture factories."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from db.runtime.repository.dataset_input import DatasetInputRepository
from db.runtime.repository.research import ResearchRuntimeExecutions
from db.runtime.repository.research_lifecycle import ResearchLifecycleRepository
from packages.case_score_ledger.models import ScoreDefinition
from packages.case_score_ledger.service import CaseScoreService, ToolObservation
from packages.contracts import VersionedReference
from packages.dataset_adapters.adapters import FairVLMedAdapter
from packages.dataset_adapters.service import DatasetPreparationService, VisibleInputReader
from packages.disease_packs import DiseasePackCatalog
from packages.fairclip_research.core import CONFIG_CHECKSUM, WEIGHT_BYTES, WEIGHT_SHA256
from packages.fairclip_research.qc import FairCLIPQCPolicy, FairCLIPQCRegistry
from packages.grounding.models import Candidate, Policy, Ref, Term, digest, validate
from packages.integrations.medgraph import (
    AdapterConfig,
    MedgraphCatalogReader,
    PostgresSource,
    SnapshotStore,
)
from packages.integrations.medgraph.configuration import load_projection_policy
from packages.integrations.medgraph.research import ReviewedCatalog
from packages.integrations.medgraph.resources import LocalResourcePolicy
from packages.object_store import LocalObjectStore, checksum
from packages.tool_admission import ToolAdmissionService
from packages.tool_admission.models import (
    AdmissionPolicy,
    ArtifactTypeBinding,
    DeploymentProfile,
    EnvironmentProbe,
    InvocationInput,
)
from packages.tool_execution.trusted_models import ReviewedFile, TrustedLocalPolicy
from packages.tool_execution.trusted_service import TrustedLocalGPUResearchService
from packages.tool_workflow import CommandAllowlist, ToolWorkflowCompiler
from packages.tool_workflow.models import TrustedResearchCompilerConfig

from .readiness import check_installation
from .registration import VisibleResearchRequests, single_choice
from .service import ResearchRuntimeService
from .single_case import ResearchSingleCaseService

ROOT = Path("/opt/data/test/MMAgent")


def medgraph_readonly_dsn(boundary):
    """Reuse original credential files, without process-global environment mutation."""
    from psycopg.conninfo import make_conninfo

    state, native = boundary.inputs()
    if (state["host"], state["port"], native["POSTGRES_DB"]) != ("127.0.0.1", 5432, "medgraph"):
        raise ValueError("research_original_medgraph_target_required")
    return make_conninfo(
        host="127.0.0.1",
        port=5432,
        dbname="medgraph",
        user="medgraph_llm",
        password=native["POSTGRES_LLM_PASSWORD"],
    )


def installed_training_adapter(selection, catalog):
    return FairVLMedAdapter(
        selection["root"],
        catalog,
        validate(VersionedReference, selection["dataset_ref"]),
        validate(VersionedReference, selection["pack_ref"]),
    )


def read_record(record, limit=1048576):
    record = validate(ReviewedFile, record)
    path = Path(record.path)
    if (
        not path.is_relative_to(ROOT)
        or any(p.is_symlink() for p in (path, *path.parents))
        or record.byte_size > limit
        or path.stat().st_size != record.byte_size
    ):
        raise ValueError("research_installation_file_bounds")
    data = path.read_bytes()
    if checksum(data) != record.checksum:
        raise ValueError("research_installation_file_drift")
    return data


def read_installed_bundle(record):
    # The resolved GPU inventory is larger than ordinary 1 MiB reviewed documents.
    # Keep a separate bounded owner-bundle read; review/path/digest limits still apply.
    return json.loads(read_record(record, limit=8388608))


def load_configuration(path):
    if path.stat().st_size > 1048576:
        raise ValueError("research_configuration_size")
    doc = json.loads(path.read_bytes())
    body = {k: v for k, v in doc.items() if k != "checksum"}
    if (
        doc.get("checksum") != digest(body)
        or doc.get("version") != "fairclip-single-deployment/1.0.0"
    ):
        raise ValueError("research_configuration_drift")
    if (doc["classification_checksum"], doc["weight_bytes"], doc["weight_sha256"]) != (
        CONFIG_CHECKSUM,
        WEIGHT_BYTES,
        WEIGHT_SHA256,
    ):
        raise ValueError("research_fixed_model_config_mismatch")
    if digest(doc["classification"]) != CONFIG_CHECKSUM:
        raise ValueError("research_preprocessing_or_classification_drift")
    if doc["budget_seconds"] > 3600 or not 1 <= doc["budget_seconds"]:
        raise ValueError("research_single_case_budget")
    return doc


FAIRCLIP_MIGRATION_PREFIXES = {
    "core": 2,
    "runtime": 13,
    "audit": 3,
    "knowledge": 18,
    "grounding": 2,
}

# This port consumes original input/06/Planner and TaskRun2, not extension queue
# workers. The existing optional 0016--0022 extension SQL is not a prerequisite.
COMPONENT_MIGRATION_PREFIXES = {
    "core": 2,
    "runtime": 15,
    "audit": 4,
    "knowledge": 19,
    "grounding": 2,
}


def required_migrations(domain, *, profile="all_installed_services"):
    if profile not in {
        "all_installed_services",
        "installed_fairclip/1.0.0",
        "installed_component/1.0.0",
    }:
        raise ValueError("research_deployment_profile")
    files = {p.name for p in (ROOT / "db" / domain / "migrations").glob("*.sql")}
    if profile == "installed_fairclip/1.0.0":
        if domain not in FAIRCLIP_MIGRATION_PREFIXES:
            raise ValueError("research_profile_domain")
        files = {n for n in files if int(n.split("_", 1)[0]) <= FAIRCLIP_MIGRATION_PREFIXES[domain]}
    elif profile == "installed_component/1.0.0":
        if domain not in COMPONENT_MIGRATION_PREFIXES:
            raise ValueError("research_profile_domain")
        files = {
            n
            for n in files
            if int(n.split("_", 1)[0]) <= COMPONENT_MIGRATION_PREFIXES[domain]
            or (domain == "runtime" and int(n.split("_", 1)[0]) == 23)
        }
    return files


def verify_actual_target(boundary, *, profile="all_installed_services"):
    """Before any upgraded Repository; old boundary alone resolves all secrets."""
    target = boundary.resolve_target("runtime")
    expected = boundary.private_json(boundary.STATE)
    admin = boundary.admin_target()
    with boundary.connection(admin, readonly=True) as c:
        row = c.execute(
            "SELECT current_database(),system_identifier::text FROM pg_control_system()"
        ).fetchone()
        if row != ("mmagent_06e_benchmark_v1", expected["cluster_identifier"]):
            raise ValueError("research_actual_instance_mismatch")
        domains = (
            tuple(FAIRCLIP_MIGRATION_PREFIXES)
            if profile == "installed_fairclip/1.0.0"
            else tuple(COMPONENT_MIGRATION_PREFIXES)
            if profile == "installed_component/1.0.0"
            else boundary.manage.DOMAINS
        )
        for domain in domains:
            from psycopg import sql

            rows = c.execute(
                sql.SQL("SELECT filename,checksum FROM {}.schema_migrations").format(
                    sql.Identifier(domain)
                )
            ).fetchall()
            actual = dict(rows)
            for name, value in actual.items():
                path = ROOT / "db" / domain / "migrations" / name
                if not path.is_file() or checksum(path.read_bytes()) != value:
                    raise ValueError("research_actual_migration_drift")
            if domain == "runtime" and "0012_research_runtime_context.sql" not in actual:
                raise ValueError("research_runtime_0012_required")
            expected_files = required_migrations(domain, profile=profile)
            if not expected_files <= set(actual) or (
                profile == "all_installed_services" and set(actual) != expected_files
            ):
                raise ValueError("research_actual_migrations_missing_review_required")
        role = c.execute(
            "SELECT pg_has_role(%s,'mmagent_runtime_app','MEMBER'),"
            "rolsuper,rolcreatedb,rolcreaterole "
            "FROM pg_roles WHERE rolname=%s",
            (target.user, target.user),
        ).fetchone()
        if role != (True, False, False, False):
            raise ValueError("research_runtime_role_mismatch")
    return target


def existing_store(path, purpose):
    path = Path(path)
    if not path.is_relative_to(ROOT / ".local") or any(
        p.is_symlink() for p in (path, *path.parents)
    ):
        raise ValueError("research_object_root")
    marker = json.loads((path / "owner.json").read_text())
    owner = "07-p2-services-v1" if purpose in {"visible", "evaluation"} else "07-j03-r01"
    if marker != {"owner": owner, "purpose": purpose}:
        raise ValueError("research_object_owner")
    for p in (path, path / "objects", path / "staging"):
        if not p.is_dir() or p.stat().st_uid != os.geteuid() or p.stat().st_mode & 0o077:
            raise ValueError("research_object_permissions")
    return LocalObjectStore(path)


class InstalledProfiles:
    def __init__(self, record, catalogs):
        self.record, self.catalogs = record, catalogs
        self.document = read_installed_bundle(record)

    def current(self):
        if read_installed_bundle(self.record) != self.document:
            raise ValueError("research_installation_changed")
        return self.document

    def read_manifest(self, *args):
        self.current()
        return self.catalogs.read_manifest(*args)

    def read_profile(self, manifest):
        value = validate(DeploymentProfile, self.current()["profile"])
        if value.manifest_checksum != manifest.checksum:
            raise ValueError("research_profile_binding")
        return value

    def read_probe(self, manifest, profile):
        value = validate(EnvironmentProbe, self.current()["probe"])
        if (value.manifest_checksum, value.profile_checksum) != (
            manifest.checksum,
            profile.checksum,
        ):
            raise ValueError("research_probe_binding")
        return value

    def read_allowlist(self, manifest, profile, compiler):
        return validate(CommandAllowlist, self.current()["allowlist"])


class CombinedResearchObjects:
    """Original visible/output routing; evaluation objects are never installed."""

    def __init__(self, visible, output):
        if visible.root == output.root:
            raise ValueError("research_object_separation")
        self.visible, self.output = visible, output

    def read_bounded(self, uri, limit):
        if self.visible._path(uri).exists() and self.output._path(uri).exists():
            raise ValueError("research_object_uri_collision")
        try:
            return self.output.read_bounded(uri, limit)
        except FileNotFoundError:
            return self.visible.read_bounded(uri, limit)

    def __getattr__(self, name):
        return getattr(self.output, name)


def assemble(
    config,
    boundary,
    *,
    source_factory=None,
    profile_factory=None,
    deployment_profile="all_installed_services",
):
    """Requires an owner-installed real 06 bundle; never creates/reviews catalog content."""
    if config["installed_bundle"] is None:
        raise ValueError("research_installation_pending")
    installed = read_installed_bundle(config["installed_bundle"])
    if installed["classification_checksum"] != CONFIG_CHECKSUM:
        raise ValueError("research_classification_binding")
    target = (
        verify_actual_target(boundary)
        if deployment_profile == "all_installed_services"
        else verify_actual_target(boundary, profile=deployment_profile)
    )
    adapter_config = validate(AdapterConfig, installed["adapter_config"])
    if adapter_config.dsn_env != "MEDGRAPH_READONLY_DSN":
        raise ValueError("research_original_medgraph_boundary_required")
    resources = LocalResourcePolicy(
        allowed_roots=tuple(installed["resource_roots"]), max_bytes=2147483648, timeout_seconds=60
    )
    policy_record = installed["projection_policy"]
    read_record(policy_record)
    projection = load_projection_policy(Path(policy_record["path"]), ROOT)
    source = (source_factory or PostgresSource)(
        adapter_config, dsn_resolver=lambda: medgraph_readonly_dsn(boundary)
    )
    with source.transaction() as connection:
        if (connection.info.host, connection.info.port, connection.info.dbname) != (
            "127.0.0.1",
            5432,
            "medgraph",
        ):
            raise ValueError("research_actual_medgraph_target_mismatch")
        role = connection.execute(
            "SELECT current_user AS role, current_setting('transaction_read_only') AS readonly"
        ).fetchone()
        if role != {"role": "medgraph_reader", "readonly": "on"}:
            raise ValueError("research_actual_medgraph_role")
    legacy = MedgraphCatalogReader(
        source,
        SnapshotStore(Path(installed["snapshot_root"])),
        adapter_config,
        projection,
    )
    catalogs = ReviewedCatalog(
        legacy, validate(ReviewedFile, installed["research_review"]), resources
    )
    if catalogs.catalog.synthetic:
        raise ValueError("research_real_catalog_required")
    catalogs.current_safety(catalogs.catalog.reference())
    owners = (profile_factory or InstalledProfiles)(config["installed_bundle"], catalogs)
    runtime = ResearchLifecycleRepository(target)
    visible = existing_store(installed["visible_root"], "visible")
    evaluator = existing_store(installed["evaluator_root"], "evaluation")
    objects = existing_store(installed["output_root"], "research_output")
    if len({visible.root, evaluator.root, objects.root}) != 3:
        raise ValueError("research_object_separation")

    combined = CombinedResearchObjects(visible, objects)
    dataset_repo = DatasetInputRepository(runtime)
    visible_reader = VisibleInputReader(dataset_repo, visible)
    requests = VisibleResearchRequests(
        runtime=runtime,
        visible_reader=visible_reader,
        objects=visible,
        candidate=validate(Candidate, installed["candidate"]),
        invocation=validate(InvocationInput, installed["invocation"]),
        catalog_ref=catalogs.catalog.reference(),
        policy=validate(Policy, installed["compatibility_policy"]),
        input_binding=validate(ArtifactTypeBinding, installed["input_type_binding"]),
        modality=validate(Term, installed["input_modality"]),
        scope_terms=installed["input_scope_terms"],
        config_checksum=config["checksum"],
    )
    admission = ToolAdmissionService(
        policy=AdmissionPolicy(max_total_bytes=2147483648),
        grounding=None,
        catalogs=catalogs,
        manifests=owners,
        profiles=owners,
        probes=owners,
        inputs=requests,
        research=requests,
        objects=combined,
        resources=resources,
        clock=lambda: datetime.now(UTC),
    )
    compiler = ToolWorkflowCompiler(
        admission, owners, validate(TrustedResearchCompilerConfig, installed["compiler"])
    )
    check_installation(installed, catalogs, owners, admission, compiler, config)
    research = ResearchRuntimeService(
        runtime=runtime, research=requests, inputs=requests, compiler=compiler, synthetic=False
    )
    qc_policy = validate(FairCLIPQCPolicy, installed["qc_policy"])
    qc = FairCLIPQCRegistry({validate(Ref, installed["qc_ref"]): qc_policy})

    def bind(plan, key):
        policy = validate(
            TrustedLocalPolicy,
            {**installed["execution_policy"], "allowed_plan_checksums": [plan.checksum]},
        )
        if len(policy.gpu_pool) != 1:
            raise ValueError("research_single_GPU_required")
        executions = ResearchRuntimeExecutions(
            target, compiler=compiler, plans=(plan.to_bundle_json(),)
        )
        executor = TrustedLocalGPUResearchService(
            compiler=compiler,
            catalogs=catalogs,
            objects=combined,
            qc=qc,
            repository=executions,
            policy=policy,
            workspace=Path(installed["execution_workspace"]),
        )
        outputs = tuple(
            (
                qc_policy.artifact_type_code,
                ScoreDefinition(
                    code="fairclip.cosine." + str(i),
                    layer="model",
                    metric="logit",
                    unit="logit",
                    definition=(
                        "Fixed text/image cosine before unit-temperature softmax; "
                        "uncalibrated research model score"
                    ),
                    direction="higher",
                ),
                "fairclip-output/1.0.0",
                "raw_cosine_" + str(i),
            )
            for i in range(2)
        )
        scores = CaseScoreService(
            runtime=runtime,
            executions=executions,
            objects=combined,
            synthetic=False,
            context_version="research-run/1.0.0",
            observations=(
                ToolObservation(
                    key=key,
                    plan_json=plan.to_bundle_json(),
                    compiler=compiler,
                    outputs=outputs,
                    model_versions=tuple(
                        validate(VersionedReference, v) for v in installed["model_versions"]
                    ),
                ),
            ),
        )
        return executor, scores

    disease_catalog = DiseasePackCatalog.from_directory(ROOT / "configs/disease_packs")
    preparation = DatasetPreparationService(
        dataset_repo, visible, evaluator, catalog=disease_catalog
    )
    selection = installed["training_selection"]

    def prepare_sample():
        # Owner-prepared selection; no search, no label-conditioned selection, Training only.
        adapter = installed_training_adapter(selection, disease_catalog)
        sample = adapter.prepare(selection["filename"], split="training")
        if (
            len(sample.images) != 1
            or checksum(sample.images[0][0]) != config["historical_visible_image_checksum"]
        ):
            raise ValueError("research_same_Training_input_required")
        return sample

    service = ResearchSingleCaseService(
        runtime=runtime,
        research=research,
        choice=lambda req: single_choice(admission.admit_research(req)),
        bind_plan=bind,
        config_checksum=config["checksum"],
        budget_seconds=config["budget_seconds"],
        preparation=preparation,
        register_request=requests.register,
    )
    return SimpleNamespace(service=service, requests=requests, prepare_sample=prepare_sample)
