"""D01 non-null installed-bundle composition, exclusively synthetic owned files/PG.

Only external target/catalog/registered input observations and GPU visibility are
substituted. The factory, installed records, readiness, compiler, trusted harmless
CPU child, journal, Runtime and H are real. Nothing here is installable production
configuration. Original 07A Training metadata is tested separately as fail-closed.
"""

import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from db import manage
from packages.contracts import CaseBundle, Laterality
from packages.fairclip_research.core import CONFIG, CONFIG_CHECKSUM, WEIGHT_SHA256
from packages.fairclip_research.models import FairCLIPOutput
from packages.fairclip_research.qc import FairCLIPQCPolicy
from packages.grounding.models import digest, seal, validate
from packages.grounding.ports import ToolSafety
from packages.grounding.synthetic import ref
from packages.integrations.medgraph.models import ReviewedDeploymentManifest
from packages.object_store import checksum
from packages.research_runtime import deployment, readiness
from packages.tool_admission.models import InvocationInput, JSONOutputRule, ResearchRequest
from packages.tool_admission.research_metadata import FAIRCLIP_METADATA
from packages.tool_workflow import CommandAllowlist, ResearchContext
from packages.tool_workflow.models import OutputSpec, TrustedResearchCompilerConfig
from tests.disease_packs.support import anonymous_case, code
from tests.grounding.test_database import database as shared_database
from tests.tool_admission.support import fixture, reseal
from tests.tool_execution.test_trusted_local import FakeGPU, pinned
from tests.tool_execution.test_trusted_local import setup as setup


@pytest.fixture
def database(database_lease):
    # Each fixed-identity factory scenario owns a distinct cluster; never delete
    # another scenario's rows or relax the original Runtime identity constraint.
    yield from shared_database.__wrapped__(database_lease)


def installed_shape(setup, monkeypatch, *, target=None, training_metadata=False):
    root = setup.root
    f = fixture(root)
    # A single synthetic input with known metadata; no patient or actual model bytes.
    old_candidate = f.mapping.candidates[0]
    candidate = validate(
        type(old_candidate),
        {**old_candidate.model_dump(mode="json"), "inputs": [old_candidate.inputs[0]]},
    )
    if training_metadata:
        from packages.grounding.models import Term

        required = candidate.inputs[0]
        constraint = validate(
            type(required.constraint),
            {
                **required.constraint.model_dump(mode="json"),
                "laterality": "UNKNOWN",
                "coordinate_space": "PIXEL_2D",
                "unit": Term(state="known", code="CONCEPT:UNIT:PIXEL"),
                "qc_required": ["UNASSESSED"],
            },
        )
        candidate = candidate.model_copy(
            update={
                "inputs": (required.model_copy(update={"constraint": constraint}),),
                "output": candidate.output.model_copy(update={"laterality": Laterality.UNKNOWN}),
            }
        )
    tool = f.catalog.tools[0].model_copy(
        update={"inputs": candidate.inputs, "outputs": (candidate.output,)}
    )
    catalog = reseal(f.catalog, synthetic=False, tools=[tool])
    old_manifest = f.owners.read_manifest(
        f.catalog.reference(), candidate.tool_ref, candidate.implementation_ref
    )
    old_profile = f.owners.read_profile(old_manifest)
    constraint = candidate.output
    qc = FairCLIPQCPolicy(
        artifact_ref=constraint.artifact_ref,
        artifact_type_code="ARTIFACT:TEST:FAIRCLIP_JSON",
        unit_term=constraint.unit,
    )
    qc_ref = ref("policy", "POLICY:TEST:D01:JSON").model_copy(update={"checksum": digest(qc)})
    compiler = seal(
        TrustedResearchCompilerConfig,
        max_steps=1,
        max_plan_bytes=8388608,
        timeout_seconds=2,
        memory_bytes=134217728,
        cpu_count=1,
        max_output_bytes=65536,
    )
    result = FairCLIPOutput(
        input_checksum=digest("replaced with bound input below"),
        model_checksum="sha256:" + WEIGHT_SHA256,
        config_checksum=CONFIG_CHECKSUM,
        raw_cosines=(0.0, 0.0),
        normalized_model_scores=(0.5, 0.5),
        predicted_index=0,
        predicted_class="non-glaucoma",
    )
    # No torch, model import, CUDA allocation or inference. JSON body is test data.
    setup.script.write_text(
        "import sys\nfrom _sha2 import sha256\n"
        "raw=open(sys.argv[1], 'rb').read()\n"
        f"body={result.model_dump_json()!r}\n"
        f"body=body.replace({result.input_checksum!r}, 'sha256:'+sha256(raw).hexdigest())\n"
        "open(sys.argv[2], 'xb').write(body.encode())\n"
    )
    # The copied minimal child requires only this standard library extension.
    import _sha2
    import shutil

    ext = Path(setup.review.environment_root) / "lib/python3.12/lib-dynload"
    ext.mkdir()
    shutil.copyfile(_sha2.__file__, ext / Path(_sha2.__file__).name)
    fake_weight = root / "code" / "test-weight.txt"
    fake_weight.write_text("SYNTHETIC NON-MODEL RESOURCE; never production.\n")
    monkeypatch.setattr(readiness, "WEIGHT_SHA256", pinned(fake_weight).checksum[7:])
    monkeypatch.setattr(readiness, "WEIGHT_BYTES", fake_weight.stat().st_size)
    runtime = dict(
        interpreter_uri=Path(setup.review.executable.path).as_uri(),
        interpreter_checksum=setup.review.executable.checksum,
        dependencies={},
        gpu="required",
    )
    resource = dict(
        resource_key="entrypoint",
        resource_role="source_code",
        uri=setup.script.as_uri(),
        sha256=pinned(setup.script).checksum[7:],
        byte_size=setup.script.stat().st_size,
        required=True,
        metadata={"synthetic_D01_only": True},
    )
    argv = [
        setup.review.executable.path,
        str(setup.script),
        "{input:" + candidate.inputs[0].port.port_name + "}",
        "{output:" + candidate.output_port.port_name + "}",
    ]
    invocation = dict(shell=False, argv=argv, timeout_seconds=2)
    pre = {
        "version": "1.0.0",
        "policy_code": "POLICY:TEST:D01:PRE",
        "fixed_preprocess": CONFIG["preprocess"],
        "metadata_requirements": FAIRCLIP_METADATA,
    }
    post = {
        "version": "1.0.0",
        "policy_code": "POLICY:TEST:D01:POST",
        "classification_checksum": CONFIG_CHECKSUM,
    }
    manifest = reseal(
        old_manifest,
        catalog_ref=catalog.reference(),
        invocation=invocation,
        ports=[
            p
            for p in old_manifest.ports
            if p.direction == "output" or p.port_name == candidate.inputs[0].port.port_name
        ],
        preprocessing_policy=pre,
        postprocessing_policy=post,
        native_manifest={
            **old_manifest.native_manifest.model_dump(mode="json"),
            "entrypoint_uri": setup.script.as_uri(),
            "resources": [resource],
            "invocation": invocation,
            "runtime_environment": {"admission_runtime": runtime},
        },
    )
    manifest = seal(
        ReviewedDeploymentManifest,
        **{
            k: v
            for k, v in manifest.model_dump(mode="json").items()
            if k not in {"schema_version", "checksum"}
        },
        review_ref=ref("source_version", "SOURCE:TEST:D01:METADATA_REVIEW").model_copy(
            update={"checksum": manifest.policy_checksum}
        ),
        original_manifest_checksum=old_manifest.checksum,
        deployment_declaration=manifest.native_manifest,
    )
    profile = reseal(
        old_profile,
        manifest_checksum=manifest.checksum,
        runtime=runtime,
        input_rules=[old_profile.input_rules[0]],
        output_rules=[
            validate(
                JSONOutputRule,
                {
                    **old_profile.output_rules[0].model_dump(mode="json"),
                    "media_types": ("application/json",),
                    "dimensions_min": (2,),
                    "dimensions_max": (2,),
                },
            )
        ],
        preprocessing_checksum=digest(pre),
        postprocessing_checksum=digest(post),
        output_target_bindings=[
            reseal(b, constraint=candidate.output) if b.port == candidate.output_port else b
            for b in old_profile.output_target_bindings
        ],
    )
    now = datetime.now(UTC)
    probe = reseal(
        f.owners.read_probe(old_manifest, old_profile),
        manifest_checksum=manifest.checksum,
        profile_checksum=profile.checksum,
        runtime_checksum=digest(profile.runtime),
        gpu="available",
        observed_at=now - timedelta(seconds=1),
        expires_at=now + timedelta(hours=1),
    )
    inv = InvocationInput(
        implementation_ref=candidate.implementation_ref,
        manifest_checksum=manifest.checksum,
        profile_checksum=profile.checksum,
        parameters={},
    )
    allowlist = seal(
        CommandAllowlist,
        version="1.0.0",
        source_ref=ref("source_version", "SOURCE:TEST:D01:ALLOWLIST"),
        manifest_checksum=manifest.checksum,
        profile_checksum=profile.checksum,
        compiler_config_checksum=compiler.checksum,
        synthetic_only=False,
        executable_uri=runtime["interpreter_uri"],
        executable_checksum=runtime["interpreter_checksum"],
        entrypoint_uri=setup.script.as_uri(),
        entrypoint_checksum=pinned(setup.script).checksum,
        resources=[resource],
        argv=argv,
        parameter_schema_version="1.0.0",
        parameter_schema_checksum=digest(manifest.parameter_schema),
        outputs=[
            OutputSpec(port=candidate.output_port, filename="output.json", qc_policy_ref=qc_ref)
        ],
    )
    files = [
        pinned(p)
        for base in (Path(setup.review.environment_root), root / "code")
        for p in sorted(base.rglob("*"))
        if p.is_file()
    ]
    review = setup.review.model_copy(
        update=dict(
            implementation_ref=candidate.implementation_ref,
            manifest_checksum=manifest.checksum,
            allowlist_checksum=allowlist.checksum,
            entrypoint=pinned(setup.script),
            files=tuple(files),
        )
    )
    execution = setup.policy.model_dump(mode="json")
    execution.update(reviews=[review.model_dump(mode="json")], allowed_plan_checksums=[])

    class Catalogs:
        def __init__(self):
            self.catalog = catalog

        def read_snapshot(self, reference):
            assert reference == catalog.reference()
            return catalog

        def current_safety(self, reference):
            self.read_snapshot(reference)
            return ToolSafety(catalog_ref=reference, disabled_implementations=(), revision=0)

        def read_manifest(self, reference, tool, implementation):
            assert (reference, tool, implementation) == (
                catalog.reference(),
                candidate.tool_ref,
                candidate.implementation_ref,
            )
            return manifest

    catalogs = Catalogs()
    monkeypatch.setattr(deployment, "verify_actual_target", lambda boundary: target)
    monkeypatch.setattr(deployment, "MedgraphCatalogReader", lambda *args: object())
    monkeypatch.setattr(deployment, "ReviewedCatalog", lambda *args: catalogs)

    class Source:
        def __init__(self, config, *, dsn_resolver=None):
            pass

        @contextmanager
        def transaction(self):
            yield SimpleNamespace(
                info=SimpleNamespace(host="127.0.0.1", port=5432, dbname="medgraph"),
                execute=lambda sql: SimpleNamespace(
                    fetchone=lambda: {"role": "medgraph_reader", "readonly": "on"}
                ),
            )

    monkeypatch.setattr(deployment, "PostgresSource", Source)
    monkeypatch.setattr(readiness, "NvidiaObservation", FakeGPU)
    from packages.tool_execution import trusted_process

    monkeypatch.setattr(trusted_process, "NvidiaObservation", FakeGPU)
    for name, purpose in (
        ("visible", "visible"),
        ("evaluation", "evaluation"),
        ("output", "research_output"),
    ):
        path = root / name
        path.mkdir(mode=0o700)
        for child in ("objects", "staging"):
            (path / child).mkdir(mode=0o700)
        (path / "owner.json").write_text(
            json.dumps(
                dict(
                    owner="07-p2-services-v1" if name != "output" else "07-j03-r01", purpose=purpose
                )
            )
        )
    (root / "snapshot").mkdir(mode=0o700)
    workspace = root / "execution"
    workspace.mkdir(mode=0o700)
    config = deployment.load_configuration(
        deployment.ROOT / "configs/research/fairclip_single_case.json"
    )
    config.update(
        adapter_entrypoint=str(setup.script),
        weight_path=str(fake_weight),
        adapter_files=[pinned(setup.script).model_dump(mode="json")],
    )
    input_record = next(
        r
        for r in f.owners.records.values()
        if r.metadata.artifact_ref == candidate.inputs[0].constraint.artifact_ref
    )
    bundle = dict(
        classification_checksum=CONFIG_CHECKSUM,
        adapter_config=json.loads((deployment.ROOT / "configs/medgraph/adapter.json").read_text()),
        projection_policy=pinned(deployment.ROOT / "configs/medgraph/projection-policy.json"),
        resource_roots=[str(root)],
        snapshot_root=str(root / "snapshot"),
        research_review=pinned(Path(setup.review.review_record.path)),
        candidate=candidate,
        invocation=inv,
        profile=profile,
        probe=probe,
        compatibility_policy=reseal(f.policy, synthetic_only=False),
        input_type_binding=input_record.type_binding,
        input_modality=input_record.metadata.modality,
        input_scope_terms=list(f.q.satisfied_terms),
        compiler=compiler,
        allowlist=allowlist,
        visible_root=str(root / "visible"),
        evaluator_root=str(root / "evaluation"),
        output_root=str(root / "output"),
        execution_workspace=str(workspace),
        execution_policy=execution,
        qc_policy=qc,
        qc_ref=qc_ref,
        model_versions=[
            dict(code="SOURCE:TEST:D01", version="1.0.0", checksum=pinned(fake_weight).checksum),
            dict(
                code="SOURCE:FAIRCLIP:TEST_DECLARATION",
                version="1.0.0",
                checksum="sha256:" + WEIGHT_SHA256,
            ),
        ],
        training_selection={"fixture": "not invoked in deployment readiness"},
    )
    if training_metadata:
        bundle["input_type_binding"] = reseal(
            input_record.type_binding,
            contract_type_code="ARTIFACT:SLO_IMAGE",
            unit_term=candidate.inputs[0].constraint.unit,
            contract_unit="px",
        )
        bundle["compatibility_policy"] = reseal(
            bundle["compatibility_policy"],
            allowed_terms=[*f.policy.allowed_terms, "CONCEPT:UNIT:PIXEL"],
        )

    def write_bundle():
        path = root / "installed-bundle.json"
        path.write_text(json.dumps(bundle, default=lambda x: x.model_dump(mode="json"), indent=2))
        config["installed_bundle"] = pinned(path).model_dump(mode="json")
        return path

    write_bundle()
    return SimpleNamespace(
        config=config,
        bundle=bundle,
        write_bundle=write_bundle,
        candidate=candidate,
        input_record=input_record,
        f=f,
        catalogs=catalogs,
        manifest=manifest,
    )


def test_non_null_factory_ports_and_frozen_bundle_drift(setup, monkeypatch):
    f = installed_shape(setup, monkeypatch)
    monkeypatch.setattr(deployment, "ResearchLifecycleRepository", lambda target: object())
    assembled = deployment.assemble(f.config, object())
    assert assembled.service.research.synthetic is False  # guard exercised; fixture only
    assert assembled.service.research.inputs is assembled.requests
    assert assembled.service.research.compiler.admission.inputs is assembled.requests
    assert assembled.service.preparation.visible_store.root == Path(f.bundle["visible_root"])
    assert assembled.service.budget_seconds == f.config["budget_seconds"]
    # All original 06 installation checks have passed without any Runtime write.
    path = f.write_bundle()
    path.write_text(path.read_text() + " ")
    with pytest.raises(
        ValueError, match="research_installation_file_bounds|research_installation_file_drift"
    ):
        deployment.assemble(f.config, object())


def test_non_null_factory_compile_execute_h_and_recover_pg(database, setup, monkeypatch):
    """New D01 port composition; not a rerun of the R01 lifecycle scenario."""
    f = installed_shape(setup, monkeypatch, target=manage.test_target("runtime"))
    assembled = deployment.assemble(f.config, object())
    runtime = assembled.service.runtime
    source = f.input_record
    visible = assembled.service.preparation.visible_store
    raw = f.f.objects.read_bounded(source.artifact.uri, 4194304)
    uri = visible.promote(visible.stage(raw), checksum(raw))
    source = validate(
        type(source),
        {
            **source.model_dump(mode="json"),
            "artifact": source.artifact.model_copy(update={"uri": uri}),
        },
    )
    f.f.owners.records = {source.artifact.asset_code: source}
    scope = f.f.q.scope
    case = anonymous_case("GOAL:SYNTHETIC:D01")
    case = CaseBundle.model_validate_json(
        case.model_copy(
            update=dict(
                case_code=scope.case_code,
                encounter_code=scope.encounter_code,
                patient_hash=scope.patient_hash,
                laterality=scope.laterality,
                scope=case.scope.model_copy(update={"case_code": scope.case_code}),
                artifacts=(source.artifact,),
            )
        ).model_dump_json()
    )
    runtime.create_case_bundle(case, code("TASK"))
    request = seal(
        ResearchRequest,
        code="RESEARCH:TEST:D01:FACTORY",
        version="1.0.0",
        purpose="shadow",
        compatibility_policy=f.bundle["compatibility_policy"],
        lock=dict(run_id=code("RUN"), catalog_ref=f.catalogs.catalog.reference()),
        scope=scope,
        window_start=f.f.q.window_start,
        window_end=f.f.q.window_end,
        snapshot_at=f.f.q.snapshot_at,
        satisfied_terms=f.f.q.satisfied_terms,
        artifacts=(source.metadata,),
        candidate=f.candidate,
        inputs=[dict(port=f.candidate.inputs[0].port, asset_codes=[source.artifact.asset_code])],
        invocation=f.bundle["invocation"],
    )
    f.f.owners.requests[(request.code, request.version, request.checksum)] = request
    # Registered-input observations are the external test port. Do not change 06
    # admission or claim this is 07A unknown-eye/time Training admission.
    monkeypatch.setattr(assembled.requests, "read_request", f.f.owners.read_request)
    monkeypatch.setattr(assembled.requests, "read_input", f.f.owners.read_input)
    compiler = assembled.service.research.compiler
    admission = compiler.admission.admit_research(request)
    assert admission.status == "preparation_only", admission.model_dump_json(indent=2)
    from packages.research_runtime.registration import single_choice

    plan = compiler.compile(
        ResearchContext(request=request, admission=admission), single_choice(admission)
    )
    executor, _ = assembled.service.bind_plan(plan, "d01-factory:tool")
    executor._validate(plan)
    report = assembled.service.run_registered(request, key="d01-factory")
    assert report["technical_status"] == "SUCCEEDED", (
        json.dumps(report, indent=2)
        + "\n"
        + "\n".join(
            p.read_text(errors="replace")
            for p in Path(f.bundle["execution_workspace"]).rglob("*.log")
        )
    )
    assert report["run_status"] == "COMPLETED_INSUFFICIENT_EVIDENCE"
    assert report["terminal"] and report["graph"] is None
    assert report["graph_scores"] == "unavailable" and not report["accepted_validations"]
    cosine_entries = [
        e for e in report["model_entries"] if e["definition"]["code"].startswith("fairclip.cosine.")
    ]
    assert len(cosine_entries) == 2
    assert [e["value"] for e in cosine_entries] == [0.0, 0.0]
    assert report["model_outputs"][0]["output"]["normalized_model_scores"] == [0.5, 0.5]
    assert report["model_outputs"][0]["output"]["predicted_class"] == "non-glaucoma"
    assert report["model_outputs"][0]["output"]["calibration"] == "unverified"
    assert runtime.lifecycle(request.lock.run_id)["phase"] == "FINISHED"
    from packages.tool_execution.trusted_service import TrustedLocalGPUResearchService

    monkeypatch.setattr(
        TrustedLocalGPUResearchService,
        "execute",
        lambda *a, **k: pytest.fail("finished recovery must not execute"),
    )
    recovered = assembled.service.run_registered(request, key="d01-factory", recover_only=True)
    assert recovered == report
    assert not (setup.lock_root / "active.json").exists()
    import os

    evidence = os.environ.get("MMAGENT_D01_EVIDENCE_DIR")
    if evidence:
        path = Path(evidence).resolve()
        assert path.is_relative_to(deployment.ROOT / ".local")
        with (path / "synthetic-factory-report.json").open("x") as output:
            json.dump(
                dict(
                    kind="SYNTHETIC_D01_FACTORY_TEST_ONLY",
                    model_forward=False,
                    actual_target_used=False,
                    actual_Training_admitted=False,
                    substituted_ports=[
                        "catalog observations",
                        "known registered inputs",
                        "FakeGPU",
                    ],
                    report=report,
                ),
                output,
                indent=2,
            )


def test_training_unknown_metadata_still_fails_original_admission():
    """Concrete current port conflict, not a fabricated historic red or clinical fix."""
    from packages.grounding.compatibility import artifact_compatibility
    from packages.grounding.query_models import ArtifactMetadata
    from tests.grounding.query_support import ready

    q, mapping, *_ = ready()
    constraint = mapping.candidates[0].inputs[0].constraint
    source = next(a for a in q.artifacts if a.artifact_ref == constraint.artifact_ref)
    # Current 07A contract fixes UNKNOWN laterality; VisibleResearchRequests sets
    # acquired_at=None. These are missing facts, not permission to invent timestamps.
    current = validate(
        ArtifactMetadata,
        {**source.model_dump(mode="json"), "laterality": "UNKNOWN", "acquired_at": None},
    )
    request = q.model_copy(
        update={"scope": q.scope.model_copy(update={"laterality": Laterality.UNKNOWN})}
    )
    errors = artifact_compatibility(current, constraint, request)
    assert "required_condition_unknown" in errors and "time_unknown" in errors


def test_large_real_file_inventory_pg_result_and_recovery(database, setup, monkeypatch):
    """Thousands of actual synthetic files reproduce GPU-env record size, not GPU work."""
    # Long valid paths cover the same >1 MiB record boundary with less filesystem IO.
    inventory = setup.root / "code"
    for part in ("a", "b", "c", "d"):
        inventory = inventory / (part * 160)
    inventory.mkdir(parents=True)
    for index in range(1600):
        (inventory / f"library-{index:05d}.txt").write_text("synthetic data; not executable\n")
    test_non_null_factory_compile_execute_h_and_recover_pg(database, setup, monkeypatch)
    records = list((setup.root / "execution").rglob("result.json"))
    assert len(records) == 1
    assert 1048576 < records[0].stat().st_size < 4194304


def test_step_journal_capacity_remains_bounded(tmp_path):
    from packages.tool_execution.models import STEP_RESULT_MAX_BYTES, ExecutionError
    from packages.tool_execution.process import read_file

    record = tmp_path / "result.json"
    record.write_bytes(b"x" * (STEP_RESULT_MAX_BYTES + 1))
    with pytest.raises(ExecutionError, match="file_type_or_size_rejected"):
        read_file(record, STEP_RESULT_MAX_BYTES)


def test_bounded_installed_inventory_record(tmp_path):
    """A resolved GPU inventory exceeds the small-document cap without being unbounded."""
    # ReviewedFile shapes matching long installed environment paths. Their content
    # observations are checked by readiness, not this byte-boundary unit test.
    files = [
        dict(
            path=str(
                tmp_path
                / "environment/lib/python3.12/site-packages/torch"
                / ("module_" + str(i) + ".py")
            ),
            checksum=digest(str(i)),
            byte_size=100,
        )
        for i in range(6000)
    ]
    bundle = {"execution_policy": {"reviews": [{"files": files}]}}
    path = tmp_path / "large-bundle.json"
    path.write_text(json.dumps(bundle))
    record = pinned(path).model_dump(mode="json")
    assert 1048576 < record["byte_size"] < 8388608
    # General small reviewed documents retain their existing limit.
    with pytest.raises(ValueError, match="research_installation_file_bounds"):
        deployment.read_record(record)
    installed = deployment.InstalledProfiles(record, None)
    assert installed.current() == bundle
    path.write_bytes(b" " * (8388608 + 1))
    with pytest.raises(ValueError, match="research_installation_file_bounds"):
        deployment.InstalledProfiles(pinned(path), None)
