"""Bounded existing native tools; shared masks are executed once per input.

Uses the original native registration/resource owners and Runtime artifact writer.
It is not a new 06 executor or a qualification upgrade. No new model installation.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.contracts import Artifact
from packages.grounding.models import canonical
from packages.object_store import checksum

from .case_execution import process, register

VERSION = "13-full-evidence-tools-v2"


def run(sample, *, registered=None, before_dispatch=None):
    from apps import av_vascular_tool as av
    from apps import disc_geometry_tool_registry as geometry
    from apps import slo_tool_registry as slo
    from apps import vessel_branch_tool as branch
    from apps.case_memory_store import assembly
    from apps.glaucoma_case_evidence import execute, validate_output
    from packages.memory.archive import WorkflowMemory

    from .generic_execution_lock import scoped_lock

    if registered is not None:
        from .generic_input import validate_registered

        registered = validate_registered(sample, registered)
    directory, binding, runtime, objects = registered or register(sample)
    output = directory / VERSION
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    with scoped_lock(output / "lock"):
        image = Artifact.model_validate(binding["image"])
        memory, _ = assembly()
        workflow = WorkflowMemory(memory)
        hints = workflow.suggest(
            target="vertical_cdr",
            modality="SLO",
            legal_tools=("segman_slo_disc_cup_segmenter", "cdr_metrics"),
            query_case=binding["case_code"],
            candidate_policy="target-modality-prefilter/1.0.0",
        )
        for hint in hints["matches"]:
            workflow.recheck(hint["key"], hint["checksum"])
        write_once(
            output / "memory-selection.json",
            dict(
                hints=hints,
                selected=hints["matches"][0]["key"] if hints["matches"] else None,
                application="reuse_confirmed_current_case_segman_then_measure_and_fallback",
                old_patient_values_copied=False,
                model_rerun=False,
            ),
        )
        results = {}
        for task in ("vessels", "avod"):
            source = directory / "input.png"
            # The native owner verifies enabled/version/resources before launch.
            with slo.native_connection() as connection:
                manifest = connection.execute(
                    "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
                    (slo.TOOLS[task][0], slo.VERSION, slo.IMPLEMENTATION),
                ).fetchone()[0]
                if manifest is None:
                    raise ValueError("full_tool_unavailable")
            # Reuse the exact confirmed original model calls; only the CPU
            # measurement view changes in v2, not the model or image.
            if before_dispatch is not None:
                before_dispatch()
            execution = execute(task, source, directory / "13-full-evidence-tools-v1")
            if execution["manifest"] != manifest:
                raise ValueError("full_tool_manifest_changed")
            record = json.loads(Path(execution["call"]["outputs"]["observation"]).read_bytes())
            validate_output(task, record, execution, source, tuple(image.dimensions))
            results[task] = dict(execution=execution, observation=record)
            print(json.dumps(dict(sample=sample, task=task, state="confirmed")), flush=True)
        disc = directory / "segman-r03/prediction/disc.png"
        cup = directory / "segman-r03/prediction/cup.png"
        # Reuse only the exact masks already bound to this case by their original owner.
        from .case_execution import measurements_from_outputs

        measurements_from_outputs(
            binding, json.loads((directory / "segman-r03-plan.json").read_bytes())
        )
        from .component_view import select_component

        views = {}
        for name, path in (("disc", disc), ("cup", cup)):
            raw, policy = select_component(path.read_bytes())
            derived = output / (name + "-component.png")
            if derived.exists():
                if derived.read_bytes() != raw:
                    raise ValueError("component_view_changed")
            else:
                with derived.open("xb") as stream:
                    stream.write(raw)
            views[name] = policy
        disc, cup = output / "disc-component.png", output / "cup-component.png"
        write_once(output / "component-views.json", views)
        inputs = dict(
            artery_mask=directory / "13-full-evidence-tools-v1/avod/artery_mask.png",
            vein_mask=directory / "13-full-evidence-tools-v1/avod/vein_mask.png",
            disc_mask=disc,
        )
        for task in ("av_metrics", "ellipse", "rim", "branch"):
            job = output / task
            job.mkdir(mode=0o700, exist_ok=True)
            record_path = job / "observation.json"
            plan_path = job / "plan.json"
            with slo.native_connection() as c:
                module = av if task == "av_metrics" else branch if task == "branch" else geometry
                module.load_verified()
                code = module.CODE if task in {"av_metrics", "branch"} else module.TOOLS[task][0]
                current = c.execute(
                    "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
                    (code, module.VERSION, module.IMPLEMENTATION),
                ).fetchone()[0]
                if current is None:
                    raise ValueError("full_tool_unavailable")
                if not plan_path.exists():
                    plan = (
                        av.resolve(c, inputs, record_path)
                        if task == "av_metrics"
                        else branch.resolve(c, inputs["vein_mask"], disc, record_path)
                        if task == "branch"
                        else geometry.resolve(
                            c, task, disc, cup if task == "rim" else None, record_path
                        )
                    )
                    write_once(
                        plan_path,
                        dict(
                            plan=plan,
                            manifest=current,
                            image_checksum=image.checksum,
                            input_checksums={
                                k: checksum(p.read_bytes()) for k, p in inputs.items()
                            },
                            cup_checksum=checksum(cup.read_bytes()),
                        ),
                    )
            saved = json.loads(plan_path.read_bytes())
            if (
                saved["manifest"] != current
                or saved["image_checksum"] != image.checksum
                or saved["input_checksums"]
                != {k: checksum(p.read_bytes()) for k, p in inputs.items()}
                or saved["cup_checksum"] != checksum(cup.read_bytes())
            ):
                raise ValueError("full_measurement_input_changed")
            try:
                if before_dispatch is not None:
                    before_dispatch()
                execution = process(saved["plan"]["argv"], directory=job, seconds=120)
            except ValueError:
                receipt = job / "process-result.json"
                if not receipt.is_file():
                    raise  # An unknown launch is not a completed failure or permission to rerun.
                execution = json.loads(receipt.read_bytes())
                if execution["argv"] != saved["plan"]["argv"] or execution["exit_code"] == 0:
                    raise
                results[task] = dict(
                    execution=execution,
                    observation=None,
                    manifest=current,
                    missing_reason="algorithm_execution_failed",
                    diagnostic_checksum=checksum((job / "process.log").read_bytes()),
                )
                print(
                    json.dumps(dict(sample=sample, task=task, state="failed_retained")), flush=True
                )
                continue
            record = json.loads(record_path.read_bytes())
            if task == "av_metrics":
                av.validate(record)
                if {k: v["sha256"] for k, v in record["input_bindings"].items()} != saved[
                    "input_checksums"
                ]:
                    raise ValueError("full_av_output_binding")
            results[task] = dict(execution=execution, observation=record, manifest=current)
            print(json.dumps(dict(sample=sample, task=task, state="confirmed")), flush=True)
        body = dict(
            schema_version="full-evidence-tools/1.0.0",
            sample=sample,
            case_code=binding["case_code"],
            image_checksum=image.checksum,
            results=results,
            memory=json.loads((output / "memory-selection.json").read_bytes()),
            original_measurements=json.loads((directory / "measurements.json").read_bytes()),
            component_views=views,
            clinical_qualification=False,
            executor="original_native_registry_adapters",
        )
        write_once(output / "result.json", body)
        raw = canonical(body)
        namespace = (
            ":" + binding["execution_namespace"] if binding.get("execution_namespace") else ""
        )
        key = identity("TASK", VERSION + ":" + binding["case_code"] + namespace)
        existing = runtime.read_observation_bundle(key)
        uri = existing[0].uri if existing else objects.promote(objects.stage(raw), checksum(raw))
        artifact = Artifact.model_validate(
            dict(
                schema_version="1.0",
                contract_type="Artifact",
                asset_code=identity("ASSET", key + checksum(raw)),
                artifact_type_code="ARTIFACT:RESEARCH_FULL_TOOL_EVIDENCE",
                uri=uri,
                checksum=checksum(raw),
                media_type="application/json",
                qc_status="UNASSESSED",
                dimensions=[],
                measurements=[],
                derived_from=[image.asset_code],
                research_only=True,
                scope=image.scope,
                laterality=image.laterality,
                coordinate_space="PIXEL_2D",
                unit=None,
                extensions=dict(generator=VERSION),
                provenance=dict(
                    actor="TOOL_AGENT",
                    created_at=existing[0].provenance.created_at if existing else datetime.now(UTC),
                    generator_version=VERSION,
                    source_codes=[image.asset_code],
                    input_checksums=[image.checksum],
                ),
            )
        )
        runtime.register_observation_bundle((artifact,), key)
        if (
            runtime.read_observation_bundle(key) != (artifact,)
            or objects.read_bounded(uri, 4194304) != raw
        ):
            raise ValueError("full_tool_saved_binding")
        write_once(output / "artifact.json", artifact)
        return dict(sample=sample, tasks=len(results), artifact=artifact.asset_code)


if __name__ == "__main__":
    import sys

    for sample in sys.argv[1:]:
        print(json.dumps(run(sample)), flush=True)
