"""Existing installed SLO recipe. Known failed steps are records, never fake measurements."""

import json
from datetime import UTC, datetime

from apps.vectorengine_single_case import write_once
from packages.grounding.models import digest


def confirmed_failure(name, out, image_checksum):
    """Only explicit original negative execution receipts authorize branch fallback."""
    if name == "segman":
        path = out / "segman-r03-process/process-result.json"
        plan = out / "segman-r03-plan.json"
        if not path.is_file() or not plan.is_file():
            return None
        receipt, binding = json.loads(path.read_bytes()), json.loads(plan.read_bytes())
        if binding["input_sha256"] != image_checksum.removeprefix("sha256:"):
            raise ValueError("generic_failure_input_changed")
        claim = json.loads((path.parent / "process-claim.json").read_bytes())
        if receipt["argv"] != claim["argv"]:
            raise ValueError("generic_failure_command_changed")
        if type(receipt["exit_code"]) is int and receipt["exit_code"] != 0:
            return dict(
                receipt=str(path), receipt_checksum=digest(receipt), exit_code=receipt["exit_code"]
            )
    elif name == "vascular":
        for task in ("vessels", "avod"):
            path = out / "13-full-evidence-tools-v1" / task / "execution.json"
            if not path.is_file():
                continue
            receipt = json.loads(path.read_bytes())
            if receipt["input_checksum"] != image_checksum:
                raise ValueError("generic_failure_input_changed")
            if type(receipt["exit_code"]) is int and receipt["exit_code"] != 0:
                return dict(
                    receipt=str(path),
                    receipt_checksum=digest(receipt),
                    exit_code=receipt["exit_code"],
                )
    return None


def acquire(spec, registered, *, recover=False, requirements=()):
    from .case_execution import segman
    from .comprehensive_tools import run as vascular
    from .fairclip_execution import execute as fairclip
    from .generic_control import Cancellation, guard
    from .generic_input import owner_key

    checker = None
    if spec.get("upstream_profile"):
        from packages.evidence_reuse.integration import DispatchCheck, memory_owner

        checker = DispatchCheck(spec, registered, memory=memory_owner())

    def before_dispatch():
        guard(spec["key"])
        if checker is not None:
            checker.check()
            guard(spec["key"])  # Owner revalidation may outlast cancellation/deadline.

    out, binding, _, _ = registered
    # This ledger is orchestration metadata. The original owners keep process attempts,
    # artifact binding, admission and unknown-execution recovery; it cannot authorize reruns.
    ledger = out / "tools"
    ledger.mkdir(exist_ok=True, mode=0o700)
    jobs = [
        (
            "segman",
            lambda: segman(
                spec["sample"],
                registered=registered,
                directory_adapter=True,
                isolated=True,
                system_path=True,
                before_dispatch=before_dispatch,
            ),
        ),
        (
            "vascular",
            lambda: vascular(
                spec["sample"], registered=registered, before_dispatch=before_dispatch
            ),
        ),
    ]
    if spec["fairclip_configuration"]:
        jobs.append(
            (
                "fairclip",
                lambda: fairclip(
                    spec["sample"],
                    registered=registered,
                    configuration=spec["fairclip_configuration"],
                    revision="generic-v1",
                    request_key=owner_key(spec, "fairclip"),
                    recover_only=recover and (out / "fairclip-generic-v1/request.json").exists(),
                    before_dispatch=before_dispatch,
                    cancel=Cancellation(spec["key"]),
                ),
            )
        )
    records = []
    for name, execute in jobs:
        guard(spec["key"])
        path = ledger / (name + ".json")
        if path.exists():
            value = json.loads(path.read_bytes())
            if value["input_checksum"] != binding["image"]["checksum"]:
                raise ValueError("generic_tool_ledger_input_conflict")
            if name == "segman" and value["status"] == "owner_returned":
                from .generic_ports import install

                install(registered)
            records.append(value)
            continue
        if name == "segman" and not (
            (checker is not None and checker.needs_segman())
            or any(
                r in requirements
                for r in (
                    "criterion:VCDR_GE_07",
                    "criterion:RIM_RADIAL_LT_01",
                    "association:DISC_SHAPE_CONTEXT",
                )
            )
        ):
            value = dict(status="not_required", reason="no_matching_source_requirement")
        elif name == "vascular" and not any(r.startswith("association:") for r in requirements):
            value = dict(status="not_required", reason="no_matching_source_requirement")
        elif name == "vascular" and not (out / "measurements.json").exists():
            value = dict(status="dependency_not_obtained", reason="disc_cup_measurements_absent")
        else:
            # Current owner validation, not an arbitrary exception, determines whether
            # an attempt is confirmed. Unknown errors stop here; never mask them as QC.
            try:
                result = execute()
                value = dict(status="owner_returned", result_checksum=digest(result))
            except ValueError:
                failure = confirmed_failure(name, out, binding["image"]["checksum"])
                if failure is None:
                    raise
                value = dict(
                    status="confirmed_execution_failure",
                    failure=failure,
                    fallback="source_backed_VLM",
                )
            # A source/reuse rejection is not an execution failure and cannot be
            # masked by an older process failure receipt.
            if checker is not None:
                checker.check()
        value.update(
            tool_step=name,
            case_code=binding["case_code"],
            input_checksum=binding["image"]["checksum"],
            recorded_at=datetime.now(UTC).isoformat(),
            fake_result_created=False,
        )
        write_once(path, value)
        records.append(value)
        if name == "segman" and value["status"] == "owner_returned":
            from .generic_ports import install

            install(registered)
    if not spec["fairclip_configuration"]:
        write_once(
            ledger / "fairclip-not-installed.json",
            dict(
                status="not_selected",
                reason="no_current_deployment_configuration_selected",
                fixed_slot_created=False,
            ),
        )
    if checker is not None:
        checker.finish(records)
    return records
