"""Observed SegMAN masks, whitelisted demographics and assistant visual review.

Original model run and old visible manifest are not rewritten. Supplemental facts
use the existing CASE Artifact/object/Audit transaction, not clinical validation.
"""

import argparse
import base64
import io
import json
import struct
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from apps import segman_tool_registry as registry
from apps.glaucoma_case_evidence import DIRECTORY, read_json
from apps.glaucoma_evidence_bindings import dump, write_new
from apps.research_single_case import credential_boundary
from db.runtime.repository.case_evidence import CaseEvidenceRepository
from db.runtime.repository.case_pipeline import identity
from packages.contracts import Artifact
from packages.dataset_adapters.reading import grayscale_image, npy
from packages.object_store import LocalObjectStore, checksum

DESTINATION = registry.ROOT / ".local/glaucoma-segman-case-v1"
PREDICTION = registry.DEPLOYMENT / "results/case-single-003"
SOURCE = Path("/opt/data/test/hf_cache/FairVLMed/Dataset/Training/data_06231.npz")
EXPECTED_SOURCE = "sha256:c37bc5f73496540d6371b415f46405c5b961a9fc03a3c4585def8e7e4fc26864"
EXPECTED_IMAGE = "sha256:2ba00b68ec05340ae68a7b475786bff05c2153344cd9ab67ae88e58c6b42c9cd"
FIELDS = ("age", "gender", "race", "ethnicity", "language", "maritalstatus")
LABELS = {
    "gender": {0: "女性", 1: "男性"},
    "race": {0: "Asian", 1: "Black", 2: "White"},
    "ethnicity": {0: "non-Hispanic", 1: "Hispanic"},
    "language": {0: "English", 1: "Spanish", 2: "Other"},
    "maritalstatus": {
        0: "Marriage or Partnered",
        1: "Single",
        2: "Divorced",
        3: "Widowed",
        4: "Legally Separated",
    },
}


def demographics(raw, expected_image):
    """Read ONLY image and six authorized fields; never open the glaucoma array."""
    if len(raw) > 4_194_304:
        raise ValueError("demographic_source_bound")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:

        def member(key):
            info = archive.getinfo(key + ".npy")
            if info.file_size > 2_097_152 or info.flag_bits & 1:
                raise ValueError("demographic_member_bound")
            return npy(archive.read(info))

        shape, dtype, pixels = member("slo_fundus")
        if (
            dtype != "|u1"
            or len(shape) != 2
            or checksum(grayscale_image(shape, pixels)) != expected_image
        ):
            raise ValueError("demographic_image_mismatch")
        fields = {}
        for key in FIELDS:
            if key + ".npy" not in archive.namelist():
                fields[key] = {
                    "category": "unavailable_current_input",
                    "label": "当前输入无法取证",
                    "case_value": None,
                    "display_value": None,
                    "reason": "源输入未提供该字段",
                }
                continue
            shape, dtype, body = member(key)
            if (
                shape != ()
                or (key == "age" and not dtype.startswith("<U"))
                or (key != "age" and dtype != "<i8")
            ):
                raise ValueError("demographic_scalar_schema")
            value = (
                body.decode("utf-32-le").rstrip("\x00")
                if key == "age"
                else struct.unpack("<q", body)[0]
            )
            missing = value in (-1, "", "Unknown", "unknown", "-1")
            if key == "age" and not missing and (not value.isdigit() or not 0 <= int(value) <= 130):
                raise ValueError("demographic_age_invalid")
            if key != "age" and not missing and value not in LABELS[key]:
                raise ValueError("demographic_code_unknown")
            fields[key] = {
                "category": "unavailable_current_input" if missing else "observed",
                "label": "当前输入无法取证" if missing else "已有结果",
                "case_value": None if missing else (int(value) if key == "age" else value),
                "display_value": None
                if missing
                else (int(value) if key == "age" else LABELS[key][value]),
                "source_member": key + ".npy",
                "unit": "year" if key == "age" else "category",
                "diagnostic_weight": None,
                "purpose": "demographic_background"
                if key in ("age", "gender")
                else "group_description_not_disease_vote",
            }
    fields["family_history"] = {
        "category": "unavailable_current_input",
        "label": "当前输入无法取证",
        "case_value": None,
        "display_value": None,
        "reason": "本次没有家族史记录",
    }
    return {
        "schema": "case-demographics/1.0.0",
        "visibility_policy": "case-demographics-supplement/1.0.0",
        "fields": fields,
        "source_checksum": checksum(raw),
        "image_checksum": expected_image,
        "mapping_source": "https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP#data",
        "old_visible_manifest_modified": False,
    }


def visual_review():
    return {
        "schema": "case-assistant-visual-review/1.0.0",
        "image_checksum": EXPECTED_IMAGE,
        "reviewer": "assistant_in_this_conversation",
        "method": "direct_image_visual_inspection",
        "vlm_tool_executed": False,
        "medical_validation_accepted": False,
        "observations": {
            "RNFL_REFLECTANCE": {
                "status": "已有观察·判读不确定",
                "summary": (
                    "原图可见盘周放射状明暗纹理，同时有横向条带和局部暗区；"
                    "无法可靠划定局灶RNFL缺损边界。"
                ),
                "finding": "indeterminate",
                "next_action": "保留本次视觉观察；以后用匹配SLO的VLM或阅片复核。",
            },
            "DISC_BLEED": {
                "status": "已有观察·判读不确定",
                "summary": (
                    "盘周可见细血管和局部暗影，但该灰度图中不能可靠区分小片出血、血管与成像暗影。"
                ),
                "finding": "indeterminate",
                "next_action": "当前不填写出血存在/不存在；以后补充适合判读的同眼图像或复核。",
            },
            "IMAGE_QC": {
                "status": "已有观察",
                "summary": "视盘区域和主要血管可见；存在横向条带、亮度不均和下方局部暗影。",
                "finding": "visible_artifacts",
                "next_action": "测量与原图叠加查看；无数值化质量分数。",
            },
        },
    }


def prepare():
    DESTINATION.mkdir(exist_ok=True)
    original = read_json(DIRECTORY / "case-evidence.json")
    if original["input"]["checksum"] != EXPECTED_IMAGE or original["sample"] != SOURCE.name:
        raise ValueError("supplement_selected_case_mismatch")
    raw = SOURCE.read_bytes()
    if checksum(raw) != EXPECTED_SOURCE:
        raise ValueError("supplement_source_drift")
    output = subprocess.run(
        [
            "/opt/data/test/envs/SLOctolyzer/bin/python",
            str(registry.ROOT / "Algorithm_Tool/segman_slo/inspect_outputs.py"),
            "--directory",
            str(PREDICTION),
            "--image",
            str(DIRECTORY / "input.png"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    check = json.loads(output.stdout)
    if "sha256:" + check["input_sha256"] != EXPECTED_IMAGE:
        raise ValueError("supplement_mask_input_mismatch")
    result = read_json(PREDICTION / "result.json")
    if (
        result["checkpoint_sha256"] != registry.WEIGHT_SHA
        or result["source_commit"] != registry.COMMIT
    ):
        raise ValueError("supplement_model_identity")
    value = {
        "schema": "glaucoma-case-supplement/1.0.0",
        "case_code": original["case_code"],
        "input_checksum": EXPECTED_IMAGE,
        "segman": result,
        "mask_check": check,
        "demographics": demographics(raw, EXPECTED_IMAGE),
        "visual_review": visual_review(),
        "observation_origin": "import_existing_successful_run_not_new_execution",
    }
    write_new(DESTINATION / "prepared.json", value)
    return value


def register():
    """Register real observed files, never rerun the segmentation model."""
    value = prepare()
    spec = registry.load_verified()
    with registry.native_connection() as c:
        manifest = registry.manifest(c)
    if manifest["provenance"]["installation_checksum"] != registry.checksum(spec):
        raise ValueError("supplement_registration_mismatch")
    original = read_json(DIRECTORY / "case-evidence.json")
    image = Artifact.model_validate(original["input"])
    repo = CaseEvidenceRepository(credential_boundary().resolve_target("runtime"))
    current = {a.asset_code: a for a in repo.get_case_bundle(value["case_code"]).artifacts}
    if current.get(image.asset_code) != image:
        raise ValueError("supplement_case_image_not_registered")
    from apps.patient_service_assembly import OBJECT_ROOTS

    store = LocalObjectStore(OBJECT_ROOTS["visible"])
    pending = DESTINATION / "artifacts.json"
    if not pending.exists():
        payloads = {
            name: (PREDICTION / (name + ".png")).read_bytes()
            for name in ("cup", "rim", "disc", "labels", "overlay")
        }
        payloads["observation"] = json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False
        ).encode()
        artifacts = []
        for port, body in payloads.items():
            uri = store.promote(store.stage(body), checksum(body))
            artifact = Artifact.model_validate(
                {
                    "schema_version": "1.0",
                    "contract_type": "Artifact",
                    "asset_code": identity(
                        "ASSET", value["case_code"] + "segman-supplement-v1" + port + checksum(body)
                    ),
                    "artifact_type_code": "ARTIFACT:CASE_SUPPLEMENT"
                    if port == "observation"
                    else "ARTIFACT:SLO_SEGMAN_" + port.upper(),
                    "uri": uri,
                    "checksum": checksum(body),
                    "media_type": "application/json" if port == "observation" else "image/png",
                    "qc_status": "PASS",
                    "dimensions": [] if port == "observation" else list(image.dimensions),
                    "measurements": [],
                    "derived_from": [image.asset_code],
                    "research_only": True,
                    "scope": {"scope_type": "CASE", "case_code": value["case_code"]},
                    "laterality": image.laterality,
                    "coordinate_space": "PIXEL_2D",
                    "unit": None,
                    "extensions": {
                        "component_task": "segman_supplement",
                        "output_port": "segman_" + port,
                    },
                    "provenance": {
                        "actor": "PLANNER",
                        "created_at": datetime.now(UTC).isoformat(),
                        "generator_version": "case-supplement-import-1.0.0",
                        "source_codes": [image.asset_code],
                        "input_checksums": [image.checksum],
                        "tool_versions": [
                            {
                                "code": "TOOL:SEGMAN:SLO",
                                "version": registry.VERSION,
                                "checksum": "sha256:" + registry.checksum(manifest),
                            }
                        ],
                    },
                }
            )
            artifacts.append(artifact)
        write_new(pending, {"artifacts": [dump(a) for a in artifacts]})
    artifacts = tuple(Artifact.model_validate(a) for a in read_json(pending)["artifacts"])
    for a in artifacts:
        if checksum(store.read_bounded(a.uri, 4_194_304)) != a.checksum:
            raise ValueError("supplement_staged_object_drift")
    repo.register_observation_bundle(
        artifacts, identity("TASK", value["case_code"] + "segman-case-supplement-v1")
    )
    write_new(
        DESTINATION / "registered.json",
        {
            "case_code": value["case_code"],
            "tool": registry.CODE,
            "manifest": manifest,
            "artifacts": [dump(a) for a in artifacts],
        },
    )
    return {"case_code": value["case_code"], "artifacts": len(artifacts)}


def attach_verified(report, repo, store):
    if not (DESTINATION / "registered.json").exists():
        return
    saved = read_json(DESTINATION / "registered.json")
    if saved["case_code"] != report["case_code"]:
        raise ValueError("supplement_cross_case")
    current = {a.asset_code: a for a in repo.get_case_bundle(report["case_code"]).artifacts}
    value = None
    for raw in saved["artifacts"]:
        a = Artifact.model_validate(raw)
        body = store.read_bounded(a.uri, 4_194_304)
        if (
            current.get(a.asset_code) != a
            or checksum(body) != a.checksum
            or a.derived_from != (report["input"]["asset_code"],)
            or a.provenance.tool_versions[0].checksum
            != "sha256:" + registry.checksum(saved["manifest"])
        ):
            raise ValueError("supplement_readback_mismatch")
        if a.media_type == "application/json":
            value = json.loads(body)
        else:
            report["images"][a.extensions["output_port"]] = (
                "data:image/png;base64," + base64.b64encode(body).decode()
            )
    if (
        value is None
        or value["input_checksum"] != report["input"]["checksum"]
        or value["case_code"] != report["case_code"]
    ):
        raise ValueError("supplement_input_binding")
    report["supplement"] = value
    report["supplement"]["artifact_codes"] = [a["asset_code"] for a in saved["artifacts"]]
    report["supplement"]["registered_tool"] = {
        "code": registry.CODE,
        "version": registry.VERSION,
        "implementation": registry.IMPLEMENTATION,
        "manifest_checksum": "sha256:" + registry.checksum(saved["manifest"]),
    }
    report["artifacts"].extend(saved["artifacts"])


def preview():
    """Offline view only; local references cannot pass store_view's DB source check."""
    from apps.evidence_graph_html import render_html, validate_snapshot
    from apps.glaucoma_case_view import project
    from apps.glaucoma_evidence_plan import checksum as document_checksum

    value = prepare()
    snapshot = read_json(registry.ROOT / "docs/exports/青光眼证据图.json")
    validate_snapshot(snapshot)
    report = snapshot["case_evidence"]
    if (
        report["input"]["checksum"] != value["input_checksum"]
        or report["case_code"] != value["case_code"]
    ):
        raise ValueError("preview_case_binding")
    report.pop("supplement", None)
    report["artifacts"] = [
        a
        for a in report["artifacts"]
        if a["extensions"].get("component_task") != "segman_supplement"
    ]
    codes = []
    for name in ("cup", "rim", "disc", "labels", "overlay"):
        path = PREDICTION / (name + ".png")
        code = "LOCAL_OBSERVATION:SEGMAN_" + name.upper()
        body = path.read_bytes()
        report["images"]["segman_" + name] = (
            "data:image/png;base64," + base64.b64encode(body).decode()
        )
        report["artifacts"].append(
            {
                "asset_code": code,
                "artifact_type_code": "LOCAL:SLO_SEGMAN_" + name.upper(),
                "scope": {"case_code": value["case_code"]},
                "uri": path.as_uri(),
                "checksum": checksum(body),
                "database_persisted": False,
                "extensions": {
                    "component_task": "segman_supplement",
                    "output_port": "segman_" + name,
                },
            }
        )
        codes.append(code)
    code = "LOCAL_OBSERVATION:CASE_SUPPLEMENT"
    report["artifacts"].append(
        {
            "asset_code": code,
            "artifact_type_code": "LOCAL:CASE_SUPPLEMENT",
            "scope": {"case_code": value["case_code"]},
            "uri": (DESTINATION / "prepared.json").as_uri(),
            "checksum": checksum((DESTINATION / "prepared.json").read_bytes()),
            "database_persisted": False,
            "extensions": {
                "component_task": "segman_supplement",
                "output_port": "segman_observation",
            },
        }
    )
    value["artifact_codes"] = [*codes, code]
    value["registered_tool"] = {
        "code": registry.CODE,
        "version": registry.VERSION,
        "registration_status": "pending_database_connection",
        "source": "verified_local_installation",
    }
    report["supplement"] = value
    report["supplement_persistence"] = "local_files_pending_database"
    snapshot.pop("case_graph_artifact", None)
    snapshot["database_readback"] = {
        "current": False,
        "basis": "previous_export_and_verified_new_local_files",
    }
    result = project(snapshot, report)
    result["title"] += " · SegMAN接入预览（新项尚未入库）"
    result["checksum"] = document_checksum({k: v for k, v in result.items() if k != "checksum"})
    output = registry.ROOT / "docs/exports/青光眼证据图_SegMAN接入预览.html"
    if not output.exists():
        with output.open("x") as f:
            f.write(render_html(result))
        write_new(output.with_suffix(".json"), result)
    return {"case_code": value["case_code"], "html": str(output)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "register", "preview"))
    args = p.parse_args()
    result = {"prepare": prepare, "register": register, "preview": preview}[args.action]()
    print(
        json.dumps({"case_code": result["case_code"], "action": args.action, "status": "completed"})
    )
