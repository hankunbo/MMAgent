"""Read-only packaging of this report's saved images, never image regeneration."""

import base64
import json
import struct
from pathlib import Path
from urllib.parse import unquote, urlparse

from packages.contracts import Artifact
from packages.object_store import checksum

LIMIT = 4 * 1024 * 1024
TITLES = {
    "input": "本例原始SLO图像",
    "display": "本轮VLM实际观察图（线性显示变换）",
    "segman_overlay": "本轮SegMAN杯盘分割叠加图",
    "segman_disc": "本轮SegMAN视盘掩膜",
    "segman_cup": "本轮SegMAN视杯掩膜",
    "segman_rim": "本轮SegMAN盘沿掩膜",
    "vessel_mask": "本轮SLOctolyzer血管掩膜",
    "artery_mask": "本轮SLOctolyzer动脉掩膜",
    "vein_mask": "本轮SLOctolyzer静脉掩膜",
    "slo_disc": "本轮SLOctolyzer视盘掩膜（非SegMAN）",
    "component_disc": "本轮几何算法使用的视盘连通区域",
    "component_cup": "本轮几何算法使用的视杯连通区域",
}


def png_shape(raw):
    if not 33 <= len(raw) <= LIMIT or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("media_png_format_or_size")
    if raw[12:16] != b"IHDR":
        raise ValueError("media_png_header")
    width, height = struct.unpack(">II", raw[16:24])
    if not width or not height or width * height > 16777216:
        raise ValueError("media_png_pixel_limit")
    return [height, width]


def checked_file(root, uri, expected):
    parsed = urlparse(uri)
    if parsed.scheme not in ("", "file") or parsed.netloc:
        raise ValueError("media_local_uri")
    path = Path(unquote(parsed.path)).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("media_file_outside_run")
    if path.stat().st_size > LIMIT:
        raise ValueError("media_png_size")
    data = path.read_bytes()
    if checksum(data) != expected:
        raise ValueError("media_checksum_mismatch")
    png_shape(data)
    return data


def build(report, *, source_checksum, root, binding, runtime, objects):
    """Original Runtime binds input/SegMAN/VLM. Native masks bind saved result SHA."""
    from .generic_ports import read

    case = report["result"]["case_code"]
    image = Artifact.model_validate_json(json.dumps(binding["image"]))
    native = report.get("native_tools") or {}
    if (
        binding["case_code"] != case
        or image.scope.case_code != case
        or native.get("case_code", case) != case
        or native.get("image_checksum", image.checksum) != image.checksum
    ):
        raise ValueError("media_case_input_binding")
    images = {}

    def add(role, raw, reference, note):
        images[role] = dict(
            title=TITLES[role],
            checksum=checksum(raw),
            shape=png_shape(raw),
            reference=reference,
            note=note,
            data_uri="data:image/png;base64," + base64.b64encode(raw).decode("ascii"),
        )

    def artifact_bytes(a):
        if a.scope.case_code != case or runtime.read_case_artifact(case, a.asset_code) != a:
            raise ValueError("media_artifact_owner_binding")
        raw = objects.read_bounded(a.uri, LIMIT)
        if checksum(raw) != a.checksum:
            raise ValueError("media_artifact_checksum")
        return raw

    add(
        "input",
        artifact_bytes(image),
        image.model_dump(mode="json"),
        "保存的原始输入；未增强、未重新推理。",
    )
    ports = read(Path(root), binding, runtime, objects)
    if ports:
        assets, raws, _ = ports
        for key, role in {
            "overlay": "segman_overlay",
            "disc_mask": "segman_disc",
            "cup_mask": "segman_cup",
            "rim_mask": "segman_rim",
        }.items():
            if key in assets:
                add(
                    role,
                    raws[key],
                    assets[key].model_dump(mode="json"),
                    "原执行已保存产物，不是本次重新分割。",
                )
    observations = (report.get("original_review") or {}).get("material", {}).get("observations", [])
    for observation in observations:
        frame = observation.get("image")
        if not frame:
            continue
        ref = frame["reference"]
        a = runtime.read_case_artifact(case, ref["code"])
        if a is None or a.checksum != ref["checksum"] or image.asset_code not in a.derived_from:
            raise ValueError("media_visual_image_binding")
        raw = artifact_bytes(a)
        if "display" in images and images["display"]["checksum"] != a.checksum:
            raise ValueError("media_multiple_display_images_need_explicit_mapping")
        add(
            "display",
            raw,
            a.model_dump(mode="json"),
            "这是本轮VLM使用的显示变换图，非原图、非新证据。",
        )
    for tool in ("vessels", "avod"):
        saved = native.get("results", {}).get(tool)
        if not saved or saved["execution"].get("exit_code") != 0:
            continue
        obs = saved["observation"]
        if "sha256:" + obs["input"]["sha256"] != image.checksum:
            raise ValueError("media_native_input_binding")
        for key, output in obs["outputs"].items():
            role = "slo_disc" if key == "disc_mask" else key
            if role not in TITLES:
                continue
            digest = "sha256:" + output["sha256"]
            raw = checked_file(root, output["uri"], digest)
            add(
                role,
                raw,
                dict(tool=tool, output_role=key, checksum=digest, input_checksum=image.checksum),
                "原报告记录的工具掩膜，已核输出摘要及本例输入。",
            )
    saved = native.get("results", {}).get("rim")
    if saved and saved["execution"].get("exit_code") == 0:
        argv = saved["execution"]["argv"]
        for key, role in (("disc_mask", "component_disc"), ("cup_mask", "component_cup")):
            flag = "--" + key.replace("_", "-")
            uri = argv[argv.index(flag) + 1]
            digest = "sha256:" + saved["observation"]["inputs"][key]["sha256"]
            component = native["component_views"][key.split("_")[0]]
            if (
                component["output"] != digest
                or component["original"] != images["segman_" + key.split("_")[0]]["checksum"]
            ):
                raise ValueError("media_component_input_binding")
            add(
                role,
                checked_file(root, uri, digest),
                dict(checksum=digest, tool="rim", input_role=key),
                "已保存的最大连通区域计算视图；不是替换原掩膜，也不增加独立证据。",
            )
    return dict(
        schema_version="evidence-media/1.0.0",
        source_checksum=source_checksum,
        case_code=case,
        input_checksum=image.checksum,
        images=images,
        missing=[],
        scope="historical_display_not_current_execution_authorization",
    )


def validate_images(packet):
    if len(packet.get("images", {})) > 32:
        raise ValueError("media_image_count")
    for item in packet.get("images", {}).values():
        value = item["data_uri"]
        prefix = "data:image/png;base64,"
        if not value.startswith(prefix) or len(value) > LIMIT * 2:
            raise ValueError("media_data_uri")
        raw = base64.b64decode(value[len(prefix) :], validate=True)
        if checksum(raw) != item["checksum"] or png_shape(raw) != item["shape"]:
            raise ValueError("media_image_integrity")


def acquisition_root(report, run_key):
    """Resolve saved Review ancestry, not arbitrary paths from a report.

    A text-only successor owns the new scores but its native masks still belong
    to the original acquisition. No current execution permission is inferred.
    """
    from packages.grounding.models import digest

    from .generic_input import directory
    from .generic_run import export

    root = directory(run_key)
    binding = json.loads((root / "binding.json").read_bytes())
    path = root / "input.json"
    selection = json.loads(path.read_bytes()) if path.exists() else {}
    version = selection.get("schema_version", "")
    if version.startswith("generic-definition-review-successor/"):
        return definition_acquisition_root(report, run_key, selection, binding)
    if version.startswith("generic-capability-successor/"):
        return capability_acquisition_root(report, run_key, selection, binding)
    if not version.startswith("generic-review-selection/"):
        return root, binding
    if version not in {"generic-review-selection/1.0.0", "generic-review-selection/1.1.0"}:
        raise ValueError("media_review_selection_version")
    parent_key = selection.get("parent_key")
    if (
        selection.get("key") != run_key
        or selection.get("review_key") != run_key
        or not isinstance(parent_key, str)
        or parent_key == run_key
    ):
        raise ValueError("media_review_parent_binding")
    parent = directory(parent_key)
    spec = json.loads((parent / "input.json").read_bytes())
    claim = json.loads((parent / "review-successor.json").read_bytes())
    parent_binding = json.loads((parent / "binding.json").read_bytes())
    if (
        spec.get("key") != parent_key
        or not spec.get("schema_version", "").startswith("fairvlmed-run-input/")
        or selection.get("input_checksum") != digest(spec)
        or claim != {k: v for k, v in selection.items() if k != "key"}
        or parent_binding != binding
    ):
        raise ValueError("media_review_acquisition_binding")
    # Original export validates the Runtime artifact/envelope/checksum. Avoid
    # load()/campaign assembly here: historical display must not dispatch work
    # or require the historical implementation to still be the current one.
    original = json.loads(Path(export(parent_key)["json"]).read_bytes())
    if original["result"]["case_code"] != report["result"]["case_code"] or original.get(
        "native_tools"
    ) != report.get("native_tools"):
        raise ValueError("media_review_acquisition_report")
    return parent, parent_binding


def collect(report, *, source_checksum, run_key):
    from .case_execution import owners
    from .generic_run import export

    # Reuse the original export owner to establish the report belongs to this run.
    original = json.loads(Path(export(run_key)["json"]).read_bytes())
    if original != report:
        raise ValueError("media_original_report_binding")
    root, binding = acquisition_root(report, run_key)
    runtime, objects, _ = owners()
    media = build(
        report,
        source_checksum=source_checksum,
        root=root,
        binding=binding,
        runtime=runtime,
        objects=objects,
    )
    if report.get("report_production"):
        from .report_production import PROFILE, media_links

        if report["report_production"].get("schema_version") != PROFILE:
            raise ValueError("media_report_production_version")
        return media_links(report, media)
    return media


def definition_acquisition_root(report, run_key, selection, binding):
    """Saved definition successor's native artifacts belong to its original case.

    Historical Runtime export binds the bytes. This display port does not load
    a provider, recompute definitions, or authorize current use of an old result.
    """
    from packages.grounding.models import digest

    from .generic_input import directory
    from .generic_run import export

    parent_key = selection.get("parent_key")
    if (
        selection.get("schema_version") != "generic-definition-review-successor/1.0.0"
        or selection.get("key") != run_key
        or not isinstance(parent_key, str)
        or parent_key == run_key
    ):
        raise ValueError("media_definition_parent_binding")
    parent = directory(parent_key)
    spec = json.loads((parent / "input.json").read_bytes())
    claim = json.loads((parent / "definition-review-successor.json").read_bytes())
    original_binding = json.loads((parent / "binding.json").read_bytes())
    history = selection.get("original_history") or {}
    scope = report.get("definition_revision") or {}
    if (
        claim != selection
        or spec.get("key") != parent_key
        or not spec.get("schema_version", "").startswith("fairvlmed-run-input/")
        or history.get("schema_version") != "generic-completed-definition-origin/1.0.0"
        or history.get("root_key") != parent_key
        or history.get("input_checksum") != digest(spec)
        or original_binding != binding
        or scope.get("original_history") != history
    ):
        raise ValueError("media_definition_origin_binding")
    saved = export(parent_key)
    original = json.loads(Path(saved["json"]).read_bytes())
    original_html = Path(saved["html"]).read_text()
    report_bundle = [original, dict(html=original_html, report_checksum=digest(original))]
    if (
        not original["result"].get("complete")
        or digest(original["result"]) != history.get("result_checksum")
        or digest(report_bundle) != history.get("report_checksum")
        or original["result"]["case_code"] != report["result"]["case_code"]
        or original.get("native_tools") != report.get("native_tools")
    ):
        raise ValueError("media_definition_original_report_binding")
    return parent, original_binding


def capability_acquisition_root(report, run_key, selection, binding):
    """Bind saved capability ancestry to the original acquisition's media only.

    No campaign/provider assembly, current-use qualification or new execution.
    Every successor claim and original report is checked before allowing paths
    within the original acquisition directory; arbitrary roots remain rejected.
    """
    from packages.grounding.models import digest

    from .generic_input import directory
    from .generic_run import export

    def read(path):
        return json.loads(path.read_bytes())

    scope = report.get("capability_revision") or {}
    parent_key = selection.get("parent_key")
    history = selection.get("original_history") or {}
    if (
        scope.get("schema_version") != "generic-capability-increment-report-scope/1.0.0"
        or scope.get("selection") != selection
        or selection.get("key") != run_key
        or not isinstance(parent_key, str)
        or parent_key == run_key
        or history.get("schema_version") != "generic-completed-definition-origin/1.0.0"
        or history.get("root_key") != parent_key
    ):
        raise ValueError("media_capability_report_binding")
    parent = directory(parent_key)
    spec, parent_binding = read(parent / "input.json"), read(parent / "binding.json")
    root_claim = read(parent / "definition-review-successor.json")
    if (
        spec.get("key") != parent_key
        or not spec.get("schema_version", "").startswith("fairvlmed-run-input/")
        or history.get("input_checksum") != digest(spec)
        or parent_binding != binding
    ):
        raise ValueError("media_capability_original_input_binding")
    node, seen = selection, set()
    while True:
        key, version = node.get("key"), node.get("schema_version")
        if not isinstance(key, str) or key in seen or len(seen) >= 4 or key == parent_key:
            raise ValueError("media_capability_ancestry_depth_or_cycle")
        seen.add(key)
        folder = directory(key)
        if (
            read(folder / "input.json") != node
            or read(folder / "binding.json") != binding
            or node.get("parent_key") != parent_key
            or node.get("original_history") != history
        ):
            raise ValueError("media_capability_ancestor_binding")
        if version in {"generic-capability-successor/1.0.0", "generic-capability-successor/1.1.0"}:
            if "continuation" in node or node != root_claim:
                raise ValueError("media_capability_root_claim")
            break
        if version not in {
            "generic-capability-successor/1.2.0",
            "generic-capability-successor/1.3.0",
        }:
            raise ValueError("media_capability_selection_version")
        continuation = node.get("continuation") or {}
        predecessor = continuation.get("predecessor_key")
        if not isinstance(predecessor, str) or continuation.get("root_claim") != root_claim:
            raise ValueError("media_capability_continuation_binding")
        prior = directory(predecessor)
        previous = read(prior / "input.json")
        if (
            previous.get("key") != predecessor
            or read(prior / "capability-continuation-successor.json") != node
            or continuation.get("inventory", {}).get("selection_checksum") != digest(previous)
        ):
            raise ValueError("media_capability_predecessor_binding")
        node = previous
    saved = export(parent_key)  # Original Runtime object/envelope/checksum owner.
    original = read(Path(saved["json"]))
    html = Path(saved["html"]).read_text()
    if (
        not original["result"].get("complete")
        or digest(original["result"]) != history.get("result_checksum")
        or digest([original, dict(html=html, report_checksum=digest(original))])
        != history.get("report_checksum")
        or original["result"]["case_code"] != report["result"]["case_code"]
        or original.get("native_tools") != report.get("native_tools")
    ):
        raise ValueError("media_capability_original_report_binding")
    return parent, parent_binding
