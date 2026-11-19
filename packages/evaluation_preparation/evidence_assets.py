"""Report-bound multi-asset presentation adapters; no medical/model-name lookup."""

import base64
import copy
import json
import math

from packages.grounding.models import digest
from packages.object_store import checksum

from .evidence_media import LIMIT, png_shape
from .evidence_presentation import origin, pointer_part, resolve, sanitize
from .presentation_contracts import EvidenceAssets
from .presentation_contracts import check as validate_envelope

VERSION = "evidence-assets/1.0.0"


def image_payload(item):
    uri = item.get("data_uri", "")
    prefix = "data:image/png;base64,"
    if not isinstance(uri, str) or not uri.startswith(prefix) or len(uri) > LIMIT * 2:
        raise ValueError("asset_image_uri")
    try:
        data = base64.b64decode(uri[len(prefix) :], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("asset_image_base64") from error
    if checksum(data) != item.get("checksum") or png_shape(data) != item.get("shape"):
        raise ValueError("asset_image_integrity")


def validate_asset(item):
    kind = item.get("kind")
    if kind == "image":
        image_payload(item)
    elif kind == "image_sequence":
        frames = item.get("frames", [])
        if not frames or len(frames) > 128:
            raise ValueError("asset_sequence_size")
        for frame in frames:
            image_payload(frame)
    elif kind == "table":
        columns, rows = item.get("columns", []), item.get("rows", [])
        if not columns or len(columns) > 100 or len(rows) > 1000:
            raise ValueError("asset_table_size")
        if any(len(r) != len(columns) for r in rows) or any(
            isinstance(v, (dict, list)) for r in rows for v in r
        ):
            raise ValueError("asset_table_cells")
    elif kind == "series":
        values = item.get("points", [])
        if not values or len(values) > 10000:
            raise ValueError("asset_series_size")
        if any(
            len(p) != 2 or any(type(x) not in (float, int) or not math.isfinite(x) for x in p)
            for p in values
        ):
            raise ValueError("asset_series_values")
    else:
        return "unsupported_media_kind"
    return None


def reference_values(value):
    """Exact identifiers/digests only, never title similarity or string prefixes of checks."""
    result = set()
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "sha256" and isinstance(v, str):
                result.add(v if v.startswith("sha256:") else "sha256:" + v)
                continue
            if k in {
                "code",
                "asset_code",
                "checksum",
                "input_checksum",
                "full_record",
            } and isinstance(v, str):
                result.add(v)
            elif k == "value" and isinstance(v, str):
                try:
                    result.update(reference_values(json.loads(v)))
                except (ValueError, TypeError):
                    pass
            else:
                result.update(reference_values(v))
    elif isinstance(value, list):
        for v in value:
            result.update(reference_values(v))
    return result


def asset_identity(item):
    reference = item.get("reference", {})
    # Equal bytes in two eyes/timepoints do not make distinct Artifact IDs interchangeable.
    code = reference.get("asset_code") or reference.get("code")
    return {code} if code else {item.get("checksum")} - {None}


def attach(packet, report, media):
    """Upgrade legacy roles as distinct IDs; new links must cite exact report pointers."""
    assets, links, diagnostics = [], [], []
    if media is None:
        packet["assets"] = dict(
            schema_version=VERSION,
            items=[],
            links=[],
            diagnostics=[dict(code="media_not_packaged")],
        )
        return packet
    if media.get("source_checksum") != packet["source_checksum"]:
        raise ValueError("asset_report_binding")
    for key in ("case_code", "input_checksum"):
        if packet.get(key) is not None and media.get(key) != packet[key]:
            raise ValueError("asset_" + key + "_binding")
    version = media.get("schema_version")
    shared_identities = {}
    if version == "evidence-media/1.0.0":
        items = [
            dict(v, id="legacy:" + k, kind="image", role=k)
            for k, v in media.get("images", {}).items()
        ]
        if len(items) > 128:
            raise ValueError("asset_package_size")
        from .shared_evidence_images import compact

        items, shared_identities = compact(report, items, image_payload)
    elif version == VERSION:
        validate_envelope(EvidenceAssets, media)
        items = copy.deepcopy(media.get("items", []))
    else:
        raise ValueError("asset_package_version")
    if len(items) > 128 or len(json.dumps(items)) > 32 * 1024 * 1024:
        raise ValueError("asset_package_size")
    seen = set()
    for item in items:
        if not isinstance(item.get("id"), str) or item["id"] in seen:
            raise ValueError("asset_duplicate_id")
        seen.add(item["id"])
        for k in ("case_code", "input_checksum"):
            if item.get(k) is not None and item[k] != packet.get(k):
                raise ValueError("asset_item_scope")
        try:
            clean = sanitize(item)
            status = validate_asset(clean)
        except (ValueError, KeyError, TypeError) as error:
            status = str(error)
            clean = sanitize(
                {k: v for k, v in item.items() if k not in {"data_uri", "frames", "rows", "points"}}
            )
        if status:
            # Retain the reference and failure, not unsafe bytes or active URLs.
            clean = {
                k: v for k, v in clean.items() if k not in {"data_uri", "frames", "rows", "points"}
            }
            diagnostics.append(dict(code=status, asset_id=item["id"]))
        clean["display_status"] = status or "ready"
        assets.append(clean)
    by_id = {a["id"]: a for a in assets}
    if version == VERSION:
        for link in media.get("links", []):
            node, aid = link["node_id"], link["asset_id"]
            if aid not in by_id or node not in {n["id"] for n in packet["nodes"]}:
                raise ValueError("asset_link_endpoint")
            ptr = link["source_pointer"]
            value = resolve(report, ptr)
            if digest(value) != link["source_value_checksum"]:
                raise ValueError("asset_link_source_binding")
            target = by_id[aid]
            refs = reference_values(value)
            identity = asset_identity(target)
            if not refs.intersection(identity):
                raise ValueError("asset_link_not_referenced")
            check = next((c for c in packet["checks"].values() if c["node_id"] == node), None)
            if check:
                # A global report pointer cannot assign another observation's image to a check.
                bound = reference_values(check)
                bound.update(v for path in check["sections"]["dependencies"] for v in path)
                if not bound.intersection(identity):
                    raise ValueError("asset_link_check_binding")
            links.append(dict(link, origin=origin(report, ptr)))
    else:
        # Exact references in the saved observation or dependency list, not glaucoma role maps.
        for check in packet["checks"].values():
            direct = reference_values(check["sections"]["observation"])
            deps = {v for path in check["sections"]["dependencies"] for v in path}
            for asset in assets:
                identity = shared_identities.get(asset["id"], asset_identity(asset))
                if identity & (direct | deps):
                    links.append(
                        dict(
                            node_id=check["node_id"],
                            asset_id=asset["id"],
                            relation="observation_asset"
                            if identity & direct
                            else "dependency_asset",
                            origin=dict(
                                kind="exact_saved_reference",
                                matched=sorted(identity & (direct | deps)),
                            ),
                        )
                    )
        # Native tool inputs/outputs are linked by recorded content digests. No check-ID map.
        for name, saved in (report.get("native_tools") or {}).get("results", {}).items():
            body = saved.get("observation", {})
            for field, relation in (
                ("inputs", "tool_input"),
                ("input_bindings", "tool_input"),
                ("outputs", "tool_output"),
            ):
                if field not in body:
                    continue
                refs = reference_values(body[field])
                ptr = "/native_tools/results/" + pointer_part(name) + "/observation/" + field
                for asset in assets:
                    if asset.get("checksum") in refs:
                        links.append(
                            dict(
                                node_id="tool:" + name,
                                asset_id=asset["id"],
                                relation=relation,
                                origin=origin(report, ptr),
                            )
                        )
    linked = {x["asset_id"] for x in links}
    if version == "evidence-media/1.0.0":
        from .shared_evidence_images import reference_redundant_previews

        reference_redundant_previews(assets, linked)
    for asset in assets:
        if asset["id"] not in linked:
            diagnostics.append(
                dict(code="asset_available_without_check_binding", asset_id=asset["id"])
            )
    packet["assets"] = dict(
        schema_version=VERSION, items=assets, links=links, diagnostics=diagnostics
    )
    return packet
