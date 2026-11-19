"""Display-only sharing proved by saved ImageBindings, never by equal bytes alone."""

import copy

from packages.grounding.models import digest, validate
from packages.llm_provider.models import ImageBinding

from .evidence_presentation import origin, pointer_part


def saved_bindings(report):
    values = {}

    def walk(value, pointer=""):
        if isinstance(value, dict):
            image = value.get("image")
            if isinstance(image, dict):
                try:
                    binding = validate(ImageBinding, image).model_dump(mode="json")
                except ValueError:
                    binding = None
                if binding:
                    code = binding["reference"]["code"]
                    previous = values.get(code)
                    if previous and previous[0] != binding:
                        raise ValueError("asset_shared_reference_conflict")
                    values[code] = (binding, pointer + "/image")
            for key, child in value.items():
                walk(child, pointer + "/" + pointer_part(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, pointer + "/" + str(index))

    walk(report)
    return values


def compact(report, items, validate_image):
    """Keep distinct Artifact IDs and eye/time/preprocessing contexts distinct.

    The returned identity map is computed here, not accepted from caller aliases.
    Original references and proof pointers survive as versioned display metadata.
    No source report, observation, score, or authorization is modified.
    """
    bindings, groups, identities, result = saved_bindings(report), {}, {}, []
    for original in items:
        item = copy.deepcopy(original)
        item.pop("shared_image_bindings", None)
        ref = item.get("reference", {})
        proof = bindings.get(ref.get("code")) if not ref.get("asset_code") else None
        signature = None
        if proof:
            binding, pointer = proof
            try:
                validate_image(item)
            except (ValueError, KeyError, TypeError):
                proof = None
            if (
                proof
                and ref == binding["reference"]
                and item.get("shape") == [binding["height"], binding["width"]]
                and item.get("checksum") == ref["checksum"]
            ):
                comparable = copy.deepcopy(binding)
                comparable["reference"].pop("code")
                signature = digest(
                    (
                        comparable,
                        {
                            k: item.get(k)
                            for k in (
                                "case_code",
                                "input_checksum",
                                "modality",
                                "laterality",
                                "timepoint",
                                "transform",
                                "limits",
                                "sampled_fraction",
                            )
                        },
                    )
                )
        if signature is None:
            result.append(item)
            continue
        member = dict(
            id=item["id"],
            reference=ref,
            title=item.get("title"),
            role=item.get("role"),
            origin=origin(report, pointer),
        )
        if signature not in groups:
            groups[signature] = (item, [member])
            identities[item["id"]] = {ref["code"]}
            result.append(item)
        else:
            shared, members = groups[signature]
            members.append(member)
            identities[shared["id"]].add(ref["code"])
            shared["shared_image_bindings"] = dict(
                schema_version="shared-saved-image/1.0.0",
                members=members,
                limitation="仅复用同一已保存视图的展示字节；各观察与医学贡献保持独立记录。",
            )
            shared["title"] = "共享输入视图（" + str(len(members)) + "项检查引用）"
    return result, identities


def reference_redundant_previews(assets, linked):
    """Unbound previews keep their identity, but can point to exact displayed bytes.

    This is NOT an observation alias or link. A view sent in an unsuccessful or
    unattempted check must not become an accepted observation through deduplication.
    Artifact outputs are always retained. Matching is only within one scoped report.
    """
    keys = (
        "checksum",
        "shape",
        "modality",
        "laterality",
        "timepoint",
        "transform",
        "limits",
        "sampled_fraction",
        "case_code",
        "input_checksum",
    )
    available = [
        a
        for a in assets
        if a["id"] in linked
        and a.get("kind") == "image"
        and a.get("display_status") == "ready"
        and a.get("shared_image_bindings")
    ]
    for item in assets:
        if (
            item["id"] in linked
            or item.get("kind") != "image"
            or item.get("display_status") != "ready"
            or item.get("reference", {}).get("asset_code")
        ):
            continue
        same = next(
            (
                a
                for a in available
                if all(a.get(k) == item.get(k) for k in keys)
                and a.get("data_uri") == item.get("data_uri")
            ),
            None,
        )
        if same:
            item.pop("data_uri")
            item["display_status"] = "shared_bytes_see_linked_asset"
            item["shared_display_asset_id"] = same["id"]
            item["display_note"] = (
                "相同视图字节在所指产物展示；本引用没有已保存观察关联，不计为新证据。"
            )
