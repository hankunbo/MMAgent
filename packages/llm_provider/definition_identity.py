"""Versioned request identities, never medical interpretation or response repair."""

from copy import deepcopy

PROMPT = "definition-draft-6.6.0"
PROMPTS = {PROMPT, "definition-draft-6.7.0"}
VERSION = "definition-request-identity/1.0.0"


def material_map(payload):
    target = payload.get("target")
    sources = payload.get("source_materials")
    if not isinstance(target, str) or not target or not isinstance(sources, list):
        raise ValueError("definition_identity_input")
    result = {}
    for source in sources:
        if (
            not isinstance(source, dict)
            or any(
                not isinstance(source.get(k), str) or not source[k]
                for k in ("code", "checksum", "text")
            )
            or source["code"] in result
        ):
            raise ValueError("definition_identity_sources")
        result[source["code"]] = source
    return result


def project(payload, schema):
    """Bind the advertised shape to this original request; old prompts do not opt in."""
    sources = material_map(payload)
    projected, wire_schema = deepcopy(payload), deepcopy(schema)
    projected["definition_identity_contract"] = dict(
        schema_version=VERSION,
        target=payload["target"],
        source_identities=[
            dict(source_code=k, source_checksum=s["checksum"]) for k, s in sources.items()
        ],
        meaning=(
            "Copy target literally, not a public query or examination name. "
            "Copy each paired source identity and verbatim quote; "
            "identity is not medical entailment or acceptance."
        ),
    )

    def bind(node):
        if isinstance(node, dict):
            props = node.get("properties", {})
            if "target" in props:
                props["target"]["const"] = payload["target"]
            if "source_code" in props and sources:
                props["source_code"]["enum"] = list(sources)
            if "source_checksum" in props and sources:
                props["source_checksum"]["enum"] = sorted({s["checksum"] for s in sources.values()})
            for value in node.values():
                bind(value)
        elif isinstance(node, list):
            for value in node:
                bind(value)

    bind(wire_schema)
    return projected, wire_schema


def failures(proposal, payload):
    """Check request-local consistency only; current source and definition owners still decide."""
    sources = material_map(payload)
    value = proposal.model_dump(mode="json") if hasattr(proposal, "model_dump") else proposal
    errors = []
    for i, draft in enumerate(value["definitions"]):
        if draft["target"] != payload["target"]:
            errors.append(
                dict(path=["definitions", i, "target"], type="exact_request_target_required")
            )
        for j, quote in enumerate(draft["excerpts"]):
            source = sources.get(quote["source_code"])
            path = ["definitions", i, "excerpts", j]
            if source is None:
                errors.append(
                    dict(path=path + ["source_code"], type="request_source_identity_required")
                )
                continue
            if quote["source_checksum"] != source["checksum"]:
                errors.append(
                    dict(path=path + ["source_checksum"], type="paired_source_checksum_required")
                )
            if quote["text"] not in source["text"]:
                errors.append(dict(path=path + ["text"], type="verbatim_request_quote_required"))
    return errors[:16]
