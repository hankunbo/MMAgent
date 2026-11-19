"""Opt-in wire subset projection. Domain schemas and returned values never change."""

import copy
import json
import re

from packages.grounding.models import digest

POLICY = "normalized_arrays/1.0.0"


class WireSchemaError(ValueError):
    def __init__(self, path, rule):
        super().__init__("wire_schema:" + "/".join(map(str, path)) + ":" + rule)
        self.path, self.rule = tuple(path), rule


def schema_nodes(schema, path=()):
    """Visit schema nodes, never enum strings, descriptions or patient values."""
    yield path, schema
    for key in ("properties", "$defs"):
        for name, node in schema.get(key, {}).items():
            yield from schema_nodes(node, (*path, key, name))
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        for index, node in enumerate(schema.get(key, [])):
            yield from schema_nodes(node, (*path, key, index))
    if isinstance(schema.get("items"), dict):
        yield from schema_nodes(schema["items"], (*path, "items"))


def normalize(schema):
    value = copy.deepcopy(schema)
    for path, node in list(schema_nodes(value)):
        if "prefixItems" in node:
            parts = node["prefixItems"]
            if (
                not parts
                or any(p != parts[0] for p in parts)
                or node.get("minItems") != len(parts)
                or node.get("maxItems") != len(parts)
                or "items" in node
            ):
                raise WireSchemaError(path, "non_equivalent_tuple")
            node["items"] = parts[0]
            del node["prefixItems"]
    preflight(value)
    return value


def preflight(schema):
    """Declared request subset, not a promise about a third-party model's support."""
    if schema.get("type") != "object":
        raise WireSchemaError((), "root_object_required")
    for path, node in schema_nodes(schema):
        for key in (
            "prefixItems",
            "allOf",
            "oneOf",
            "not",
            "if",
            "then",
            "else",
            "dependentSchemas",
            "patternProperties",
        ):
            if key in node:
                raise WireSchemaError((*path, key), "unsupported_subset_keyword")
        if "$ref" in node:
            ref = node["$ref"]
            if (
                not isinstance(ref, str)
                or not ref.startswith("#/$defs/")
                or ref[8:] not in schema.get("$defs", {})
            ):
                raise WireSchemaError((*path, "$ref"), "unresolved_reference")
        if node.get("type") == "object":
            if node.get("additionalProperties") is not False or set(
                node.get("required", [])
            ) != set(node.get("properties", {})):
                raise WireSchemaError(path, "strict_object_shape")
        if node.get("type") == "array" and not isinstance(node.get("items"), dict):
            raise WireSchemaError(path, "array_missing_items")
        if "enum" in node and not node["enum"]:
            raise WireSchemaError(path, "empty_enum")
    return dict(
        policy=POLICY,
        schema_checksum=digest(schema),
        subset_preflight_passed=True,
        remote_compatibility_not_proven=True,
    )


def apply(body, policy, mode):
    if policy != POLICY:
        raise ValueError("wire_policy_unknown")
    body = copy.deepcopy(body)
    schema = normalize(body["response_format"]["json_schema"]["schema"])
    if mode == "normalized_strict":
        body["response_format"]["json_schema"]["schema"] = schema
    elif mode == "json_object":
        body["response_format"] = {"type": "json_object"}
        # Explicit JSON prompt version tells the model to use this owner-derived contract.
        body["messages"][1]["content"].append(
            {
                "type": "text",
                "text": "OUTPUT_JSON_SCHEMA:\n"
                + json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
            }
        )
    else:
        raise ValueError("wire_mode_unknown")
    return body


def safe_schema_error(error, schema, key=""):
    """Return only schema-owned key/index paths and a closed rule vocabulary."""
    import ast

    message = error.get("message")
    if not isinstance(message, str) or len(message) > 8192:
        return dict(schema_paths=[], schema_rules=[])
    # Secret-bearing responses can still be classified, but never contribute paths.
    paths = []
    if not (key and key in message):
        for literal in re.findall(r"context\s*=\s*(\([^\n)]{0,1024}\))", message, re.I)[:4]:
            try:
                parts = ast.literal_eval(literal)
                if not isinstance(parts, tuple) or len(parts) > 24:
                    continue
                node = schema
                for part in parts:
                    if isinstance(node, dict) and type(part) is str and part in node:
                        node = node[part]
                    elif isinstance(node, list) and type(part) is int and 0 <= part < len(node):
                        node = node[part]
                    else:
                        raise ValueError()
                if (tuple(parts), node) in list(schema_nodes(schema)):
                    paths.append(list(parts))
            except (ValueError, SyntaxError, TypeError, KeyError):
                continue
    rules = {
        "array_missing_items": "array schema missing items",
        "required_property": "required",
        "additional_properties": "additionalproperties",
        "unsupported_keyword": "not supported",
        "invalid_schema": "invalid schema",
    }
    return dict(
        schema_paths=paths,
        schema_rules=[name for name, text in rules.items() if text in message.lower()],
    )
