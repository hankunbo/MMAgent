"""Fail-closed bounded JSON Schema subset. No defaults, coercion, remote refs or code."""

import math

from packages.grounding.models import canonical

from .models import AdmissionError

KEYS = {
    "object": {"properties", "required", "additionalProperties"},
    "array": {"items", "minItems", "maxItems"},
    "string": {"minLength", "maxLength"},
    "number": {"minimum", "maximum"},
    "integer": {"minimum", "maximum"},
    "boolean": set(),
    "null": set(),
}


def inspect_schema(schema, depth=0):
    if depth > 8 or type(schema) is not dict:
        raise AdmissionError("parameter_schema_unsupported")
    kind = schema.get("type")
    if (
        type(kind) is not str
        or kind not in KEYS
        or set(schema) - KEYS[kind] - {"type", "title", "description", "enum", "default"}
    ):
        raise AdmissionError("parameter_schema_unsupported")
    if "enum" in schema and (
        type(schema["enum"]) is not list or not 1 <= len(schema["enum"]) <= 64
    ):
        raise AdmissionError("parameter_schema_unsupported")
    for key in ("minimum", "maximum"):
        if key in schema and (
            type(schema[key]) not in (int, float) or not math.isfinite(schema[key])
        ):
            raise AdmissionError("parameter_schema_unsupported")
    for low, high, limit in [("minItems", "maxItems", 64), ("minLength", "maxLength", 512)]:
        for key in (low, high):
            if key in schema and (type(schema[key]) is not int or not 0 <= schema[key] <= limit):
                raise AdmissionError("parameter_schema_unsupported")
        if schema.get(low, 0) > schema.get(high, limit):
            raise AdmissionError("parameter_schema_unsupported")
    if schema.get("minimum", -math.inf) > schema.get("maximum", math.inf):
        raise AdmissionError("parameter_schema_unsupported")
    if kind == "object":
        props, required = schema.get("properties"), schema.get("required", [])
        if (
            schema.get("additionalProperties") is not False
            or type(props) is not dict
            or len(props) > 64
            or type(required) is not list
            or any(type(k) is not str for k in required)
            or not set(required) <= props.keys()
            or len(required) != len(set(required))
        ):
            raise AdmissionError("parameter_schema_unsupported")
        for child in props.values():
            inspect_schema(child, depth + 1)
    if kind == "array":
        inspect_schema(schema.get("items"), depth + 1)


def validate_parameters(schema, value):
    if len(canonical(schema)) > 16384 or len(canonical(value)) > 16384:
        raise AdmissionError("parameter_byte_limit")
    inspect_schema(schema)

    def check(s, v):
        kind = s["type"]
        valid = {
            "object": type(v) is dict,
            "array": type(v) is list,
            "string": type(v) is str,
            "number": type(v) in (int, float),
            "integer": type(v) is int,
            "boolean": type(v) is bool,
            "null": v is None,
        }[kind]
        if not valid or ("enum" in s and not any(type(v) is type(e) and v == e for e in s["enum"])):
            raise AdmissionError("parameter_invalid")
        if kind == "object":
            if set(v) - s["properties"].keys() or not set(s.get("required", [])) <= v.keys():
                raise AdmissionError("parameter_invalid")
            for key, item in v.items():
                check(s["properties"][key], item)
        elif kind == "array":
            if not s.get("minItems", 0) <= len(v) <= s.get("maxItems", 64):
                raise AdmissionError("parameter_invalid")
            for item in v:
                check(s["items"], item)
        elif kind == "string":
            if not s.get("minLength", 0) <= len(v) <= s.get("maxLength", 512):
                raise AdmissionError("parameter_invalid")
        elif kind in {"number", "integer"}:
            if not math.isfinite(v) or not s.get("minimum", -math.inf) <= v <= s.get(
                "maximum", math.inf
            ):
                raise AdmissionError("parameter_invalid")

    check(schema, value)
