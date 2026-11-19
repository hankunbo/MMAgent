"""Pure interpreter for the shared, versioned JSON template expression language.

The SQL interpreter uses the identical expression tree. Medical/provider wording
lives only in query_template.json; migrations carry an immutable checked snapshot.
"""

import json
from pathlib import Path

PROGRAM = json.loads(Path(__file__).with_name("query_template.json").read_text())


def evaluate(node, context):
    if not isinstance(node, list):
        return node
    op, *args = node

    def ev(x):
        return evaluate(x, context)

    if op == "get":
        value = context.get(args[0])
        for key in args[1:]:
            value = value.get(key) if isinstance(value, dict) else None
        return value
    if op == "if":
        return ev(args[1]) if ev(args[0]) else ev(args[2])
    if op in ("map", "filter"):
        values = ev(args[0])
        results = [evaluate(args[2], {**context, args[1]: v}) for v in values]
        return (
            results if op == "map" else [v for v, keep in zip(values, results, strict=True) if keep]
        )
    values = [ev(x) for x in args]
    if op == "array":
        return values
    if op == "concat":
        return "".join(str(v) for v in values)
    if op == "join":
        return values[1].join(values[0])
    if op == "flatten":
        return [v for group in values[0] for v in group]
    if op == "unique":
        return sorted(set(values[0]))
    if op == "replace":
        return values[0].replace(values[1], values[2])
    if op == "default":
        return values[0] if values[0] is not None else values[1]
    if op == "eq":
        return values[0] == values[1]
    if op == "in":
        return values[0] in values[1]
    if op == "truth":
        return bool(values[0])
    if op == "or":
        return any(values)
    raise ValueError("unknown template instruction")


def render(plan, adapter):
    program = PROGRAM
    if plan["template_version"] == "4.0.0":
        from .targeted import template_program

        if adapter == "guideline":
            raise ValueError("targeted_guideline_search_not_supported")
        program = template_program()
    elif plan["template_version"] == "3.0.0":
        from .broad import template_program

        program = template_program()
    if plan["template_version"] != program["version"]:
        raise ValueError("unsupported template version")
    context = {"p": plan, "adapter": adapter}
    return tuple(evaluate(program["texts"], context)), evaluate(program["mode"], context)
