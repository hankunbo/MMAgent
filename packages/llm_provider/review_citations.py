"""Versioned wire projection of owner-validated task citation identities only."""

import copy

from packages.grounding.models import canonical, validate

PROMPT_VERSION = "review-batch-1.1.0"
PROMPT_VERSIONS = (
    PROMPT_VERSION,
    "review-batch-1.2.0",
    "review-batch-1.3.0",
    "review-batch-1.4.0",
    "review-batch-1.5.0",
    "review-batch-1.6.0",
    "review-batch-1.7.0",
    "review-batch-1.8.0",
    "review-direction-2.0.0",
    "review-direction-2.1.0",
    "review-direction-2.3.0",
    "review-direction-2.4.0",
    "review-direction-2.5.0",
    "review-direction-2.7.0",
    "review-direction-2.8.0",
    "review-direction-2.9.0",
    "review-direction-2.10.0",
)


def citation_array(node, codes):
    """Empty sets permit only []; never emit an invalid empty enum."""
    node["items"] = {"type": "string", **({"enum": list(codes)} if codes else {})}
    node["maxItems"] = min(node.get("maxItems", 32), len(codes))
    node["uniqueItems"] = True


def project(
    schema, payload, *, primary_status=False, exact_primary_status=False, definition_gate=False
):
    from packages.evidence_details.review_batches import ReviewBatchPayload

    projected = validate(ReviewBatchPayload, payload)
    tasks = projected.review_tasks.tasks
    observed = {o.reference.code: o for o in projected.material.observations}
    if not tasks:
        raise ValueError("review_citation_tasks_required")
    variants, table = [], []
    for task in tasks:
        observations = [task.observation.code]
        sources = list(dict.fromkeys(r.code for r in task.allowed_sources))
        item_name = schema["properties"]["items"]["items"]["$ref"].rsplit("/", 1)[-1]
        item = copy.deepcopy(schema["$defs"][item_name])
        props = item["properties"]
        props["assessment_id"]["enum"] = [task.assessment_id]
        citation_array(props["observation_citations"], observations)
        citation_array(props["reference_citations"], sources)
        availability = {}
        if primary_status:
            obs = observed[task.observation.code]
            available = obs.status == "obtained" and bool(obs.value and obs.value.strip())
            required = (
                "input_missing"
                if obs.status in {"missing", "unknown"}
                else "observation_ungradable"
                if obs.status == "ungradable"
                else None
            )
            props["observation_available"]["enum"] = [available]
            if required is not None:
                props["status"]["enum"] = [required]
            elif exact_primary_status:
                props["status"]["enum"] = [
                    s
                    for s in props["status"]["enum"]
                    if s not in {"input_missing", "observation_ungradable"}
                ]
            availability = dict(
                required_primary_status=required,
                required_observation_available=available,
            )
            if exact_primary_status:
                availability["allowed_primary_statuses"] = props["status"]["enum"]
        if definition_gate:
            # Mirror the existing owner admission condition, not a medical judgment.
            admitted = task.definition.interpretation_status == "installed" and task.applicable
            if not admitted:
                props["status"]["enum"] = [s for s in props["status"]["enum"] if s != "assessed"]
            availability.update(
                allowed_primary_statuses=props["status"]["enum"],
                interpretation_status=task.definition.interpretation_status,
                owner_applicable=task.applicable,
                assessed_permitted_by_definition=admitted,
            )
        variants.append(item)
        table.append(
            dict(
                assessment_id=task.assessment_id,
                observation_citations=observations,
                reference_citations=sources,
                **availability,
            )
        )
    # anyOf is in the existing normalized_arrays subset. Each branch binds its ID
    # and both citation fields together; no batch-wide union or positional mapping.
    schema["properties"]["items"]["items"] = {"anyOf": variants}
    return (
        "REVIEW_CITATION_CODES:\n"
        + canonical(dict(schema_version="review-citation-codes/1.0.0", tasks=table)).decode()
    )


def failure_paths(template, candidate):
    """Safe schema-owned paths only, after the original owner has rejected a row."""
    tasks = {t.assessment_id: t for t in template.tasks}
    paths = []
    for index, item in enumerate(candidate.items):
        task = tasks.get(item.assessment_id)
        if task is None:
            continue
        for field, allowed in (
            ("observation_citations", {task.observation.code}),
            ("reference_citations", {r.code for r in task.allowed_sources}),
        ):
            actual = getattr(item, field)
            if len(actual) != len(set(actual)) or not set(actual) <= allowed:
                paths.append(f"items[{index}].{field}")
    return tuple(paths)
