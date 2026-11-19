"""Public proposition vocabulary, explicitly distinct from v5 discovery.

These are search terms, not staging criteria, patient facts or medical approval.
Unknown goals fail closed; callers cannot put case text into a public query.
"""

import copy
import json

from packages.knowledge_connectors.ports import digest

from .broad import BROAD_CATALOG, broad_plan
from .broad import template_program as broad_template

TARGETS = {
    "GOAL:EARLY_GLAUCOMA": ("early glaucoma", "early-stage glaucoma", "mild glaucoma"),
    "GOAL:MODERATE_GLAUCOMA": ("moderate glaucoma", "moderate-stage glaucoma"),
    "GOAL:ADVANCED_GLAUCOMA": ("advanced glaucoma", "late-stage glaucoma", "severe glaucoma"),
}
TARGET_CODES = {k: "CONCEPT:TARGET:" + k.removeprefix("GOAL:") for k in TARGETS}
TARGET_CATALOG = {
    **BROAD_CATALOG,
    "source_version": "3.0.0",
    "policy_version": "patient-free/5.0.0",
    "concepts": {
        **BROAD_CATALOG["concepts"],
        **{TARGET_CODES[k]: sorted(v) for k, v in TARGETS.items()},
    },
}
TARGET_CHECKSUM = digest(
    json.dumps(TARGET_CATALOG, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
)


def template_program():
    result = copy.deepcopy(broad_template())
    result["version"] = "4.0.0"
    # Even a local title-only catalog must receive the target words, not silently
    # collapse the new proposition query back to generic glaucoma.
    result["texts"][2] = ["array", copy.deepcopy(result["texts"][3][1][2])]
    return result


def targeted_plan(provider_codes, *, target, **kwargs):
    from .models import KnowledgeQueryPlan

    if target not in TARGETS:
        raise ValueError("unsupported_public_proposition")
    # Preserve the existing bounded budgets and controlled topic normalization.
    value = broad_plan(provider_codes, **kwargs).model_dump(mode="json")
    value.update(query_plan_version="6.0.0", template_version="4.0.0")
    value["privacy"].update(
        version="patient-free/5.0.0", checksum=TARGET_CHECKSUM, term_version="3.0.0"
    )
    for term in value["diseases"] + value["concepts"]:
        term["terminology_version"] = "3.0.0"
    value["concepts"].append(
        dict(
            terminology_version="3.0.0",
            code=TARGET_CODES[target],
            label=TARGETS[target][0],
            synonyms=list(TARGETS[target][1:]),
        )
    )
    return KnowledgeQueryPlan.model_validate_json(json.dumps(value))
