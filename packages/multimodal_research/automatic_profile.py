"""Bind a public execution template to the original task, never a dataset label.

This selects search context, not medical definitions or current source authority.
The exact source catalog, method policies and budget limits are retained.
"""

import json
from copy import deepcopy
from pathlib import Path

from packages.grounding.models import digest

POLICY = "automatic-requirement-selection/1.0.0"


def derive(template, task):
    from .active_requirements import load
    from .requirement_catalog import projection, read

    template = Path(template).absolute()
    raw = json.loads(template.read_bytes())
    validated = load(template)
    projection(read(validated["requirement_catalog"]), task.target_code)
    if task.output_kind != "binary_evidence" or "automatic_requirement_binding" in raw:
        raise ValueError("automatic_requirement_template_or_question")
    value = deepcopy(raw)
    old_target = value["definition_expansion"]["public_target"]
    value["definition_expansion"].update(
        schema_version="campaign-definition-expansion/1.44.0",
        public_target_vocabulary="registered-task-public/1.0.0",
        public_task_checksum=task.checksum,
    )
    value["definition_expansion"]["public_target"] = task.target_code
    # A token projection expresses the requested research question. It is not a
    # new medical relation or a statement about the patient's actual diagnosis.
    query = task.target_code.split(":", 1)[1].replace("_", " ").lower()
    replacements = []
    for name, plan in value["definition_expansion"].get("passage_plans", {}).items():
        targets = [q for q in plan["queries"] if plan.get("query_intents", {}).get(q) == "target"]
        if not targets or old_target == task.target_code:
            continue
        remaining = [q for q in plan["queries"] if q not in targets]
        plan["queries"] = [query, *remaining]
        plan["query_intents"] = {
            q: kind for q, kind in plan["query_intents"].items() if q not in targets
        }
        plan["query_intents"][query] = "target"
        replacements.append(dict(plan=name, previous=targets, current=[query]))
    proof = dict(
        schema_version=POLICY,
        template=str(template),
        template_checksum=digest(raw),
        task_checksum=task.checksum,
        target=task.target_code,
        public_query_basis="registered_task_code_not_patient_diagnosis",
        replaced_query_terms=replacements,
        medical_catalog_changed=False,
        execution_authorized=False,
    )
    value["id"] = "automatic-requirements-" + digest((raw, task))[7:27]
    value["automatic_requirement_binding"] = proof
    return value, proof


def check(spec):
    from packages.dataset_adapters.multimodal.tasks import load_intent

    active = spec.get("active_acquisition")
    if not active:
        if "automatic_requirement_selection" in spec:
            raise ValueError("automatic_requirement_active_profile_missing")
        return
    raw = json.loads(Path(active["path"]).read_bytes())
    proof = spec.get("automatic_requirement_selection")
    if proof is None:
        if "automatic_requirement_binding" in raw:
            raise ValueError("automatic_requirement_selection_missing")
        return
    task, registry = load_intent(spec["task"], spec["registry_path"])
    if registry != spec["registry"] or task.model_dump(mode="json") != spec["intent"]:
        raise ValueError("automatic_requirement_original_task_changed")
    current, expected = derive(proof["template"], task)
    if proof != expected or raw != current:
        raise ValueError("automatic_requirement_derivation_changed")
