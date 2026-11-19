from typing import Literal

from packages.grounding.models import Value


class EvaluationPreparation(Value):
    schema_version: Literal["evaluation-preparation/1.0.0"] = "evaluation-preparation/1.0.0"
    status: Literal["preparation_only"] = "preparation_only"
    final_test_unlocked: Literal[False] = False
    selection_uses_labels: Literal[False] = False
    patient_independence_proven: Literal[False] = False
    primary_question: str
    proposed_metrics: tuple[str, ...]
    proposed_baselines: tuple[str, ...]
    decisions_pending: tuple[str, ...]
    prior_exposure_policy: Literal["used_or_unknown_not_independent_test"] = (
        "used_or_unknown_not_independent_test"
    )


class Registry(Value):
    schema_version: Literal["evaluation-registry/1.0.0"] = "evaluation-registry/1.0.0"
    stage: Literal["13A"] = "13A"
    entries: tuple[dict, ...]
    source_bindings: dict[str, str]
    preparation: EvaluationPreparation
    checksum_scope: Literal["selected_content_and_full_name_size_inventory"] = (
        "selected_content_and_full_name_size_inventory"
    )
    model_executions: Literal[0] = 0
    runtime_writes: Literal[0] = 0
    evaluation_results: tuple[()] = ()
