"""Task identity must not require a classifier or turn preparation into admission."""

import importlib

import pytest

from packages.grounding.models import digest, seal, validate


def intent(**changes):
    module = importlib.import_module("packages.research_runtime.task_context")
    values = dict(
        code="TASK:OPHTHALMIC_REVIEW",
        version="1.0.0",
        target_code="GOAL:GLAUCOMA",
        output_kind="binary_evidence",
        allowed_modalities=("OCT", "RNFLT", "VF", "structured_measurement", "color_fundus"),
        definition_refs=(
            dict(
                role="task_definition",
                code="SOURCE:TASK",
                version="1.0.0",
                checksum=digest("definition"),
            ),
        ),
    )
    return seal(module.TaskIntent, **dict(values, **changes))


def test_task_intent_has_no_dataset_or_fixed_model():
    value = intent()
    assert "dataset" not in type(value).model_fields
    assert "candidate" not in type(value).model_fields
    assert "invocation" not in type(value).model_fields
    assert value.target_code == "GOAL:GLAUCOMA"
    assert value.research_only is True


def test_task_target_and_output_semantics_are_explicit():
    with pytest.raises(ValueError, match="ordered_labels"):
        intent(output_kind="ordinal_evidence")
    assert intent(output_kind="ordinal_evidence", ordered_labels=("none", "early", "advanced"))
    with pytest.raises(ValueError, match="ordered_labels"):
        intent(ordered_labels=("none", "early"))
    with pytest.raises(ValueError, match="unique"):
        intent(allowed_modalities=("OCT", "OCT"))


def test_task_intent_does_not_accept_hidden_answers_or_bad_checksum():
    v = intent()
    with pytest.raises(ValueError):
        validate(type(v), dict(v.model_dump(mode="json"), answer="glaucoma"))
    with pytest.raises(ValueError):
        validate(type(v), v.model_copy(update=dict(target_code="GOAL:CHANGED")))


def test_research_run_reader_unknown_version_is_rejected():
    module = importlib.import_module("packages.research_runtime.task_context")
    with pytest.raises(ValueError, match="version"):
        module.read_context({"schema_version": "research-run/99.0.0"})
