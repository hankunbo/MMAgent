"""Unsigned formal request sets use the same normalizer as controlled 05/06B admission."""

import pytest

from packages.grounding.models import validate
from packages.tool_workflow import FormalContext
from tests.tool_workflow.support import prepared


@pytest.mark.parametrize("field", ["artifacts", "environment", "patient_data_types"])
def test_formal_request_set_order_does_not_change_plan(tmp_path, field):
    f = prepared(tmp_path)
    original = f.compiler.compile(f.context, f.choice)
    request = f.context.request.model_dump(mode="json")
    assert len(request[field]) > 1
    request[field].reverse()
    changed = validate(FormalContext, {**f.context.model_dump(mode="json"), "request": request})
    assert f.compiler.compile(changed, f.choice) == original


@pytest.mark.parametrize("boundary", ["context", "nested_request"])
def test_copied_context_cannot_hide_unknown_approval_fields(tmp_path, boundary):
    f = prepared(tmp_path)
    if boundary == "context":
        context = f.context.model_copy(update={"approved": True})
    else:
        context = f.context.model_copy(
            update={"request": f.context.request.model_copy(update={"approved": True})}
        )
    with pytest.raises(ValueError, match="unknown fields"):
        f.compiler.compile(context, f.choice)
