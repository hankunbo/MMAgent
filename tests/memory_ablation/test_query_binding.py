"""Exercise the actual query constructor before PG preparation is needed."""

from datetime import timedelta

from packages.planner_control.models import Binding
from tests.memory_ablation.support import query_binding
from tests.planner.support import binding


def test_query_profile_and_deadline_are_valid_at_construction():
    original = binding()
    query = query_binding(
        start=original.started_at,
        case_code=original.case_code,
        run_id=original.run_id,
        run_context_checksum=original.run_context_checksum,
        graph=original.graph,
    )
    assert query.profile.seconds == 3600
    assert query.deadline == original.started_at + timedelta(seconds=3600)
    assert Binding.model_validate_json(query.model_dump_json()) == query
