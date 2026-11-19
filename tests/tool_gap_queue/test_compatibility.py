"""Original v1 formal Task/gap/document ports on the additive migration."""

from tests.database import test_runtime_repository_02b as original
from tests.database.test_runtime_repository_02b import database as database


def test_pg_original_formal_channels_and_requirements(database):
    original.test_contract_round_trips_idempotency_and_optimistic_graph_version(database)
    original.test_run_task_channels_contract_outputs_and_offline_review_are_separate(database)
