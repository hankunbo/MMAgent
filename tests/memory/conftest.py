"""One unified owned PG; original synthetic Runtime/acceptance/Planner setup."""

from types import SimpleNamespace

import pytest

from apps.memory_sources import ResearchSources, SourceRegistry
from db import manage
from db.memory.admin import bootstrap_memory_principals
from db.memory.repository import MemoryRepository
from tests.counterfactual.support import fixture
from tests.database.knowledge_support_04a import free_port, prepare_database
from tests.expert_agent.conftest import no_network_or_tools  # noqa: F401
from tests.expert_agent.pg_support import runtime_fixture


@pytest.fixture(scope="module")
def memory(database_lease, tmp_path_factory):
    with database_lease.cluster(free_port()):
        db = prepare_database()
        manage.bootstrap_runtime_principal(db.admin, manage.test_target("runtime").user)
        bootstrap_memory_principals(
            db.admin, *(manage.test_target(c).user for c in manage.MEMORY_CHANNELS)
        )
        before = {d: manage.recorded_migrations(db.migration, d) for d in manage.DOMAINS}
        manage.migrate(db.migration)
        assert before == {d: manage.recorded_migrations(db.migration, d) for d in manage.DOMAINS}
        pg = runtime_fixture(tmp_path_factory.mktemp("memory-objects"))
        x = fixture(pg=pg)
        owners = ResearchSources(
            fact_reader=x.reader,
            applicability=("USE:SYNTHETIC",),
            simulation=x.service,
            documents=pg.documents,
        )
        registry = SourceRegistry({pg.binding.run_id: owners})
        repos = {
            c: MemoryRepository(manage.test_target("memory_" + c), capability=c, sources=registry)
            for c in ("writer", "reviewer", "reader")
        }
        yield SimpleNamespace(db=db, pg=pg, x=x, owners=owners, registry=registry, **repos)
        import json
        import os
        from pathlib import Path

        evidence = os.environ.get("MMAGENT_10A_EVIDENCE_DIR")
        if evidence:
            output = Path(evidence) / "examples"
            output.mkdir(parents=True, exist_ok=True)
            names = {
                "episode": "roundtrip",
                "graph_snapshot": "graph-example",
                "trajectory_step": "step",
                "tool_experience": "tool",
                "counterfactual_pair": "simulation-pair",
                "recovery_experience": "recovery-item",
                "abstract_pattern": "two-case-pattern",
            }
            for name, key in names.items():
                value = repos["reader"].read(key, historical=True)
                (output / (name + ".json")).write_text(value.model_dump_json(indent=2) + "\n")
            for name, value in (
                ("review", repos["reader"].read_review("agent-opinion", historical=True)),
                ("embedding-index", repos["reader"].read_index("index", historical=True)),
            ):
                (output / (name + ".json")).write_text(value.model_dump_json(indent=2) + "\n")
            (output / "README.md").write_text(
                "# 匿名合成历史导出\n\n来自本次自有隔离PG及原领域接口。临时库已清理；"
                "这些JSON只解释历史，不能作为当前授权。全部非临床、无真实模型。\n"
            )
            (Path(evidence) / "migration-checksums.json").write_text(json.dumps(before, indent=2))
            (Path(evidence) / "owned-lease.json").write_text(
                json.dumps(
                    {"root": str(database_lease.root), "cluster_id": database_lease.cluster_id},
                    indent=2,
                )
            )
