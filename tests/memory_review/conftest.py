"""One owned cluster, real Runtime/Writer/Distiller/Memory/Audit; no clinical data."""

from datetime import timedelta
from types import SimpleNamespace

import pytest

from apps.memory_execution_sources import ExecutionSources
from apps.memory_sources import SourceRegistry
from db import manage
from db.memory.admin import bootstrap_memory_principals, bootstrap_publication_principal
from db.memory.publication import PublicationRepository
from db.memory.repository import MemoryRepository
from db.runtime.repository.research_lifecycle import ResearchLifecycleRepository
from packages.grounding.models import digest
from packages.memory_distillation import MemoryDistiller
from packages.memory_writer import MemoryWriter
from tests.database.knowledge_support_04a import free_port, prepare_database, sql
from tests.expert_agent.conftest import no_network_or_tools  # noqa: F401
from tests.expert_agent.pg_support import runtime_fixture
from tests.memory.support import second_case


@pytest.fixture(scope="module")
def publication_db(database_lease, tmp_path_factory):
    with database_lease.cluster(free_port()):
        db = prepare_database()
        manage.bootstrap_runtime_principal(db.admin, manage.test_target("runtime").user)
        bootstrap_memory_principals(
            db.admin, *(manage.test_target(c).user for c in manage.MEMORY_CHANNELS)
        )
        login = "mmagent_test_memory_publisher"
        sql(db.admin, "CREATE ROLE mmagent_test_memory_publisher LOGIN NOINHERIT")
        bootstrap_publication_principal(db.admin, login)
        publisher_target = manage.ConnectionTarget(
            db.admin.host,
            db.admin.port,
            db.admin.database,
            login,
            "MMAGENT_TEST_PUBLICATION_PASSWORD",
            None,
        )
        pg = runtime_fixture(tmp_path_factory.mktemp("memory-publication-objects"))
        # This is a test run limit, not a medical threshold or real deployment setting.
        pg.binding = pg.binding.model_copy(
            update={
                "profile": pg.binding.profile.model_copy(update={"seconds": 3600}),
                "deadline": pg.binding.started_at + timedelta(seconds=3600),
            }
        )
        other = second_case(pg)
        consumer = second_case(pg, same_case=True)
        runtime = ResearchLifecycleRepository(pg.target)
        registry = SourceRegistry({})
        writer = MemoryRepository(
            manage.test_target("memory_writer"), capability="writer", sources=registry
        )
        episodes = []
        for n, p in enumerate((pg, other)):
            context = runtime.read_research_run(p.binding.run_id)
            runtime.reject_preparation(
                context, "10d-synthetic-failure-" + str(n), digest("synthetic failure")
            )
            source = ExecutionSources(
                runtime=runtime, run_id=context.run_id, applicability=("USE:SYNTHETIC",)
            )
            registry.owners[context.run_id] = source
            episodes.append(
                MemoryWriter(repository=writer, sources=source).write(
                    run_id=context.run_id, request_key="10d-episode-" + str(n)
                )
            )
        distiller = MemoryDistiller(repository=writer)
        candidate = distiller.write(
            request_key="10d-two-case-candidate", episode_keys=tuple(e.record.key for e in episodes)
        )
        single = distiller.write(
            request_key="10d-single-case", episode_keys=(episodes[0].record.key,)
        )
        repos = {
            c: PublicationRepository(
                manage.test_target("memory_" + c), capability=c, sources=registry, contexts=runtime
            )
            for c in ("writer", "reviewer", "reader")
        }
        publisher = PublicationRepository(
            publisher_target, capability="publisher", sources=registry, contexts=runtime
        )
        api_calls = {}
        for name, repo in {**repos, "publisher": publisher}.items():
            for operation in (
                "register_policy",
                "record_opinion",
                "read_opinion",
                "publish",
                "read_publication",
                "publication_keys",
                "revoke_qualification",
                "read",
            ):
                method = getattr(repo, operation)
                label = name + "." + operation

                def tracked(*args, _method=method, _label=label, **kwargs):
                    api_calls[_label] = api_calls.get(_label, 0) + 1
                    return _method(*args, **kwargs)

                setattr(repo, operation, tracked)
        assert len({s.case_code for s in candidate.record.sources}) == 2
        yield SimpleNamespace(
            db=db,
            api_calls=api_calls,
            pg=pg,
            other=other,
            consumer=consumer,
            contexts=runtime,
            registry=registry,
            candidate=candidate,
            single=single,
            episodes=episodes,
            distiller=distiller,
            content_writer=writer,
            publisher=publisher,
            **repos,
        )
