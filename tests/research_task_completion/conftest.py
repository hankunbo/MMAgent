"""One owned PG, real harmless 06/H and Memory; no network/model/patient tools."""

from datetime import timedelta
from types import SimpleNamespace

import pytest

from apps.memory_sources import SourceRegistry
from db import manage
from db.memory.admin import bootstrap_memory_principals, bootstrap_publication_principal
from db.memory.publication import PublicationRepository
from db.memory.repository import MemoryRepository
from db.runtime.repository.research_lifecycle import ResearchLifecycleRepository
from packages.memory_distillation import MemoryDistiller
from tests.database.knowledge_support_04a import free_port, prepare_database, sql
from tests.expert_agent.pg_support import runtime_fixture
from tests.research_task_completion.support import prepare_support
from tests.tool_execution.support import prepared


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("10E task qualification forbids external network")

    monkeypatch.setattr("urllib.request.OpenerDirector.open", forbidden)


@pytest.fixture(scope="module")
def task_db(database_lease, tmp_path_factory):
    with database_lease.cluster(free_port()):
        db = prepare_database()
        manage.bootstrap_runtime_principal(db.admin, manage.test_target("runtime").user)
        bootstrap_memory_principals(
            db.admin, *(manage.test_target(c).user for c in manage.MEMORY_CHANNELS)
        )
        login = "mmagent_test_memory_publisher"
        sql(db.admin, "CREATE ROLE mmagent_test_memory_publisher LOGIN NOINHERIT")
        bootstrap_publication_principal(db.admin, login)
        target = manage.ConnectionTarget(
            db.admin.host,
            db.admin.port,
            db.admin.database,
            login,
            "MMAGENT_TEST_PUBLICATION_PASSWORD",
            None,
        )
        root = tmp_path_factory.mktemp("research-task-owned")
        ready = prepared(root / "shared-tool")
        registry = SourceRegistry({})
        runtime = ResearchLifecycleRepository(manage.test_target("runtime"))
        content = MemoryRepository(
            manage.test_target("memory_writer"), capability="writer", sources=registry
        )
        supports = [
            prepare_support(
                root=root / str(i),
                ready=ready,
                registry=registry,
                repository=content,
                identity="independent-" + str(i),
                key="10e-support-" + str(i),
            )
            for i in range(2)
        ]
        assert len({s.proof.definition.profile for s in supports}) == 1
        assert len({s.request.scope.patient_hash for s in supports}) == 2
        assert len({s.request.artifacts[0].checksum for s in supports}) == 2
        distiller = MemoryDistiller(repository=content)
        candidate = distiller.write(
            request_key="task-success-pool",
            episode_keys=tuple(s.episode.record.key for s in supports),
        )
        # The original query Runtime/Planner is a distinct synthetic case and patient.
        consumer = runtime_fixture(root / "independent-query")
        consumer.binding = consumer.binding.model_copy(
            update={
                "profile": consumer.binding.profile.model_copy(update={"seconds": 3600}),
                "deadline": consumer.binding.started_at + timedelta(seconds=3600),
            }
        )
        scope = runtime.read_research_run(consumer.binding.run_id).request.scope
        assert scope.case_code not in {s.request.scope.case_code for s in supports}
        assert scope.patient_hash not in {s.request.scope.patient_hash for s in supports}
        repos = {
            c: PublicationRepository(
                manage.test_target("memory_" + c), capability=c, sources=registry, contexts=runtime
            )
            for c in ("writer", "reader", "reviewer")
        }
        publisher = PublicationRepository(
            target, capability="publisher", sources=registry, contexts=runtime
        )
        api_calls = {}
        for role, repo in {**repos, "publisher": publisher}.items():
            for operation in (
                "register_policy",
                "record_opinion",
                "publish",
                "read_publication",
                "read_opinion",
                "revoke_qualification",
                "read",
            ):
                method = getattr(repo, operation)
                label = role + "." + operation

                def tracked(*args, _method=method, _label=label, **kwargs):
                    api_calls[_label] = api_calls.get(_label, 0) + 1
                    return _method(*args, **kwargs)

                setattr(repo, operation, tracked)
        x = SimpleNamespace(
            db=db,
            api_calls=api_calls,
            root=root,
            ready=ready,
            registry=registry,
            contexts=runtime,
            content_writer=content,
            supports=supports,
            distiller=distiller,
            candidate=candidate,
            consumer=consumer,
            publisher=publisher,
            **repos,
        )
        yield x
