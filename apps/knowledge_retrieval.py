"""04B7 service identities and offline replay CLI. Planner receives Published service only."""

import argparse
import json
from pathlib import Path

from db.knowledge.models import KnowledgeReference
from db.knowledge.repository.retrieval import published, replay, research, role


class SourceCandidateService:
    def __init__(self, repository, object_store):
        role(repository, ("editor", "reviewer"))
        self._repository = repository
        self._store = object_store

    def search_source_candidates(self, index_keys, query, context):
        return research(self._repository, index_keys, query, context)

    def replay_release(self, release):
        return replay(self._repository, release, self._store)


class PublishedKnowledgeService:
    """Read-only capability. No candidate/search/archive/review/publish method."""

    def __init__(self, repository, object_store):
        role(repository, ("reader",))
        self._repository = repository
        self._store = object_store
        self._cache = {}

    def retrieve_published_knowledge(
        self, release, query, context, config, *, run_id=None, **providers
    ):
        return published(
            self._repository,
            release,
            run_id,
            query,
            context,
            config,
            self._store,
            cache=self._cache,
            **providers,
        )

    def retrieve_for_run(self, runtime_repository, run_id, query, context, config, **providers):
        lock = runtime_repository.get_knowledge_lock(run_id)
        ref = KnowledgeReference.model_validate_json(json.dumps(lock["release_ref"]))
        return self.retrieve_published_knowledge(
            ref, query, context, config, run_id=run_id, **providers
        )

    def validate_rule_package_for_run(
        self, runtime_repository, run_id, graph, package, query, context, config, **providers
    ):
        from packages.rule_engine import validate_rule_package

        lock = runtime_repository.get_knowledge_lock(run_id)
        result = self.retrieve_for_run(
            runtime_repository, run_id, query, context, config, **providers
        )
        accepted = {
            h["artifact"].record.reference().to_versioned_reference() for h in result["hits"]
        }
        templates = [
            h["artifact"].record for h in result["hits"] if h["artifact"].record.kind == "template"
        ]
        template_match = any(
            t.graph_template is not None
            and t.graph_template.template_code == graph.template_ref.code
            for t in templates
        )
        if (
            not template_match
            or str(graph.graph_code) != lock["graph_code"]
            or graph != runtime_repository.get_graph_snapshot(graph.graph_code, graph.graph_version)
            or graph.template_ref.model_dump(mode="json") != lock["template_ref"]
            or package.template_ref != graph.template_ref
            or {(r.code, r.version, r.checksum) for r in package.rule_refs}
            != {(r["code"], r["version"], r["checksum"]) for r in lock["rule_refs"]}
            or not set(package.rule_refs).issubset(accepted)
            or tuple(package.provenance.knowledge_versions)
            != (result["release"].to_versioned_reference(),)
        ):
            raise ValueError("rule package references do not match current run locked release")
        return validate_rule_package(
            graph,
            package,
            expected_version=package.package_version,
            expected_checksum=package.package_checksum,
        )


class PlannerKnowledgeService:
    """The injected Planner port: every request resolves a persistent Runtime lock."""

    def __init__(self, published_service, runtime_repository):
        if type(published_service) is not PublishedKnowledgeService:
            raise TypeError("Planner requires the authenticated Published service")
        self._published = published_service
        self._runtime = runtime_repository

    def retrieve_published_knowledge(self, run_id, query, context, config, **providers):
        return self._published.retrieve_for_run(
            self._runtime, run_id, query, context, config, **providers
        )


def main():
    from db.knowledge.repository import KnowledgeRepository
    from db.manage import _target_from_environment
    from packages.object_store import LocalObjectStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("replay",))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    args = parser.parse_args()
    if args.request.stat().st_size > 8192:
        parser.error("bounded exact release request required")
    reference = KnowledgeReference.model_validate_json(args.request.read_bytes())
    service = SourceCandidateService(
        KnowledgeRepository(_target_from_environment("knowledge_editor"), capability="editor"),
        LocalObjectStore(args.object_root),
    )
    result = service.replay_release(reference)
    result["records"] = [r.model_dump(mode="json") for r in result["records"]]
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
