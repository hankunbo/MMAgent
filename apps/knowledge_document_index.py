"""04B4 service-owned CLI: build/update immutable indexes and bounded retrieval."""

import argparse
import json
from pathlib import Path

from db.knowledge.models import KnowledgeReference
from db.knowledge.repository import KnowledgeRepository
from db.manage import _target_from_environment
from packages.knowledge_index.models import ChunkPolicy, Filters, IndexConfig, ParsePolicy
from packages.knowledge_index.ports import FakeEmbedder, FakeReranker
from packages.object_store import LocalObjectStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=("build", "update", "candidate-query", "published-query", "replay")
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument(
        "--offline-fixture", action="store_true", help="Explicit synthetic providers only"
    )
    args = parser.parse_args()
    if args.request.stat().st_size > 262144:
        parser.error("request exceeds bound")
    request = json.loads(args.request.read_text())
    role = "reader" if args.operation == "published-query" else "editor"
    repository = KnowledgeRepository(_target_from_environment("knowledge_" + role), capability=role)
    store = LocalObjectStore(args.object_root)
    providers = {}
    if args.offline_fixture:
        providers["embedder"] = FakeEmbedder(request.get("dimension", 3))
        if request.get("reranker_fixture", False):
            providers["reranker"] = FakeReranker()
    if args.operation in ("build", "update"):
        if "reranker" in providers:
            del providers["reranker"]
        result = repository.build_document_index(
            request["receipt"],
            store,
            IndexConfig.model_validate_json(json.dumps(request["config"])),
            parse_policy=ParsePolicy.model_validate_json(
                json.dumps(request.get("parse_policy", {}))
            ),
            chunk_policy=ChunkPolicy.model_validate_json(
                json.dumps(request.get("chunk_policy", {}))
            ),
            applicability_refs=tuple(
                KnowledgeReference.model_validate_json(json.dumps(r))
                for r in request["applicability_refs"]
            ),
            targets=tuple(request.get("targets", ())),
            **providers,
        )
    elif args.operation == "replay":
        result = repository.replay_document_index(request["key"], store)
    else:
        query = (
            repository.query_published_index
            if role == "reader"
            else repository.query_candidate_index
        )
        result = query(
            request["key"],
            request["query"],
            Filters.model_validate_json(json.dumps(request["filters"])),
            **providers,
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
