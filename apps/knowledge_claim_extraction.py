"""04B5 explicit service-owned batch CLI; defaults require offline structured fixtures."""

import argparse
import json
from pathlib import Path

from db.knowledge.repository import KnowledgeRepository
from db.manage import _target_from_environment
from packages.knowledge_claims.models import BatchPolicy
from packages.knowledge_claims.ports import (
    Budget,
    ConfiguredStructuredExtractor,
    FakeStructuredExtractor,
    load_llm_config,
)
from packages.knowledge_index.models import Filters
from packages.object_store import LocalObjectStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live-batch", action="store_true")
    mode.add_argument("--offline-fixture", type=Path)
    args = parser.parse_args()
    if args.request.stat().st_size > 262144:
        parser.error("request exceeds bound")
    request = json.loads(args.request.read_text())
    if set(request) != {"hits", "filters", "budget"} or not 1 <= len(request["hits"]) <= 64:
        parser.error("explicit hits, filters and budget required")
    policy = BatchPolicy.model_validate_json(json.dumps(request["budget"]))
    config = load_llm_config()
    if args.live_batch:
        extractor = ConfiguredStructuredExtractor(config)
    else:
        if args.offline_fixture.stat().st_size > 65536:
            parser.error("fixture exceeds bound")
        extractor = FakeStructuredExtractor(args.offline_fixture.read_bytes())
    repository = KnowledgeRepository(
        _target_from_environment("knowledge_editor"), capability="editor"
    )
    store = LocalObjectStore(args.object_root)
    budget = Budget(policy)
    filters = Filters.model_validate_json(json.dumps(request["filters"]))
    for hit in request["hits"]:
        result = repository.extract_claims(
            hit,
            store,
            extractor,
            budget,
            filters=filters,
            allow_live=args.live_batch,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        # Structured records and finite reasons only, never provider transcripts.
        if "records" in result:
            result["records"] = [r.model_dump(mode="json") for r in result["records"]]
        if "audit" in result:
            result["audit"] = result["audit"].model_dump(mode="json")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
