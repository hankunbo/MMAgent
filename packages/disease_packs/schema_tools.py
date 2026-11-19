"""B03 schema ownership inventory; never generates contracts v1."""

import argparse
import json
from pathlib import Path

from db.knowledge.models import Release, ReleasePolicy
from packages.knowledge_review.research import KnowledgeUseRequest, ResearchScope
from packages.retrieval.knowledge.models import RetrievalContext
from packages.retrieval.knowledge.research import ResearchRetrievalContext

from .models import DatasetReference, DiseasePack, PackDefinition, ResearchDiseasePack


def schemas():
    root = Path(__file__).resolve().parents[1]
    return {
        **{
            root / "disease_packs/schemas" / f"{m.__name__}.schema.json": m
            for m in (DiseasePack, DatasetReference, PackDefinition, ResearchDiseasePack)
        },
        **{
            root / "knowledge_review/schemas" / f"{m.__name__}.schema.json": m
            for m in (Release, ReleasePolicy, ResearchScope, KnowledgeUseRequest)
        },
        **{
            root / "retrieval/knowledge/schemas" / f"{model.__name__}.schema.json": model
            for model in (RetrievalContext, ResearchRetrievalContext)
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    for path, model in schemas().items():
        expected = model.model_json_schema()
        if args.write:
            if not path.exists() or json.loads(path.read_text()) != expected:
                path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n")
        elif json.loads(path.read_text()) != expected:
            raise SystemExit("B03 schema drift: " + str(path))
    print(f"{len(schemas())} owned schemas verified")


if __name__ == "__main__":
    main()
