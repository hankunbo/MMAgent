"""Deterministic B02 owned schemas; frozen contracts v1 are never written here."""

import argparse
import json
from pathlib import Path

from db.knowledge.models import ClinicalClaim

from .models import (
    CandidateRequest,
    CandidateSourceBinding,
    CandidateSubmission,
    ClinicalClaimCandidate,
    N124RuleExecution,
    RuleExecution,
    StructuredOutput,
)
from .n124_models import (
    N124CandidateRequest,
    N124ClinicalClaimCandidate,
    N124DraftOutput,
    N124Metadata,
    N124Scope,
    N124StructuredOutput,
)


def schemas():
    root = Path(__file__).resolve().parent
    return {
        **{
            root / "schemas" / (model.__name__ + ".schema.json"): model
            for model in (
                N124CandidateRequest,
                N124DraftOutput,
                N124Metadata,
                N124Scope,
                N124StructuredOutput,
                N124ClinicalClaimCandidate,
                N124RuleExecution,
                CandidateRequest,
                CandidateSourceBinding,
                CandidateSubmission,
                ClinicalClaimCandidate,
                RuleExecution,
                StructuredOutput,
            )
        },
        root.parent / "knowledge_review/schemas/ClinicalClaim.schema.json": ClinicalClaim,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    for path, model in schemas().items():
        expected = model.model_json_schema()
        if args.write:
            path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n")
        else:
            if json.loads(path.read_text()) != expected:
                raise SystemExit("schema mismatch: " + str(path))
    print(f"{len(schemas())} owned schemas verified")


if __name__ == "__main__":
    main()
