"""04B6 offline CLI: authenticated roles, structured requests, no LLM or online Agent."""

import argparse
import json
from pathlib import Path

from db.knowledge import models as m
from db.knowledge.repository import KnowledgeRepository
from db.knowledge.repository.review import (
    OfflineReleaseService,
    admit,
    detect,
    resolve,
    submit_review,
)
from db.manage import _target_from_environment
from packages.knowledge_index.models import Filters
from packages.object_store import LocalObjectStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=(
            "admit",
            "detect",
            "review",
            "resolve",
            "publish",
            "replay",
            "withdraw",
            "deprecate",
        ),
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--object-root", type=Path)
    parser.add_argument("--reviewer-login", action="append", default=[])
    args = parser.parse_args()
    if args.request.stat().st_size > 262144:
        parser.error("bounded structured request required")
    request = json.loads(args.request.read_text())

    def repo(role):
        return KnowledgeRepository(
            _target_from_environment("knowledge_" + role),
            capability=role,
            reviewer_logins=tuple(args.reviewer_login) if role == "reviewer" else (),
        )

    def ref(value):
        return m.KnowledgeReference.model_validate_json(json.dumps(value))

    def store():
        if args.object_root is None:
            parser.error("object-root required for evidence validation")
        return LocalObjectStore(args.object_root)

    op = args.operation
    if op == "admit":
        result = admit(
            repo("editor"),
            ref(request["claim"]),
            request["hit"],
            store(),
            filters=Filters.model_validate_json(json.dumps(request["filters"])),
        )
    elif op == "detect":
        result = detect(repo("editor"), tuple(ref(r) for r in request["claims"]))
    elif op == "review":
        review = m.ExpertReview.model_validate_json(json.dumps(request["review"]))
        result = submit_review(
            repo("editor" if review.reviewer_role == m.ReviewerRole.EXPERT_AGENT else "reviewer"),
            review,
        )
        if request.get("advance_reviewed", False):
            result = repo("reviewer").review(
                review.subject_ref, review.reference(), review.policy_ref
            )
    elif op == "resolve":
        result = resolve(
            repo("editor"),
            ref(request["conflict"]),
            ref(request["review"]),
            version=request["version"],
        )
    elif op == "publish":
        result = OfflineReleaseService(repo("publisher"), repo("editor"), store()).publish(
            tuple(
                (ref(i["subject"]), ref(i["review"]), ref(i["policy"]), i["revision"])
                for i in request["items"]
            ),
            idempotency_key=request["idempotency_key"],
        )
    elif op == "replay":
        result = repo("editor").replay(ref(request["reference"]))
    elif op == "withdraw":
        result = repo("publisher").withdraw(ref(request["reference"]), request["reason"])
    else:
        result = repo("publisher").deprecate(
            ref(request["reference"]), expected_revision=request["revision"]
        )
    if isinstance(result, tuple):
        result = [s.model_dump(mode="json") for s in result]
    elif isinstance(result, m.KnowledgeSnapshot):
        result = result.model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
