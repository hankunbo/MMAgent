"""Terminal batch supervision over the original registered case commands."""

import argparse
import json

from packages.evaluation_preparation import terminal_cohort as owner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    initialize = sub.add_parser("init")
    initialize.add_argument("--manifest", required=True)
    initialize.add_argument("--output", required=True)
    lease = sub.add_parser(
        "lease-preflight", help="Check supervision and child locks without API or tool dispatch."
    )
    lease.add_argument("--input", required=True)
    inspect = sub.add_parser(
        "inspect-multimodal",
        help="Read saved original case/review-successor mirrors; no API, DB or recovery.",
    )
    inspect.add_argument("--input", required=True)
    audit = sub.add_parser(
        "audit-recovery",
        help="Original-owner readonly recovery/Review/report audit; not expansion permission.",
    )
    audit.add_argument("--input", required=True)
    audit.add_argument("--report", required=True)
    audit.add_argument("--history-root", required=True)
    audit.add_argument(
        "--local-closure",
        help="Separate zero-request closure directory; --input remains the original failed run.",
    )
    archive = sub.add_parser(
        "archive-result",
        help="Prepare an original-owner historical snapshot; --commit writes only Memory.",
    )
    archive.add_argument("--input", required=True)
    archive.add_argument("--commit", action="store_true")
    gaps = sub.add_parser(
        "gap-inventory", help="Saved report/audit-bound gaps; no dispatch permission."
    )
    gaps.add_argument("--report", required=True)
    gaps.add_argument("--evidence", required=True)
    gaps.add_argument("--case", required=True)
    recover = sub.add_parser(
        "recover-accepted", help="Supervise original zero-API aggregation; no case replay."
    )
    recover.add_argument("--input", required=True, help="Original cohort directory")
    recover.add_argument("--case", required=True)
    recover.add_argument("--plan", required=True, help="Explicit accounted aggregation plan 1.3")
    recover.add_argument("--output", required=True, help="Separate recovery supervision directory")
    source = sub.add_parser(
        "recover-source-locked", help="Explicit Runtime-locked Source backlog after a stopped run."
    )
    source.add_argument("--input", required=True, help="Original frozen cohort")
    source.add_argument("--case", required=True)
    source.add_argument("--output", required=True, help="Separate recovery evidence directory")
    source.add_argument("--phase", choices=("prepare", "preflight", "run", "export"), required=True)
    for action in (
        "prepare",
        "preflight",
        "run",
        "execute",
        "export",
        "recover-source",
        "export-recovery",
        "status",
        "progress-html",
    ):
        command = sub.add_parser(action)
        command.add_argument("--input", required=True)
        if action not in {"status", "progress-html"}:
            command.add_argument("--case", action="append", required=True)
    args = parser.parse_args()
    if args.action == "init":
        result = owner.initialize(args.manifest, args.output)
    elif args.action == "lease-preflight":
        result = owner.lease_preflight(args.input)
    elif args.action == "inspect-multimodal":
        from packages.evaluation_preparation.cohort_progress import multimodal

        result = multimodal(args.input)
    elif args.action == "audit-recovery":
        from packages.evaluation_preparation.cohort_recovery import audit_case

        options = {"local_closure": args.local_closure} if args.local_closure else {}
        result = audit_case(args.input, args.report, args.history_root, **options)
    elif args.action == "archive-result":
        from packages.multimodal_research.memory import archive_saved_snapshot

        result = archive_saved_snapshot(args.input, commit=args.commit)
    elif args.action == "gap-inventory":
        from packages.evaluation_preparation.cohort_recovery import gap_inventory

        result = gap_inventory(args.report, args.evidence, args.case)
    elif args.action == "status":
        result = owner.status(args.input)
    elif args.action == "progress-html":
        result = owner.export_status(args.input)
    elif args.action == "recover-source-locked":
        from packages.evaluation_preparation.terminal_source_recovery import run

        result = run(args.input, args.case, args.output, args.phase)
    elif args.action == "recover-accepted":
        result = owner.recover_accepted(args.input, args.case, args.plan, args.output)
    else:
        result = owner.run(args.input, args.action, args.case)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.action == "lease-preflight":
        return 0 if result["execution_lock_ready"] else 2
    if args.action == "archive-result":
        return 2 if result["status"] == "not_archived_source_capacity" else 0
    if args.action in {"recover-accepted", "recover-source-locked"}:
        return 0 if result["exit_code"] == 0 else 2
    if isinstance(result, list):
        return 0 if all(r["exit_code"] == 0 for r in result) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
