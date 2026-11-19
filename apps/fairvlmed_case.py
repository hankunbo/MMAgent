"""Generic FairVLMed input preparation. Execution uses registered original owners."""

import argparse
import json

from packages.evaluation_preparation.generic_input import inspect_input, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("check", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("--dataset", default="fairvlmed")
        command.add_argument("--root", required=True)
        command.add_argument("--sample", required=True)
        command.add_argument("--split", default="training")
        if name == "prepare":
            command.add_argument("--key", required=True)
            command.add_argument("--fairclip-configuration")
            command.add_argument("--evidence-profile", help="Opt-in: source-complete-view/1.0.0")
            command.add_argument("--upstream-profile", help="Opt-in: evidence-reuse/1.0.0")
            command.add_argument("--report-profile", help="Opt-in: case-report-production/1.0.0")
            command.add_argument(
                "--execution-lock-profile",
                choices=["runtime-case-host-resource/1.0.0"],
                help="New selected-profile inputs only; original Runtime case/resource leases.",
            )
            command.add_argument(
                "--provider-profile",
                help="Opt-in: relay-grounded-v1, relay-grounded-shape-v1, "
                "relay-grounded-shape-proxy-v1, relay-grounded-shape-proxy-alternate-v1, "
                "relay-local-gpt6-shape-v1 or relay-environment-gpt6-shape-v1 "
                "(requires the Runtime execution lock) "
                "(new runs only)",
            )
    for name in ("plan", "run", "recover", "export", "status", "cancel"):
        command = commands.add_parser(name)
        command.add_argument("--key", required=True)
    command = commands.add_parser("review")
    command.add_argument("--key", required=True)
    command.add_argument("--review-key", required=True)
    for name in ("definition-plan", "definition-review"):
        command = commands.add_parser(name)
        command.add_argument("--key", required=True)
        command.add_argument("--review-key", required=True)
        command.add_argument("--definition-policy", required=True)
    command = commands.add_parser("definition-export")
    command.add_argument("--key", required=True)
    command = commands.add_parser("definition-aggregate-plan")
    command.add_argument("--key", required=True)
    command.add_argument("--output", required=True)
    command = commands.add_parser("definition-audit")
    command.add_argument("--input", required=True)
    command.add_argument("--report", required=True)
    command.add_argument("--tree", required=True)
    for name in ("capability-preflight", "capability-run"):
        command = commands.add_parser(name)
        command.add_argument("--key", required=True)
        command.add_argument("--successor-key", required=True)
        command.add_argument(
            "--network-route", help="Explicit public route for the new window only"
        )
    for name in ("capability-continue-preflight", "capability-continue"):
        command = commands.add_parser(name)
        command.add_argument("--key", required=True)
        command.add_argument("--successor-key", required=True)
        command.add_argument("--history-software", required=True)
        command.add_argument("--acknowledge-rejected-request", action="append", default=[])
    command = commands.add_parser("capability-export")
    command.add_argument("--key", required=True)
    command = commands.add_parser("capability-close-unsent-review")
    command.add_argument("--key", required=True)
    command = commands.add_parser("capability-aggregate-plan")
    command.add_argument("--key", required=True)
    command.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    operation = args.pop("operation")
    if operation == "check":
        _, result = inspect_input(**args)
    elif operation == "prepare":
        result = prepare(**args)
    elif operation == "plan":
        from packages.evaluation_preparation.generic_campaign import source_plan
        from packages.evaluation_preparation.generic_input import load, register

        spec = load(args["key"])
        result = source_plan(spec, register(spec))
    elif operation in {"status", "cancel"}:
        from packages.evaluation_preparation.generic_control import cancel, status

        result = (cancel if operation == "cancel" else status)(args["key"])
    elif operation == "capability-preflight":
        from packages.evaluation_preparation.generic_capability_successor import preflight

        result = preflight(
            args["key"],
            args["successor_key"],
            **({"network_route": args["network_route"]} if args["network_route"] else {}),
        )
    elif operation == "capability-run":
        from packages.evaluation_preparation.generic_capability_execution import run

        result = run(
            args["key"],
            args["successor_key"],
            **({"network_route": args["network_route"]} if args["network_route"] else {}),
        )
    elif operation in {"capability-continue-preflight", "capability-continue"}:
        from packages.evaluation_preparation.generic_capability_execution import continue_run

        result = continue_run(
            args["key"],
            args["successor_key"],
            args["history_software"],
            args["acknowledge_rejected_request"],
            check_only=operation == "capability-continue-preflight",
        )
    elif operation == "capability-export":
        from packages.evaluation_preparation.generic_definition_execution import export_saved

        result = export_saved(args["key"])
    elif operation == "capability-close-unsent-review":
        from packages.evaluation_preparation.generic_capability_execution import close_unsent_review

        result = close_unsent_review(args["key"])
    elif operation == "capability-aggregate-plan":
        from packages.evaluation_preparation.generic_capability_aggregation import prepare_plan

        result = prepare_plan(args["key"], args["output"])
    elif operation == "definition-plan":
        from packages.evaluation_preparation.generic_definition_successor import preflight

        result = preflight(args["key"], args["review_key"], args["definition_policy"])
    elif operation in {"definition-review", "definition-export"}:
        from packages.evaluation_preparation.generic_definition_execution import export_saved, run

        result = (
            run(args["key"], args["review_key"], args["definition_policy"])
            if operation == "definition-review"
            else export_saved(args["key"])
        )
    elif operation == "definition-aggregate-plan":
        from packages.evaluation_preparation.generic_definition_aggregation import prepare_plan

        result = prepare_plan(args["key"], args["output"])
    elif operation == "definition-audit":
        from packages.evaluation_preparation.generic_definition_aggregation import audit_wholecase

        result = audit_wholecase(args["input"], args["report"], args["tree"])
    elif operation == "review":
        from packages.evaluation_preparation.generic_review import run_review

        result = run_review(args["key"], args["review_key"])
    else:
        from packages.evaluation_preparation.generic_run import finish_export, run

        result = (
            finish_export(args["key"])
            if operation == "export"
            else run(args["key"], recover=operation == "recover")
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if (
        operation
        in {
            "run",
            "recover",
            "review",
            "definition-review",
            "definition-export",
            "capability-run",
            "capability-continue",
            "capability-export",
        }
        and "result" in result
    ):
        return 0 if result["result"]["complete"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
