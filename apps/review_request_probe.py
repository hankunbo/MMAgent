"""Public Review diagnostic CLI; preparation/status never read credentials or send HTTP."""

import argparse
import importlib.util
import json

from packages.llm_provider.review_probe import MODELS, ROOT, prepare
from packages.llm_provider.review_probe_execution import execute, status


def secret():
    spec = importlib.util.spec_from_file_location(
        "original_local_key_loader", ROOT / ".local/vectorengine/run.py"
    )
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    return loader.load_secret()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="new public arithmetic matrix; zero API")
    prep.add_argument("--output", required=True)
    prep.add_argument("--model", choices=MODELS, default=MODELS[0])
    prep.add_argument("--repetitions", type=int, choices=(1, 2, 3), default=3)
    prep.add_argument("--provider-config", help="explicit public configs/agents Provider binding")
    run = commands.add_parser("run", help="explicit real public API; never resumes a claimed run")
    run.add_argument("--input", required=True)
    run.add_argument("--execute", action="store_true", required=True)
    read = commands.add_parser("status", help="read saved progress only; zero API")
    read.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(
                args.output,
                args.model,
                repetitions=args.repetitions,
                provider_config=args.provider_config,
            )
            print(
                json.dumps(
                    dict(
                        output=args.output,
                        requests=0,
                        max_requests=result["max_requests"],
                        entries=[(e["name"], e["inventory"]) for e in result["entries"]],
                    )
                )
            )
            return 0
        result = execute(args.input, secret) if args.command == "run" else status(args.input)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if args.command == "status" or result["three_rung_gate_passed"] else 2
    except Exception as error:
        # Library/config/loader errors can contain credentials; only report the type.
        print(
            json.dumps(
                dict(
                    status="stopped",
                    error_type=type(error).__name__,
                    instruction="Inspect saved status; do not replay a claimed run.",
                )
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
