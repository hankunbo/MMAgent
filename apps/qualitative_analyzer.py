"""Standalone data understanding; no Codex, disease proposition, DB or full agent run."""

import argparse
import json
from pathlib import Path

from packages.dataset_adapters.patient_storage import verify as verify_packet
from packages.qualitative_analysis import runner
from packages.qualitative_analysis.storage import read, seal, write_once


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--packet", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/research/qualitative_analysis_v2.json")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--archive-description", action="store_true")
    g.add_argument(
        "--scope", help="Explicit component/background IDs; no implicit future selection"
    )
    for name in ("run", "resume", "status", "verify", "export", "cancel"):
        p = commands.add_parser(name)
        p.add_argument("--input", required=True)
        if name in {"run", "resume"}:
            p.add_argument("--live", action="store_true")
            p.add_argument(
                "--local-key",
                action="store_true",
                help="Load existing private launcher key in process; never print or copy it",
            )
    commands.add_parser("schema")
    p = commands.add_parser("integrate-saved")
    p.add_argument("--input", required=True, help="Exact saved v1 result JSON; no new API")
    p.add_argument("--output", required=True)
    p = commands.add_parser("compare")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("combine-confirmed")
    p.add_argument("--current", required=True)
    p.add_argument("--previous", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("supplement-confirmed")
    p.add_argument("--current", required=True)
    p.add_argument("--supplement", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "supplement-confirmed":
            from packages.qualitative_analysis.supplement import supplement_confirmed

            result = supplement_confirmed(args.current, args.supplement, args.output)
        elif args.command == "combine-confirmed":
            from packages.qualitative_analysis.comparison import combine_confirmed

            result = combine_confirmed(args.current, args.previous, args.output)
        elif args.command == "integrate-saved":
            from packages.qualitative_analysis.comparison import integrate_saved

            result = integrate_saved(args.input, args.output)
        elif args.command == "compare":
            from packages.qualitative_analysis.comparison import compare

            result = compare(args.before, args.after, args.output)
        elif args.command == "plan":
            if args.archive_description:
                packet = verify_packet(args.packet)
                scope = dict(
                    mode="archive_description",
                    component_ids=[c.id for c in packet.components],
                    background_ids=[b.id for b in packet.background],
                )
            else:
                scope = read(args.scope)
            result = runner.plan(args.packet, args.output, args.config, scope)
        elif args.command in {"run", "resume"}:
            result = runner.run(args.input, live=args.live, local_key=args.local_key)
        elif args.command == "schema":
            from packages.qualitative_analysis.models import VisualResultV2

            result = VisualResultV2.model_json_schema()
        elif args.command == "cancel":
            runner.load(args.input, current=False)
            path = Path(args.input).absolute() / "cancel.json"
            write_once(path, seal(dict(reason="user_cancelled_no_new_dispatch")))
            result = runner.status(args.input)
        else:
            result = getattr(runner, args.command)(args.input)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        if (
            args.command in {"run", "resume"}
            and args.live
            and not result["all_planned_jobs_completed"]
        ):
            return 2
        return 0
    except Exception as error:
        # Pydantic/HTTP errors may contain patient input; never echo raw exceptions.
        print(
            json.dumps(
                dict(
                    error=type(error).__name__,
                    status="qualitative_command_failed_inspect_saved_records",
                )
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
