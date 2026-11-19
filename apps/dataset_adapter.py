"""Standalone patient material preparation. Never invokes agents, models or databases."""

import argparse
import json
import sys

from packages.dataset_adapters.patient_decoders import ADAPTERS, candidates
from packages.dataset_adapters.patient_models import PatientPacket
from packages.dataset_adapters.patient_packet import prepare_patient
from packages.dataset_adapters.patient_storage import persist, summary, verify
from packages.dataset_adapters.reading import AdapterError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "prepare"):
        p = commands.add_parser(name)
        p.add_argument("--adapter", required=True, choices=sorted(ADAPTERS))
        p.add_argument("--root", required=True)
        p.add_argument("--selection")
        if name == "prepare":
            p.add_argument("--subject", required=True)
            p.add_argument("--output", required=True)
        else:
            p.add_argument("--limit", type=int, default=3)
    p = commands.add_parser("verify")
    p.add_argument("--input", required=True)
    p.add_argument(
        "--root", help="Additionally recheck original files; otherwise verify saved packet"
    )
    commands.add_parser("schema")
    args = parser.parse_args(argv)
    try:
        if args.command == "schema":
            result = PatientPacket.model_json_schema()
        elif args.command == "list":
            result = dict(
                adapter=args.adapter,
                subjects=candidates(
                    args.adapter, args.root, limit=args.limit, selection=args.selection
                ),
            )
        elif args.command == "prepare":
            value = prepare_patient(args.adapter, args.root, args.subject, selection=args.selection)
            out = persist(value, args.output, source_root=args.root)
            result = dict(output=str(out), **summary(value.packet))
        else:
            result = summary(verify(args.input, source_root=args.root))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (AdapterError, ValueError, OSError, KeyError) as error:
        # Show a bounded diagnostic, never raw table contents or private settings.
        print(
            json.dumps(
                dict(error=type(error).__name__, detail=str(error)[:400]), ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
