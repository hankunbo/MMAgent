"""Two-step, no-inference development benchmark CLI."""

import argparse
import json
from pathlib import Path

from packages.tool_benchmark.development import evaluate, prepare
from packages.tool_benchmark.development_models import DevelopmentProtocol


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        protocol = prepare()
        write(args.output, protocol.model_dump(mode="json"))
        print(
            json.dumps(
                dict(
                    protocol_checksum=protocol.checksum,
                    samples=len(protocol.samples),
                    truth_decoded=False,
                )
            )
        )
    else:
        if args.protocol is None:
            parser.error("evaluate requires --protocol")
        protocol = DevelopmentProtocol.model_validate_json(args.protocol.read_text())
        args.output.mkdir(mode=0o700, exist_ok=False)
        rows, report = evaluate(protocol)
        write(args.output / "restricted-samples.json", rows)
        (args.output / "restricted-samples.json").chmod(0o600)
        write(args.output / "summary.json", report)
        print(
            json.dumps(
                dict(selected=report["selected"], computed=report["computed"], new_model_calls=0)
            )
        )
        if report["technical_failures"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
