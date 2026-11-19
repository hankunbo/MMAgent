"""Public, zero-network upstream preparation/probe commands. No execution grants."""

import argparse
import json

from apps.vectorengine_single_case import write_once


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    papila = commands.add_parser("papila-components")
    papila.add_argument("--root", required=True)
    papila.add_argument("--sample", required=True)
    papila.add_argument("--output", required=True)
    evaluation = commands.add_parser("evaluate-saved")
    evaluation.add_argument("--input", required=True)
    evaluation.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.operation == "papila-components":
        from packages.dataset_adapters.components import papila_components
        from packages.dataset_adapters.papila import PapilaClinicalPreparation

        value = papila_components(PapilaClinicalPreparation(args.root), args.sample).model_dump(
            mode="json"
        )
    else:
        from packages.evaluation_preparation.evidence_view import load_report
        from packages.evidence_reuse.evaluation import evaluate_saved

        report, _ = load_report(args.input)
        value = evaluate_saved(report)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_once(path, value)
    print(
        json.dumps(
            dict(
                output=str(path),
                schema_version=value["schema_version"],
                api_requests=0,
                model_starts=0,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
