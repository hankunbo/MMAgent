"""Deterministic JSON Schema for judgment-details consumers."""

import argparse
import json
from pathlib import Path

from packages.evidence_details.models import Ledger

PATH = Path(__file__).with_name("schema.json")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = (
        json.dumps(Ledger.model_json_schema(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    if args.write:
        PATH.write_text(expected)
    elif PATH.read_text() != expected:
        raise SystemExit("judgment detail schema drift")
    print("judgment detail schema OK")


if __name__ == "__main__":
    main()
