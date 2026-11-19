"""Generate/check the additive observation/interpretation JSON Schema."""

import argparse
import json
from pathlib import Path

from packages.evidence_details.records_models import EvidenceRecords


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = p.parse_args()
    path = Path(__file__).with_name("observation-interpretation-schema-v1.json")
    content = (
        json.dumps(
            EvidenceRecords.model_json_schema(), ensure_ascii=False, sort_keys=True, indent=2
        )
        + "\n"
    )
    if args.write:
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
    elif not path.exists() or path.read_text() != content:
        raise SystemExit("observation interpretation schema drift")
    print("observation interpretation schema OK")


if __name__ == "__main__":
    main()
