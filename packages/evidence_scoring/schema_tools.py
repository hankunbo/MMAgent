"""Generate or verify the research-score JSON Schema."""

import argparse
import json
from pathlib import Path

from packages.evidence_scoring.cohort import Protocol
from packages.evidence_scoring.continuous_models import ContinuousScores
from packages.evidence_scoring.current_models import CurrentScores
from packages.evidence_scoring.interpretation_models import InterpretationScores
from packages.evidence_scoring.models import Scores
from packages.evidence_scoring.participation_models import ParticipationScores


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, model in [
        ("schema.json", Scores),
        ("schema-v2.json", ContinuousScores),
        ("schema-v3.json", ParticipationScores),
        ("schema-v4.json", InterpretationScores),
        ("schema-v5.json", CurrentScores),
        ("cohort-protocol-schema-v1.json", Protocol),
    ]:
        path = Path(__file__).with_name(name)
        expected = (
            json.dumps(model.model_json_schema(), ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        )
        if args.write:
            if not path.exists() or path.read_text() != expected:
                path.write_text(expected)
        elif path.read_text() != expected:
            raise SystemExit("research score schema drift: " + name)
    print("research score schema OK")


if __name__ == "__main__":
    main()
