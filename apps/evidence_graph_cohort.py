"""Read-only whole-graph cohort preflight; labels are opt-in and evaluator-only."""

import argparse
import json
from pathlib import Path

from apps.glaucoma_case_storage import read_view
from packages.evidence_scoring.cohort import evaluate, preflight


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--labels", type=Path)
    a = p.parse_args()
    if a.protocol.stat().st_size > 1_048_576:
        p.error("cohort_protocol_bound")
    try:
        protocol = json.loads(a.protocol.read_bytes())
        if a.labels:
            if a.labels.stat().st_size > 1_048_576:
                p.error("cohort_labels_bound")
            result = evaluate(protocol, read_view, json.loads(a.labels.read_bytes()))
        else:
            result = preflight(protocol, read_view)
    except (ValueError, KeyError, TypeError):
        # Validation errors can embed supplied identifiers/labels; keep CLI errors bounded.
        p.exit(2, "cohort_validation_failed; check frozen manifest/scope and input formats\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
