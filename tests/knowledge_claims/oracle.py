"""Human annotated examples. No production regex or label inference is imported."""

import copy
import json
from pathlib import Path


def cases(name="policy-red.json"):
    return json.loads((Path(__file__).parent / "fixtures" / name).read_text())


def annotated_output(case, offset=0):
    text = case["text"]
    row = {
        "claim_type": "evidence_direction",
        "statement": text,
        "target_proposition": text,
        "quote": text,
        "span": {"start": offset, "end": offset + len(text)},
        "polarity": case["polarity"],
        "subject": case.get("subject", "OCT"),
        "confidence": 0.8,
        **{
            key: []
            for key in (
                "uncertainty",
                "recommendation_strength",
                "population",
                "disease_stage",
                "condition",
                "intervention",
                "exam",
                "modality",
                "outcome",
                "time",
                "laterality",
                "device",
                "applicability",
            )
        },
        "numbers": copy.deepcopy(case["numbers"]),
    }
    for field, values in case.get("fields", {}).items():
        row[field] = values
    for number in row["numbers"]:
        number["span"]["start"] += offset
        number["span"]["end"] += offset
    return json.dumps({"candidates": [row]}, ensure_ascii=False).encode()
