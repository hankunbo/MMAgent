"""Deterministic schemas for 07B application inputs; no frozen contracts change."""

import argparse
import json
from pathlib import Path

from .demands import DemandResolution
from .models import CaseInput, PipelineConfig, PipelinePolicy, RequestState
from .tool_evidence import ToolEvidenceCheck


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "schemas"
    for model in (
        CaseInput,
        PipelineConfig,
        PipelinePolicy,
        RequestState,
        DemandResolution,
        ToolEvidenceCheck,
    ):
        path = root / (model.__name__ + ".schema.json")
        data = (
            json.dumps(model.model_json_schema(), sort_keys=True, ensure_ascii=False, indent=2)
            + "\n"
        )
        if args.write:
            root.mkdir(exist_ok=True)
            path.write_text(data)
        elif not path.exists() or path.read_text() != data:
            raise SystemExit("07B schema drift: " + path.name)


if __name__ == "__main__":
    main()
