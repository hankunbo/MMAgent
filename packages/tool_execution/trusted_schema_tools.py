"""Independent versioned GPU research schemas. Legacy 06D schema files stay exact."""

import argparse
import json
from pathlib import Path

from packages.tool_workflow.models import TrustedResearchCompilerConfig

from .trusted_models import (
    TrustedCapabilities,
    TrustedLocalPolicy,
    TrustedTechnicalOutcome,
    TrustedToolReview,
)


def schemas():
    return {
        f"{model.__name__}.schema.json": json.dumps(
            model.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
        for model in (
            TrustedCapabilities,
            TrustedLocalPolicy,
            TrustedTechnicalOutcome,
            TrustedToolReview,
            TrustedResearchCompilerConfig,
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "schemas/trusted-local-v1"
    for name, content in schemas().items():
        if args.write:
            root.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(content)
        else:
            assert (root / name).read_text() == content, name
    print("5 trusted local research schemas " + ("written" if args.write else "verified"))


if __name__ == "__main__":
    main()
