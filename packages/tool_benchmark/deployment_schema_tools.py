"""06E versioned deployment-only schemas; no shared contracts v1 changes."""

import argparse
import json
from pathlib import Path

from packages.integrations.medgraph.models import ReviewedDeploymentManifest
from packages.integrations.medgraph.research import ResearchReview
from packages.tool_admission.method_metadata import MethodMetadata
from packages.tool_admission.models import DeploymentProfile, JSONOutputRule
from packages.tool_execution.geometry_qc import DiscPNGPolicy, GeometryJSONPolicy


def schemas():
    return {
        f"{model.__name__}.schema.json": json.dumps(
            model.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
        for model in (
            ResearchReview,
            ReviewedDeploymentManifest,
            DeploymentProfile,
            JSONOutputRule,
            DiscPNGPolicy,
            GeometryJSONPolicy,
            MethodMetadata,
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "schemas/deployment-v1"
    for name, content in schemas().items():
        if args.write:
            root.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(content)
        else:
            assert (root / name).read_text() == content, name
    print(
        f"{len(schemas())} benchmark deployment schemas "
        + ("written" if args.write else "verified")
    )


if __name__ == "__main__":
    main()
