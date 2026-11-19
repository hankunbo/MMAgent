"""Bounded new-case graph prepare/execute; no old patient observation cloning."""

import argparse
import json

from packages.evaluation_preparation.case_execution import SAMPLES
from packages.evaluation_preparation.case_live import execute, prepare
from packages.grounding.models import canonical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("--sample", choices=SAMPLES, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        _, _, _, _, state, output = prepare(args.sample)
        print(
            json.dumps(
                dict(
                    status="prepared_not_executed",
                    output=str(output),
                    scope=state.material.scope.model_dump(mode="json"),
                )
            )
        )
        return 0
    result = execute(args.sample)
    print(
        canonical(
            dict(
                complete=result["complete"],
                run_id=result["run_id"],
                requests=result["request_summary"],
            )
        ).decode()
    )
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
