"""Explicit bounded direct-LLM baseline for the two approved development images."""

import argparse
import json

from packages.evaluation_preparation.case_execution import SAMPLES
from packages.evaluation_preparation.direct_baseline import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=SAMPLES, required=True)
    args = parser.parse_args()
    result = run(args.sample)
    print(
        json.dumps({k: result[k] for k in ("status", "error", "api_requests", "retries", "usage")})
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
