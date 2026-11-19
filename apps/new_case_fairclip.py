"""Invoke the existing installed classifier for an approved new visible case."""

import argparse

from packages.evaluation_preparation.fairclip_execution import execute
from packages.grounding.models import canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True, choices=("data_00002.npz", "data_00003.npz"))
    parser.add_argument("--recover-only", action="store_true")
    parser.add_argument("--configuration", default="configs/research/fairclip_single_case.json")
    parser.add_argument(
        "--revision", help="Explicit new repair identity; never resumes a stopped run"
    )
    args = parser.parse_args()
    result = execute(
        args.sample,
        recover_only=args.recover_only,
        configuration=args.configuration,
        revision=args.revision,
    )
    print(canonical(result).decode())
    if args.revision and result.get("technical_status") != "SUCCEEDED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
