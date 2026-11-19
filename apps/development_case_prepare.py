"""Prepare new development cases; never sends API requests or connects to a database."""

import argparse
import json
import os
from pathlib import Path

from packages.evaluation_preparation.new_cases import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        os.environ["MMAGENT_FAIRVLMED_ROOT"], os.environ["MMAGENT_PAPILA_ROOT"], args.output
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
