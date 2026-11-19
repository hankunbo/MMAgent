"""Re-render saved old/new evidence without API, database writes or rescoring."""

import argparse
import json

from packages.evaluation_preparation.evidence_view import convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--layout", choices=("tree-v3", "tree-v2", "tree-v1", "cards-v1"), default="tree-v3"
    )
    media = parser.add_mutually_exclusive_group()
    media.add_argument(
        "--run-key", help="Read and bind this run's saved images; no model/API calls"
    )
    media.add_argument("--media-path", help="Reuse a report-bound saved image package offline")
    args = parser.parse_args()
    print(
        json.dumps(
            convert(
                args.input,
                args.output,
                layout=args.layout,
                run_key=args.run_key,
                media_path=args.media_path,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
