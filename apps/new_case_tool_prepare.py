"""Two new SLO cases; original registration and bounded SegMAN native component."""

import argparse
import json

from packages.evaluation_preparation.case_execution import SAMPLES, register, segman


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("register", "segman", "segman-fixed", "segman-isolated", "segman-current"),
    )
    parser.add_argument("--sample", choices=SAMPLES, required=True)
    args = parser.parse_args()
    if args.action == "register":
        directory, binding, _, _ = register(args.sample)
        result = dict(directory=str(directory), case_code=binding["case_code"], registered=True)
    else:
        result = segman(
            args.sample,
            directory_adapter=args.action in {"segman-fixed", "segman-isolated", "segman-current"},
            isolated=args.action in {"segman-isolated", "segman-current"},
            system_path=args.action == "segman-current",
        )
        result = dict(
            directory=result["directory"],
            case_code=result["binding"]["case_code"],
            exit_code=result["result"]["exit_code"],
            llm_requests=0,
            scope="native_tool_result_not_yet_graph_acceptance",
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
