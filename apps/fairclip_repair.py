"""Explicit deployment/confirmed-model review commands; no automatic extra rounds."""

import argparse

from packages.grounding.models import canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    install = sub.add_parser("prepare")
    install.add_argument("--output", required=True)
    install.add_argument("--gpu", required=True)
    review = sub.add_parser("review")
    review.add_argument("--sample", required=True, choices=("data_00002.npz", "data_00003.npz"))
    args = parser.parse_args()
    if args.command == "prepare":
        from packages.evaluation_preparation.fairclip_startup_deployment import install

        print(install(args.output, selected_gpu=args.gpu))
    else:
        from packages.evaluation_preparation.model_repair_campaign import ModelRepairCampaign

        result = ModelRepairCampaign(args.sample).execute()
        print(canonical(result).decode())
        if not result["complete"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
