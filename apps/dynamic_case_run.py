"""Run one approved development case through the same source-driven campaign."""

import argparse

from packages.evaluation_preparation.dynamic_campaign import Campaign
from packages.grounding.models import canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True, choices=("data_00002.npz", "data_00003.npz"))
    parser.add_argument(
        "--quality-weighted", action="store_true", help="新质量加权版本；独立有界轮次，不重跑旧轮次"
    )
    parser.add_argument("--quality-revision", choices=("v1", "wire-r1"), default="v1")
    args = parser.parse_args()
    if args.quality_revision != "v1" and not args.quality_weighted:
        parser.error("--quality-revision requires --quality-weighted")
    if args.quality_weighted:
        from packages.evaluation_preparation.quality_campaign import QualityCampaign

        result = QualityCampaign(args.sample, revision=args.quality_revision).execute()
    else:
        result = Campaign(args.sample).execute()
    print(canonical(result).decode())
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
