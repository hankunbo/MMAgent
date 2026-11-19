"""Finite review-only successor; acquired observations remain original owner facts.

Never resumes a stopped/expired request. Carries every previous reservation into
the same 12-request increment and 40-request cumulative ceiling.
"""

import json

from .comprehensive_campaign import ComprehensiveCampaign


class ComprehensiveReview(ComprehensiveCampaign):
    version = "13-full-review-r1"
    export_suffix = "_全清单与部分观察闭环版"
    max_vision_requests = 0

    def __init__(self, sample):
        # The acquisition inventory remains the same, but is never executed here.
        self.max_vision_requests = 7
        super().__init__(sample)
        self.acquisition = ComprehensiveCampaign(sample)
        self.max_vision_requests = 0
        used = self.acquisition.usage()
        self.max_new_requests = max(0, 12 - used)
        if not self.max_new_requests:
            raise ValueError("full_increment_budget_exhausted")

    def before_review(self):
        pass

    def paths(self):
        return [self.output / "review"]

    def phase_path(self, target):
        return self.acquisition.phase_path(target) if target else self.output / "review"

    def prior(self):
        previous = self.acquisition.prior()
        for path in self.acquisition.paths():
            request_path = path / "request.json"
            if request_path.exists():
                run = json.loads(request_path.read_bytes())["lock"]["run_id"]
                current = self.sessions.read(run)
                if current is None or current.phase != "stopped" or current.pending:
                    raise ValueError("full_predecessor_requires_control_confirmation")
        used = self.acquisition.usage()
        return dict(
            previous=previous,
            acquisition_requests=used,
            reservations=previous["reservations"] + used,
        )

    def accepted_observations(self):
        return self.acquisition.accepted_observations()

    def freeze_observations(self):
        return dict(
            acquisition=self.acquisition.freeze_observations(),
            observations=self.accepted_observations(),
        )


if __name__ == "__main__":
    import sys

    campaign = ComprehensiveReview(sys.argv[1])
    print(json.dumps(campaign.execute(), ensure_ascii=False))
