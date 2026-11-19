"""Read confirmed same-case observations for a new, text-only Review attempt.

An unknown Vision remains unknown. This port never dispatches or resumes it.
"""

import json
from pathlib import Path

from packages.evidence_details.acceptance_models import DomainState, read_receipt, read_records
from packages.grounding.models import digest, validate


class ConfirmedObservations:
    def __init__(self, campaign):
        self.campaign = campaign
        base = campaign.base
        if not base.known_only or campaign.targets or not base.previous:
            raise ValueError("confirmed_observations_text_only")
        previous = Path(base.previous)
        spec = json.loads((previous / "entry.json").read_bytes())
        self.result = json.loads((previous / "result.json").read_bytes())
        if spec.get("confirmed_observations_only") or spec.get("known_only"):
            raise ValueError("confirmed_observations_single_continuation")
        campaign.previous_attempts()  # Original scope/control/unknown/budget gates.
        self.config = validate(type(campaign.config), spec["provider"])
        self.values, self.origins = self.read()
        if not self.values:
            raise ValueError("confirmed_observations_empty")
        self.binding = digest((self.values, self.origins))

    def read(self):
        from .campaign import Phase

        c, b = self.campaign, self.campaign.base
        values, origins = {}, []
        for result in self.result["phases"]:
            target = result["target"]
            if target is None:
                raise ValueError("confirmed_observations_vision_parent_required")
            if not result["accepted"]:
                continue
            if target in values:
                raise ValueError("confirmed_observations_duplicate_target")
            docs = c.documents(result["run_id"])
            if docs.read("dynamic-phase-result") != (result,):
                raise ValueError("confirmed_observations_result_binding")
            parent = validate(DomainState, docs.read("dynamic-phase-input")[0])
            b.contexts[result["run_id"]] = c.contexts.read_research_run(result["run_id"])
            phase = Phase(b, target=target)
            phase.project(run_id=result["run_id"], config=self.config)
            phase.verify(parent)  # Current input, medical sources and original definition.
            saved = docs.read("accepted:" + result["acceptance_key"])
            records, receipt = read_records(saved[0]), read_receipt(saved[3])
            if (
                receipt.status != "accepted"
                or receipt.records_checksum != digest(records)
                or records.scope != parent.material.scope
                or len(records.observations) != 1
            ):
                raise ValueError("confirmed_observations_receipt")
            observation = records.observations[0]
            image, _, _ = b.view(target)
            if (
                observation.image != image
                or observation.reference.code != parent.material.observations[0].reference.code
            ):
                raise ValueError("confirmed_observations_image_binding")
            values[target] = observation
            origins.append(
                dict(
                    target=target,
                    run_id=result["run_id"],
                    acceptance_key=result["acceptance_key"],
                    receipt_checksum=digest(receipt),
                    records_checksum=digest(records),
                )
            )
        return values, origins

    def check(self):
        if digest(self.read()) != self.binding:
            raise ValueError("confirmed_observations_changed")
        return self.values

    def report(self):
        return dict(
            schema_version="confirmed-visual-review-input/1.0.0",
            observations_checksum=digest(self.values),
            origins=self.origins,
            new_vision_requests=0,
            unknown_vision_replayed=False,
        )
