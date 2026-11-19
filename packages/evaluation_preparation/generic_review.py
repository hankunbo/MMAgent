"""One explicit bounded review-only successor; no model/vision re-execution."""

import json
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.grounding.models import digest
from packages.llm_provider.config import read_provider_config

from . import generic_pending_review as pending_review
from .dynamic_campaign import Campaign
from .generic_campaign import GenericCampaign
from .generic_input import (
    PROFILE_INPUTS,
    PROVIDER,
    SHAPE_PROFILE,
    SHAPE_PROVIDER,
    directory,
    load,
    provider_config,
    register,
)


def review_configuration(origin, profile):
    """Explicit Review-only shape revision; no acquisition or historical reinterpretation."""
    from packages.evidence_details.live_permission import validate_live_config

    if profile in pending_review.PROFILES:
        return pending_review.configuration(origin, profile)
    if profile != SHAPE_PROFILE:
        raise ValueError("generic_review_profile_unknown")
    expected = origin.model_dump(mode="json")
    role = next(r for r in expected["roles"] if r["role"] == "review")
    if (
        origin.schema_version != "agent-provider-config/7.0.0"
        or origin.transport.max_retries != 0
        or role["prompt_version"] != "review-batch-1.8.0"
    ):
        raise ValueError("generic_review_profile_origin")
    role.update(
        prompt_version="review-batch-1.9.0",
        prompt_checksum=digest(Path("configs/prompts/agents/review-batch-1.9.0.txt").read_text()),
    )
    # Existing confirmed_http_once recovery refuses UNKNOWN; all retries count
    # toward the unchanged cumulative budget and max_extra_requests=3.
    expected["transport"]["max_retries"] = 1
    selected = validate_live_config(
        read_provider_config(json.loads(Path(SHAPE_PROVIDER).read_bytes()))
    )
    if selected.model_dump(mode="json") != expected:
        raise ValueError("generic_review_profile_delta")
    return selected


class GenericReview(GenericCampaign):
    def __init__(
        self,
        spec,
        registered,
        key,
        *,
        review_profile=None,
        acknowledge_truncated_request=None,
        acknowledge_unknown_request=None,
    ):
        super().__init__(spec, registered)
        self.acquisition = GenericCampaign(spec, registered)
        if spec["schema_version"] not in PROFILE_INPUTS:
            self.config = read_provider_config(json.loads(Path(PROVIDER).read_bytes()))
            self.provider_path = PROVIDER
        if review_profile is not None:
            expected_version = (
                "fairvlmed-run-input/1.6.0"
                if review_profile == pending_review.PROFILE
                else "fairvlmed-run-input/1.5.0"
            )
            allowed_versions = (
                {"fairvlmed-run-input/1.7.0", "fairvlmed-run-input/1.8.0"}
                if review_profile
                in {pending_review.CURRENT_PROFILE, pending_review.UNSTARTED_PROFILE}
                else {expected_version}
            )
            if spec["schema_version"] not in allowed_versions:
                raise ValueError("generic_review_profile_origin")
            self.config = review_configuration(self.config, review_profile)
            self.provider_path = (
                pending_review.PROVIDER
                if review_profile == pending_review.PROFILE
                else self.provider_path
                if review_profile
                in {pending_review.CURRENT_PROFILE, pending_review.UNSTARTED_PROFILE}
                else SHAPE_PROVIDER
            )
        self.pending_review_proof = None
        self.review_profile = review_profile
        self.acknowledge_unknown_request = acknowledge_unknown_request
        self.acknowledge_truncated_request = acknowledge_truncated_request
        validate_acknowledgment(
            review_profile, acknowledge_truncated_request, acknowledge_unknown_request
        )
        if acknowledge_truncated_request and review_profile != pending_review.PROFILE:
            raise ValueError("generic_review_truncation_requires_pending_profile")
        self.review_key = key
        self.structured_completion = True
        from .generic_aliases import GenericPhase

        self.phase_class = GenericPhase
        if review_profile in pending_review.PROFILES:
            self.pending_review_proof = pending_review.current_selection(
                self.acquisition, acknowledge_truncated_request, **self.selection_options()
            )
            self.phase_class = pending_review.phase_type(lambda: self.pending_review_proof)
        self.version = "generic-review-v1:" + key
        self.output = directory(key) / "evidence"
        self.output.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.export_suffix = "_通用评阅续接_" + key
        self.max_vision_requests = 0
        self.max_new_requests = min(10, 40 - self.prior()["reservations"])
        if self.max_new_requests <= 0:
            raise ValueError("generic_review_budget_exhausted")
        if (
            self.pending_review_proof
            and pending_review.minimum_requests(
                self.pending_review_proof, self.config, self.review_profile
            )
            > self.max_new_requests
        ):
            raise ValueError("generic_pending_selection_exceeds_request_budget")
        self.duration_seconds = 2700

    def selection_options(self):
        if self.review_profile == pending_review.UNSTARTED_PROFILE:
            return dict(profile=self.review_profile)
        if self.review_profile == pending_review.CURRENT_PROFILE:
            return dict(
                profile=self.review_profile,
                acknowledge_unknown_request=self.acknowledge_unknown_request,
            )
        return {}

    def before_review(self):
        if (
            getattr(self, "pending_review_proof", None) is not None
            and pending_review.current_selection(
                self.acquisition, self.acknowledge_truncated_request, **self.selection_options()
            )
            != self.pending_review_proof
        ):
            raise ValueError("generic_pending_original_changed_before_dispatch")

    def export_result(self, result):
        if self.pending_review_proof is not None:
            return pending_review.export_increment(self, result)
        return super().export_result(result)

    def archive_completed(self, result):
        if self.pending_review_proof is not None:
            # A selected increment is not a complete-case skill or Memory success.
            return result
        return super().archive_completed(result)

    def freeze_observations(self):
        frozen = dict(
            acquisition=self.acquisition.freeze_observations(),
            observations=digest(self.accepted_observations()),
        )
        if self.pending_review_proof is not None:
            frozen.update(
                pending_review=self.pending_review_proof,
                pending_owner_implementation=pending_review.implementation(),
            )
        return frozen

    def paths(self):
        return [self.output / "review"]

    def phase_path(self, target):
        return self.acquisition.phase_path(target) if target else self.output / "review"

    def accepted_observations(self):
        return self.acquisition.accepted_observations()

    def prior(self):
        old = json.loads((self.acquisition.output / "result.json").read_bytes())
        final = old["phases"][-1]
        if self.documents(final["run_id"]).read("dynamic-case-result") != (old,):
            raise ValueError("generic_review_requires_bound_predecessor")
        saved = self.documents(final["run_id"]).read("dynamic-case-report")
        if not saved or saved[0]["result"] != old:
            raise ValueError("generic_review_predecessor_report_missing")
        completion = (saved[0].get("scores") or {}).get("review_completion", {})
        if old["complete"] and not completion.get("pending_ids"):
            raise ValueError("generic_review_already_complete")
        for path in self.acquisition.paths():
            request = path / "request.json"
            if request.exists():
                run = json.loads(request.read_bytes())["lock"]["run_id"]
                current = self.sessions.read(run)
                if current is None or current.phase != "stopped" or current.pending is not None:
                    raise ValueError("generic_review_predecessor_not_closed")
        return dict(
            reservations=self.acquisition.usage(),
            predecessor_result=digest(old),
            predecessor_report=digest(saved[0]),
        )

    def guard(self, *, dispatch=False):
        from packages.llm_provider.provider import ProviderError

        from .generic_control import guard

        if (directory(self.spec["key"]) / "cancel-request.json").exists():
            raise ProviderError("cancelled")
        guard(self.review_key)
        return Campaign.guard(self, dispatch=dispatch)

    def cancellation_signal(self):
        from .generic_control import Cancellation

        parent = self.spec["key"]

        class ReviewCancellation(Cancellation):
            def is_set(inner):
                return super().is_set() or (directory(parent) / "cancel-request.json").exists()

        return ReviewCancellation(self.review_key)

    def request_key(self, target):
        if target is not None:
            raise ValueError("generic_review_no_acquisition")
        return "fv:review:" + digest(dict(input=self.spec, review=self.review_key))[7:]

    def archive_key(self, target):
        return (
            "fv:memory:" + digest(dict(input=self.spec, review=self.review_key, target=target))[7:]
        )


def validate_acknowledgment(profile, truncated, unknown):
    if profile == pending_review.UNSTARTED_PROFILE:
        if truncated is not None or unknown is not None:
            raise ValueError("generic_unstarted_no_reassessment_authorization")
        return
    if profile == pending_review.CURRENT_PROFILE:
        if truncated is not None or not unknown:
            raise ValueError("generic_pending_exact_unknown_ack_required")
    elif unknown is not None:
        raise ValueError("generic_pending_unknown_requires_current_profile")


def run_review(
    key,
    review_key,
    *,
    review_profile=None,
    acknowledge_truncated_request=None,
    acknowledge_unknown_request=None,
    method_origin=None,
    review_method_profile=None,
    prior_review_key=None,
    network_route=None,
    accepted_review_key=None,
    acknowledge_reassessment_request=(),
    continue_review_key=None,
    preflight=False,
):
    if continue_review_key is not None:
        from .generic_review_continuation import run

        if any(
            (
                accepted_review_key,
                prior_review_key,
                network_route,
                review_profile,
                method_origin,
                review_method_profile,
                acknowledge_truncated_request,
                acknowledge_unknown_request,
            )
        ):
            raise ValueError("generic_continuation_explicit_scope")
        return run(
            key,
            review_key,
            continue_review_key,
            acknowledge_reassessment_request,
            preflight=preflight,
        )
    if accepted_review_key is not None or acknowledge_reassessment_request:
        from .generic_reassessment import run

        if (
            not accepted_review_key
            or not acknowledge_reassessment_request
            or prior_review_key
            or network_route
            or review_profile
            or method_origin
            or review_method_profile
            or acknowledge_truncated_request
            or acknowledge_unknown_request
        ):
            raise ValueError("generic_reassessment_explicit_scope")
        return run(
            key,
            review_key,
            accepted_review_key,
            acknowledge_reassessment_request,
            preflight=preflight,
        )
    if prior_review_key is not None or network_route is not None or preflight:
        from .generic_review_followup import run

        if (
            not prior_review_key
            or not network_route
            or review_profile
            or method_origin
            or review_method_profile
            or acknowledge_truncated_request
            or acknowledge_unknown_request
        ):
            raise ValueError("generic_followup_explicit_scope")
        return run(key, review_key, prior_review_key, network_route, preflight=preflight)
    validate_acknowledgment(
        review_profile, acknowledge_truncated_request, acknowledge_unknown_request
    )
    if method_origin is not None or review_method_profile is not None:
        from .generic_method_review import run

        if (
            not method_origin
            or not review_method_profile
            or review_profile
            or acknowledge_truncated_request
            or acknowledge_unknown_request
        ):
            raise ValueError("generic_method_explicit_scope")
        return run(
            key,
            review_key,
            method_origin=method_origin,
            review_method_profile=review_method_profile,
        )
    import fcntl

    from .generic_control import guard, start
    from .generic_run import export, persist_export

    if key == review_key:
        raise ValueError("generic_review_requires_distinct_key")
    if acknowledge_truncated_request and review_profile != pending_review.PROFILE:
        raise ValueError("generic_review_truncation_requires_pending_profile")
    spec = load(key)
    selected = (
        review_configuration(provider_config(spec), review_profile)
        if review_profile is not None
        else None
    )
    parent = directory(key)
    out = directory(review_key)
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (parent / "operation.lock").open("a") as parent_lock:
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = dict(
            schema_version="generic-review-selection/1.0.0",
            parent_key=key,
            review_key=review_key,
            input_checksum=digest(spec),
        )
        if selected is not None:
            selection.update(
                schema_version="generic-review-selection/1.1.0",
                review_profile=review_profile,
                provider_checksum=digest(selected),
                new_vision_requests=0,
                max_new_requests=10,
                max_cumulative_requests=40,
                max_extra_requests=selected.max_extra_requests,
                response_contract="research-review-batch/1.0.0",
            )
        if review_profile == pending_review.PROFILE:
            selection.update(
                schema_version="generic-review-selection/1.2.0",
                acknowledge_truncated_request=acknowledge_truncated_request,
                selection_kind="pending_only_requires_later_whole_case_aggregation",
            )
        if review_profile == pending_review.CURRENT_PROFILE:
            selection.update(
                schema_version="generic-review-selection/1.3.0",
                acknowledge_unknown_request=acknowledge_unknown_request,
                selection_kind="explicit_new_reassessment_old_unknown_unchanged",
                provider_derivation="bound-provider-batch-size-only/1.0.0",
                origin_provider_checksum=digest(provider_config(spec)),
            )
        if review_profile == pending_review.UNSTARTED_PROFILE:
            selection.update(
                schema_version="generic-review-selection/1.4.0",
                selection_kind="unstarted_only_original_failed_review_unchanged",
                provider_derivation="bound-provider-unchanged/1.0.0",
                origin_provider_checksum=digest(provider_config(spec)),
            )
        prepared = None
        if review_profile in pending_review.PROFILES:
            if (parent / "cancel-request.json").exists():
                raise ValueError("generic_review_parent_cancelled")
            # Rebuild the current original owner BEFORE claiming the single
            # successor or starting its time window. A bad acknowledgment or
            # changed input must not consume that one permitted successor.
            registered = register(spec)
            prepared = GenericReview(
                spec,
                registered,
                review_key,
                review_profile=review_profile,
                acknowledge_truncated_request=acknowledge_truncated_request,
                acknowledge_unknown_request=acknowledge_unknown_request,
            )
            selection.update(
                selected_ids=prepared.pending_review_proof["selected_ids"],
                eligibility_checksum=digest(prepared.pending_review_proof),
            )
        # At most one successor in this version. A different key cannot create an
        # unbounded retry chain or reset the same case's cumulative budget.
        write_once(parent / "review-successor.json", selection)
        write_once(out / "input.json", dict(selection, key=review_key))
        if (out / "export-reference.json").exists():
            return dict(status="already_recorded", exports=export(review_key))
        if (parent / "cancel-request.json").exists():
            raise ValueError("generic_review_parent_cancelled")
        start(dict(selection, key=review_key))
        guard(review_key)
        if prepared is None:
            registered = register(spec)
        write_once(out / "binding.json", json.loads((parent / "binding.json").read_bytes()))
        campaign = prepared or GenericReview(
            spec,
            registered,
            review_key,
            review_profile=review_profile,
            acknowledge_truncated_request=acknowledge_truncated_request,
            acknowledge_unknown_request=acknowledge_unknown_request,
        )
        result = campaign.execute()
        return persist_export(review_key, campaign, result)


def main(argv=None):
    """Explicit profile CLI over the same owner, preserving frozen legacy app bytes."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True)
    parser.add_argument("--review-key", required=True)
    parser.add_argument("--review-profile", choices=(SHAPE_PROFILE, *pending_review.PROFILES))
    parser.add_argument(
        "--method-origin",
        default=argparse.SUPPRESS,
        help="Exact accepted aggregation for a new text method",
    )
    parser.add_argument(
        "--review-method-profile", default=argparse.SUPPRESS, choices=("relay-retained-generic-v1",)
    )
    parser.add_argument("--acknowledge-truncated-request", default=argparse.SUPPRESS)
    parser.add_argument("--acknowledge-unknown-request", default=argparse.SUPPRESS)
    parser.add_argument("--accepted-review-key", default=argparse.SUPPRESS)
    parser.add_argument("--continue-review-key", default=argparse.SUPPRESS)
    parser.add_argument(
        "--acknowledge-reassessment-request", action="append", default=argparse.SUPPRESS
    )
    parser.add_argument("--prior-review-key", default=argparse.SUPPRESS)
    parser.add_argument("--network-route", default=argparse.SUPPRESS)
    parser.add_argument("--preflight", action="store_true", default=argparse.SUPPRESS)
    args = vars(parser.parse_args(argv))
    if (
        not args.get("review_profile")
        and not args.get("review_method_profile")
        and not args.get("prior_review_key")
        and not args.get("accepted_review_key")
        and not args.get("continue_review_key")
    ):
        parser.error("an explicit review or review-method profile is required")
    result = run_review(**args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("result", {}).get("complete", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
