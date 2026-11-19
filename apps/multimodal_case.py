"""Multidataset preparation and explicit bounded task-relative research entry."""

import argparse
import json
from pathlib import Path

from packages.dataset_adapters.multimodal.adapters import prepare as prepare_input
from packages.dataset_adapters.multimodal.preparation import persist, summary, verify
from packages.dataset_adapters.reading import AdapterError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scope_finalize = commands.add_parser(
        "finalize-input-scope", help="Zero-request recovery of confirmed scope storage overflow"
    )
    for name in ("previous", "target", "output", "authorization"):
        scope_finalize.add_argument("--" + name, required=True)
    scope_finalize.add_argument("--check-only", action="store_true")
    assembly_repair = commands.add_parser(
        "assembly-repair-prepare", help="Tested-code repair before any campaign phase exists"
    )
    assembly_repair.add_argument("--input", required=True)
    assembly_repair.add_argument("--test-proof", action="append", required=True)
    coverage = commands.add_parser(
        "coverage", help="Read-only saved demand/observation/Review ledger; no execution authority"
    )
    coverage.add_argument("--report", required=True)
    coverage.add_argument(
        "--history-root", help="Read only checksum-bound explicit ancestry under this directory"
    )
    phase_repair = commands.add_parser(
        "phase-budget-repair-prepare",
        help="Tested-code amendment for an original zero-dispatch narrowed phase",
    )
    phase_repair.add_argument("--input", required=True)
    phase_repair.add_argument("--test-proof", action="append", required=True)
    phase_repair.add_argument("--supersede-prewire-proof", action="store_true")
    context_repair = commands.add_parser(
        "review-context-repair-prepare",
        help="Tested-code amendment for never-dispatched whole-case context Review",
    )
    context_repair.add_argument("--input", required=True)
    context_repair.add_argument("--test-proof", action="append", required=True)
    context_review = commands.add_parser(
        "context-review-prepare", help="New text Review with verified map/display context"
    )
    context_review.add_argument("--previous", required=True)
    context_review.add_argument("--output", required=True)
    context_review.add_argument("--assessment-id", action="append", required=True)
    context_review.add_argument("--authorization", required=True)
    context_review.add_argument(
        "--metadata-profile",
        choices=[
            "grape-author-methods/1.0.0",
            "harvard-gdp-author-methods/1.0.0",
            "grape-rnfl-device-unit/1.0.0",
            "grape-tonopachy-device-unit/1.0.0",
        ],
        help="Exact author measurement metadata; keep original provider, facts and units",
    )
    capability_window = commands.add_parser(
        "capability-window-prepare",
        help="Automatic demand successor using original facts and cumulative budget",
    )
    capability_window.add_argument("--previous", required=True)
    capability_window.add_argument("--output", required=True)
    capability_window.add_argument("--authorization", required=True)
    capability_window.add_argument("--check-only", action="store_true")
    current_window = commands.add_parser(
        "capability-current-prepare",
        help="Current-demand transition with qualified Review reuse and cumulative budget",
    )
    current_window.add_argument("--previous", required=True)
    current_window.add_argument("--output", required=True)
    current_window.add_argument("--authorization", required=True)
    current_window.add_argument("--check-only", action="store_true")
    current_window.add_argument(
        "--network-route", help="Explicit public route-only successor; never changes old requests"
    )
    scope_unknown = commands.add_parser(
        "capability-scope-attempt-prepare",
        help="Explicit new bounded Scope attempt after original UNKNOWN proof; no old replay",
    )
    scope_unknown.add_argument("--previous", required=True)
    scope_unknown.add_argument("--output", required=True)
    scope_unknown.add_argument("--authorization", required=True)
    scope_unknown.add_argument("--acknowledge-request", required=True)
    scope_unknown.add_argument("--network-route", required=True)
    scope_unknown.add_argument("--check-only", action="store_true")
    scope_unknown.add_argument(
        "--confirmed-invalid", action="store_true", default=argparse.SUPPRESS
    )
    scope_successor = commands.add_parser(
        "capability-scope-successor-prepare",
        help="Reuse accepted local Scope recovery; dispatch only never-attempted survey and Review",
    )
    scope_successor.add_argument("--previous", required=True)
    scope_successor.add_argument("--output", required=True)
    scope_successor.add_argument("--authorization", required=True)
    scope_successor.add_argument("--check-only", action="store_true")
    recovery = commands.add_parser(
        "capability-window-recover",
        help="New bounded successor only after original owners prove no request was dispatched",
    )
    recovery.add_argument("--previous", required=True)
    recovery.add_argument("--output", required=True)
    recovery.add_argument("--authorization", required=True)
    capability_review = commands.add_parser(
        "capability-review-origin-prepare",
        help="Append-only accepted Review correction before any successor dispatch",
    )
    capability_review.add_argument("--output", required=True)
    capability_review.add_argument("--test-proof", required=True)
    capability_review.add_argument("--authorization", required=True)
    renewal_window = commands.add_parser(
        "renewal-window-prepare",
        help="Explicit bounded successor for an expired zero-dispatch acquisition; same provider",
    )
    renewal_window.add_argument("--previous", required=True)
    renewal_window.add_argument("--output", required=True)
    renewal_window.add_argument("--authorization", required=True)
    renewal_activation = commands.add_parser(
        "renewal-window-run",
        help=(
            "Verify unused expired acquisition, prepare a bounded successor and execute immediately"
        ),
    )
    renewal_activation.add_argument("--previous", required=True)
    renewal_activation.add_argument("--output", required=True)
    renewal_activation.add_argument("--authorization", required=True)
    local_review = commands.add_parser(
        "renewal-local-review-run",
        help="New bounded text Review after owner-proven zero-dispatch local failure",
    )
    local_review.add_argument("--previous", required=True)
    local_review.add_argument("--output", required=True)
    local_review.add_argument("--authorization", required=True)
    provider_transition = commands.add_parser(
        "provider-transition-prepare",
        help="New bounded provider window over an expired, never-executed Review plan",
    )
    provider_transition.add_argument("--previous", required=True)
    provider_transition.add_argument("--output", required=True)
    provider_transition.add_argument("--authorization", required=True)
    unscored = commands.add_parser(
        "unscored-acquisition-prepare",
        help="Route-only bounded successor for original-owner unattempted work; no UNKNOWN replay",
    )
    for name in ("previous", "output", "provider-config", "authorization"):
        unscored.add_argument("--" + name, required=True)
    unscored.add_argument("--request-limit", type=int, default=24)
    unscored.add_argument("--seconds", type=int, default=2700)
    unscored.add_argument("--acquisition-target", action="append", default=[])
    unscored.add_argument("--check-only", action="store_true", help="No claim, files or requests")
    progress_repair = commands.add_parser("unscored-progress-repair")
    progress_repair.add_argument("--input", required=True)
    progress_repair.add_argument("--test-proof", required=True)
    provider_transition.add_argument(
        "--model-only-acquisition",
        action="store_true",
        help="Model-only successor of an expired zero-dispatch selected acquisition increment",
    )
    aggregation = commands.add_parser(
        "aggregate-accepted",
        help="Zero-request original-owner aggregation of accepted local judgments",
    )
    aggregation.add_argument("--plan", required=True)
    aggregation.add_argument("--output", required=True)
    aggregation.add_argument(
        "--retry-from", help="Exact rejected local aggregation; new zero-API successor only"
    )
    trial = commands.add_parser("definition-trial-prepare", help="Scope a fresh prepared entry")
    trial.add_argument("--input", required=True)
    trial.add_argument("--expansion-policy", required=True)
    trial.add_argument("--authorization", required=True)
    trial.add_argument("--max-new-requests", type=int, default=12)
    trial.add_argument("--seconds", type=int, default=3600)
    trial.add_argument(
        "--entry-time-window",
        action="store_true",
        help="Explicit fresh trial window bounded by the original entry",
    )
    commands.add_parser("definition-trial-initialize").add_argument("--input", required=True)
    definition_alternative = commands.add_parser(
        "definition-alternative-prepare",
        help="One alternative observation over an admitted, not-yet-reviewed definition",
    )
    for name in ("previous", "output", "target", "authorization"):
        definition_alternative.add_argument("--" + name, required=True)
    definition_alternative.add_argument("--request-limit", type=int, default=3)
    definition_alternative.add_argument("--check-only", action="store_true")
    renewal = commands.add_parser(
        "renewal-prepare", help="Explicit new bounded window, never resume an old unknown"
    )
    checkpoint = commands.add_parser(
        "source-checkpoint-prepare",
        help="Successor for expired confirmed sources-only work; preserve cumulative budget",
    )
    for name in ("previous", "output", "authorization"):
        checkpoint.add_argument("--" + name, required=True)
    checkpoint.add_argument("--check-only", action="store_true")
    renewal.add_argument("--facts", required=True)
    renewal.add_argument("--previous", required=True)
    renewal.add_argument("--output", required=True)
    renewal.add_argument("--mode", choices=("review", "definitions", "acquisition"), required=True)
    renewal.add_argument("--authorization", required=True)
    interrupted = commands.add_parser(
        "definition-interruption-prepare",
        help="New bounded Source window over confirmed interrupted public material; no HTTP replay",
    )
    for name in ("previous", "output", "authorization"):
        interrupted.add_argument("--" + name, required=True)
    interrupted.add_argument("--request-limit", type=int, default=5)
    interrupted.add_argument("--seconds", type=int, default=3600)
    interrupted.add_argument(
        "--local-index-failure",
        action="store_true",
        help="Repair confirmed fetched material through Knowledge; preserve old local failure",
    )
    interrupted.add_argument(
        "--check-only", action="store_true", help="Original-owner reads only; no claim or dispatch"
    )
    confirmed_failure = commands.add_parser(
        "definition-format-successor-prepare",
        help="Format-corrected successor of confirmed Source errors; cumulative budget retained",
    )
    for name in ("previous", "output", "authorization"):
        confirmed_failure.add_argument("--" + name, required=True)
    confirmed_failure.add_argument("--request-limit", type=int, default=4)
    confirmed_failure.add_argument("--seconds", type=int, default=3600)
    confirmed_failure.add_argument("--check-only", action="store_true")
    renewal.add_argument(
        "--review-method-profile",
        choices=("relay-retained-direction-v1",),
        help="Explicit different-model text reassessment; retain facts, never reuse old scores",
    )
    renewal.add_argument(
        "--method-diagnostic",
        action="append",
        default=[],
        help="Exact separately-accounted retained-fact diagnostic; never authorizes replay",
    )
    renewal.add_argument(
        "--unstarted-context-review",
        action="store_true",
        help="New text-only window after a stopped zero-request Review deadline; same provider",
    )
    renewal.add_argument(
        "--alternative-view-method",
        choices=(
            "original_plus_registered_disc_roi_rgb_green/1.0.0",
            "original_plus_full_frame_quadrants_rgb_green/1.0.0",
        ),
        help="One owner-confirmed uninterpretable visual check with an alternative view; "
        "not UNKNOWN replay",
    )
    renewal.add_argument("--expansion-policy")
    renewal.add_argument("--reuse-confirmed-review", action="store_true")
    renewal.add_argument(
        "--review-capacity-recovery",
        action="store_true",
        help="Text-only successor for owner-proven zero-wire batch capacity failure",
    )
    renewal.add_argument(
        "--budget-stop-review",
        action="store_true",
        help="Explicit same-profile successor after confirmed Review dispatch-budget stop; "
        "reuse successful batches, never resend UNKNOWN or corrected batches",
    )
    renewal.add_argument(
        "--same-profile-review",
        action="store_true",
        help=(
            "New bounded text window after a partial Review deadline; "
            "reuse confirmed batches and preserve provider/prompt"
        ),
    )
    renewal.add_argument(
        "--acquisition-increment-only",
        action="store_true",
        help="After local accepted aggregation, acquire/review only previously unattempted checks",
    )
    renewal.add_argument(
        "--pending-review",
        action="store_true",
        help=(
            "Unstarted items of partial Review; with --pending-new-window also "
            "allows an initial failed Review's owner-proven unsent items. "
            "Initial-failure UNKNOWN batches remain excluded."
        ),
    )
    renewal.add_argument(
        "--pending-new-window",
        action="store_true",
        help=(
            "With --pending-review: one bounded text-only window "
            "for owner-proven unsent items after expiry"
        ),
    )
    renewal.add_argument(
        "--pending-runtime-lease",
        action="store_true",
        help="Explicit same-window unsent-only Review under the original Runtime lease",
    )
    renewal.add_argument(
        "--pending-reassessment-window",
        action="store_true",
        help="One explicit new text batch after expiry for an exact acknowledged UNKNOWN; "
        "retain old request/cost, use original Runtime lease, never replay accepted work",
    )
    renewal.add_argument(
        "--recovery-network-route",
        help=(
            "Public route config for a retained-context or unsent-item text successor; "
            "does not replay old unknown requests or change historical receipts"
        ),
    )
    renewal.add_argument(
        "--recovery-http-client",
        help="Explicit public client-only transition for selected unattempted acquisition; "
        "retain original accepted judgments and cumulative budget, never replay UNKNOWN.",
    )
    renewal.add_argument(
        "--historical-review-origin",
        help="With explicit pending reassessment: original failed ancestor campaign; "
        "retain current ledger and accepted work",
    )
    pending = commands.add_parser(
        "pending-review-preview", help="Read-only original pending inventory; no new run"
    )
    pending.add_argument("--facts", required=True)
    pending.add_argument("--historical-review-origin")
    pending.add_argument("--acknowledge-unknown-request")
    pending.add_argument("--acknowledge-format-request", action="append", default=[])
    renewal.add_argument(
        "--repair-review-format",
        action="store_true",
        help="Format-only successor; reuse confirmed batches, account known schema failures",
    )
    renewal.add_argument("--request-limit", type=int)
    renewal.add_argument(
        "--review-acquisition-gap",
        action="store_true",
        help="Text-only closure of accepted new observations excluded by an inherited Review scope",
    )
    renewal.add_argument(
        "--exact-request-budget",
        action="store_true",
        help="Selected acquisition: charge actual dispatches, retain zero-request closure",
    )
    renewal.add_argument(
        "--shape-contract-review",
        action="store_true",
        help="Uniform text reassessment with a new format prompt; reuse observations only",
    )
    renewal.add_argument(
        "--root-direction-review",
        action="store_true",
        help=(
            "New root-bound text question; uniformly reassess retained observations, not old scores"
        ),
    )
    renewal.add_argument(
        "--local-finalization",
        help="Exact saved partial closure; only with --root-direction-review",
    )
    renewal.add_argument(
        "--review-batch-size",
        type=int,
        choices=(1, 2, 3),
        help="Explicit citation-closure root successor; qualified wire, no implicit old upgrade",
    )
    renewal.add_argument(
        "--acquisition-target",
        action="append",
        default=[],
        help="Explicit next slice: 1-3 owner-eligible unattempted checks, remaining work preserved",
    )
    renewal.add_argument(
        "--acknowledge-unknown-acquisition-request",
        action="append",
        default=[],
        help="Old Vision request IDs for one bounded new attempt; original UNKNOWN remains",
    )
    renewal.add_argument("--acknowledge-format-request", action="append", default=[])
    renewal.add_argument(
        "--historical-acquisition-origin",
        help="Exact ancestor Vision UNKNOWN with an already bound HTTP-client-only upgrade",
    )
    renewal.add_argument(
        "--acknowledge-confirmed-invalid-acquisition-request",
        action="append",
        default=[],
        help="Exact schema-failed Vision request: one new attempt, not UNKNOWN replay",
    )
    truncated = commands.add_parser("truncated-review-inspect")
    truncated.add_argument("--input", required=True)
    truncated.add_argument("--request-id", required=True)
    renewal.add_argument(
        "--acknowledge-truncated-request",
        help="Exact confirmed truncated Review; one bounded 3-to-1 text-only successor",
    )
    renewal.add_argument(
        "--format-closure-seconds",
        type=int,
        help="Explicit new text-only known-format closure window, at most 2700 seconds",
    )
    renewal.add_argument(
        "--repair-acquired-contract",
        action="store_true",
        help="One 2.3 citation-contract successor after confirmed 1.19 HTTP400; same window",
    )
    renewal.add_argument(
        "--review-acquired-only",
        action="store_true",
        help="Text Review of confirmed new observations after acquisition stops; same window",
    )
    renewal.add_argument(
        "--continue-acquisition",
        action="store_true",
        help="One confirmed-decline acquisition successor; retain original deadline/budget",
    )
    renewal.add_argument(
        "--acknowledge-unknown-request",
        help="Exact old request ID for one independent text reassessment; UNKNOWN remains charged",
    )
    renewal.add_argument(
        "--review-definition-increment",
        action="store_true",
        help=(
            "Review accepted interpretation after a stopped zero-wire preparation timeout; "
            "no new Source/Vision"
        ),
    )
    renewal.add_argument(
        "--clarify-review",
        action="append",
        default=[],
        help=(
            "New root-proposition clarification for at most two accepted items; "
            "may share a selected --exact-request-budget acquisition Review batch"
        ),
    )
    renewal.add_argument(
        "--confirmed-facts-review",
        action="store_true",
        help="New text-only review window over confirmed facts; no old request replay",
    )
    renewal.add_argument("--definition-increment-only", action="store_true")
    renewal.add_argument(
        "--continuation",
        action="store_true",
        help="Retain definition gate deadline and remaining budget; indexed text only",
    )
    commands.add_parser("renewal-run").add_argument("--input", required=True)
    execution_lease = commands.add_parser(
        "renewal-execution-lease",
        help="Explicit Runtime lease for a prepared, never-started acquisition with ENOLCK",
    )
    execution_lease.add_argument("--input", required=True)
    execution_lease.add_argument("--proof", required=True)
    execution_lease.add_argument("--authorization", required=True)
    renewal_repair = commands.add_parser("renewal-repair")
    renewal_repair.add_argument("--input", required=True)
    renewal_repair.add_argument("--proof", required=True)
    renewal_repair.add_argument("--authorization", required=True)
    recovery = commands.add_parser(
        "restore-report", help="Zero-dispatch saved-result export recovery"
    )
    recovery.add_argument("--input", required=True)
    recovery.add_argument(
        "--retained-context",
        action="store_true",
        help="Separate owner-checked context report revision; preserve saved score/report",
    )
    review_prepare = commands.add_parser(
        "review-status-prepare", help="One confirmed-status Review successor, original deadline"
    )
    review_prepare.add_argument("--previous", required=True)
    review_prepare.add_argument("--output", required=True)
    review_run = commands.add_parser(
        "review-status-run", help="Run the explicitly prepared text-only Review successor"
    )
    review_run.add_argument("--input", required=True)
    repair = commands.add_parser(
        "authorize-local-search-repair",
        help="Zero-wire failed assembly only; retain original budget/deadline",
    )
    repair.add_argument("--input", required=True)
    repair.add_argument("--failure-log", required=True)
    repair.add_argument("--test-proof", required=True, action="append")
    material_repair = commands.add_parser("authorize-material-filter-repair")
    material_repair.add_argument("--input", required=True)
    material_repair.add_argument("--diagnostic", required=True)
    material_repair.add_argument("--test-proof", required=True, action="append")
    projection_repair = commands.add_parser("authorize-definition-projection-repair")
    projection_repair.add_argument("--input", required=True)
    projection_repair.add_argument("--proof", required=True)
    projection_repair.add_argument("--authorization", required=True)
    trees = commands.add_parser(
        "proposition-trees", help="Zero-API independent historical proposition trees"
    )
    trees.add_argument("--source", required=True)
    trees.add_argument("--output", required=True)
    final = commands.add_parser(
        "finalize-partial", help="Zero-API historical partial fact closure; no resume"
    )
    final.add_argument("--previous", required=True)
    final.add_argument("--output", required=True)
    final.add_argument(
        "--recover-confirmed-raw",
        action="store_true",
        help="Recover confirmed raw batches locally after a stopped text-method deadline",
    )
    final.add_argument(
        "--citation-rejection",
        action="store_true",
        help="Explicit zero-API closure of confirmed batches after exact citation rejection",
    )
    recorded = commands.add_parser(
        "finalize-recorded", help="Zero-API complete recorded Review closure"
    )
    recorded.add_argument("--previous", required=True)
    recorded.add_argument("--output", required=True)
    recorded.add_argument(
        "--lossless-explanations",
        action="store_true",
        help="Explicit zero-API interpretation of identical repeated explanation fields only",
    )
    live = commands.add_parser("live-prepare")
    live.add_argument(
        "--dataset", required=True, choices=("harvard_gdp", "gamma", "grape", "fairvlmed")
    )
    live.add_argument("--root", required=True)
    live.add_argument("--sample", required=True)
    live.add_argument(
        "--discovery-continuation-of",
        help="Stopped discovery successor; same case, deadline and budget, no failed-query replay",
    )
    live.add_argument("--task", required=True)
    live.add_argument(
        "--definition-failure-of",
        help="Explicit same-window successor of a confirmed Source schema failure",
    )
    live.add_argument(
        "--defer-failed-definition-expansion",
        action="store_true",
        help="Explicit Source backlog (qualified GRAPE schema, component or batched root timeout); "
        "no Source replay, retain original window",
    )
    live.add_argument(
        "--retain-resolved-definition-merges",
        action="store_true",
        help="Explicitly preserve owner-confirmed non-applied definition merges as history; "
        "requires routed Source deferral, does not replay or apply the unused relation",
    )
    live.add_argument(
        "--retain-closed-public-wire-failures",
        action="store_true",
        help="Versioned routed Source backlog: retain original locally closed public HTTP "
        "no-response receipts and budget; never replay or use them as source material",
    )
    live.add_argument(
        "--source-fallback-seconds",
        type=int,
        help="Explicit routed Source-failure successor window, 600–5400 seconds; "
        "original failure and cumulative request budget remain",
    )
    live.add_argument(
        "--source-fallback-authorization",
        help="Reason for the explicit bounded successor; requires --source-fallback-seconds",
    )
    live.add_argument(
        "--registry",
        help="Explicit task registry; automatic entry defaults to v2, legacy entry to v1",
    )
    live.add_argument(
        "--component-consumer",
        help="Pinned original-06 confirmed result; same registered case, no tool re-execution",
    )
    live.add_argument("--reuse-procedures", action="store_true")
    live.add_argument(
        "--active-acquisition",
        help="Explicit source-bound demand profile; native-tool-first acquisition",
    )
    live.add_argument(
        "--automatic-requirements",
        action="store_true",
        help="Match registered questions to the proposition and standardized input capabilities; "
        "root-question entry binds current input-survey and method policies",
    )
    live.add_argument(
        "--active-continuation-of",
        help="One new bounded active attempt; retain old request budget and confirmed facts",
    )
    live.add_argument(
        "--resilient-vision", action="store_true", help="Compact target-only wire and phase timing"
    )
    live.add_argument(
        "--continuation-of", help="Explicit new bounded acquisition; reuse confirmed parent facts"
    )
    live.add_argument(
        "--review-from", help="New bounded Review for unstarted propositions; no new images"
    )
    live.add_argument("--view-profile", choices=("overview-v1", "detail-v1"), default="overview-v1")
    live.add_argument("--input-semantics", choices=("harvard-gdp-v1", "grape-v1"))
    live.add_argument(
        "--review-batch-size",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Explicit first-entry root Review batch size; no mutation of old run configuration",
    )
    live.add_argument(
        "--execution-profile",
        choices=("relay-context-v1", "relay-snapshot-context-v1", "relay-root-question-v1"),
        help="Explicit whole-case profile; root-question-v1 binds the root and Review 2.7",
    )
    live.add_argument(
        "--network-route",
        help="Public frozen HTTPS proxy policy; fresh batched-root entry only, "
        "no old request replay",
    )
    live.add_argument(
        "--model-selection",
        help="Public model-only binding for a new routed root campaign; never replay UNKNOWN.",
    )
    live.add_argument(
        "--http-client",
        help="Explicit public bounded client binding; new entry, never mutate an old request.",
    )
    live.add_argument(
        "--entry-execution-lock",
        choices=("multimodal-runtime-entry/1.0.0", "multimodal-runtime-source-backlog/1.0.0"),
        help="Explicit fresh or Source-backlog Runtime lease; old inputs are not upgraded.",
    )
    live.add_argument(
        "--vision-feedback",
        choices=("state-consistency-once/1.0.0",),
        help="Explicit new Vision contract: one confirmed state-consistency correction, "
        "charged to the original budget; no UNKNOWN replay.",
    )
    live.add_argument(
        "--review-contract",
        choices=("rating-presence/1.0.0", "rating-presence/1.1.0"),
        default=argparse.SUPPRESS,
        help="Explicit Review Rating presence / owner identity projection contract; "
        "old runs and scores are not upgraded.",
    )
    live.add_argument(
        "--independent-propositions",
        action="store_true",
        help="Shared acquisition; separate original Review runs and trees",
    )
    live.add_argument("--selection", choices=("training", "development_preparation"))
    live.add_argument("--cutoff-visit", type=int)
    live.add_argument("--output", required=True)
    live.add_argument(
        "--previous", help="Completed technical-failure directory; budget is retained"
    )
    live.add_argument("--known-measurements-only", dest="known_only", action="store_true")
    live.add_argument("--confirmed-observations-only", action="store_true")
    live.add_argument(
        "--review-replacement",
        action="store_true",
        help="One single-item replacement of cancelled read-only Review; not tool recovery",
    )
    commands.add_parser("run").add_argument("--input", required=True)
    commands.add_parser(
        "preflight", help="Local entry/resource check; no credentials or DB"
    ).add_argument("--input", required=True)
    for action in ("check", "prepare"):
        cmd = commands.add_parser(action)
        cmd.add_argument(
            "--dataset", required=True, choices=("harvard_gdp", "gamma", "grape", "fairvlmed")
        )
        cmd.add_argument("--root", required=True)
        cmd.add_argument("--sample", required=True)
        cmd.add_argument("--selection", choices=("training", "development_preparation"))
        cmd.add_argument("--cutoff-visit", type=int)
        cmd.add_argument(
            "--input-semantics",
            choices=("harvard-gdp-v1", "grape-v1"),
            help="Original adapter's explicit release semantics; default keeps historical decoding",
        )
        if action == "prepare":
            cmd.add_argument("--output", required=True)
    commands.add_parser("verify").add_argument("--input", required=True)
    capabilities = commands.add_parser(
        "capability-plan",
        help="Verify input and save representation capabilities; no API, DB or clinical grant",
    )
    capabilities.add_argument("--input", required=True, help="Original input/preparation.json")
    capabilities.add_argument(
        "--output", required=True, help="New capability snapshot, never overwrite"
    )
    capability_verify = commands.add_parser(
        "capability-verify",
        help="Read capability snapshot back against original input; no execution",
    )
    capability_verify.add_argument("--input", required=True)
    capability_verify.add_argument("--snapshot", required=True)
    selection = commands.add_parser(
        "requirements-plan",
        help="Read original Knowledge/input owners and preview automatic requirements; no dispatch",
    )
    selection_origin = selection.add_mutually_exclusive_group(required=True)
    selection_origin.add_argument("--input")
    selection_origin.add_argument(
        "--previous", help="Reconcile a stopped original-owner campaign; no execution grant"
    )
    selection.add_argument("--task")
    selection.add_argument("--output", required=True)
    selection.add_argument("--registry", default="configs/research/multimodal_propositions_v1.json")
    plan = commands.add_parser("plan")
    plan.add_argument("--input", required=True)
    plan.add_argument("--output", required=True)
    reuse_plan = commands.add_parser(
        "reuse-plan",
        help=(
            "Build an initial proposition evidence-tree plan from the prepared input, "
            "qualitative result and procedure-only Memory hints; no execution"
        ),
    )
    reuse_input = reuse_plan.add_mutually_exclusive_group(required=True)
    reuse_input.add_argument("--input", help="Verified preparation.json")
    reuse_input.add_argument("--scope-input", help="Verified patient-level planning scope")
    reuse_plan.add_argument("--task", required=True, help="Task/proposition id in the registry")
    reuse_plan.add_argument("--output", required=True, help="New output directory")
    reuse_plan.add_argument(
        "--registry", default="configs/research/multimodal_propositions_v1.json"
    )
    reuse_plan.add_argument(
        "--catalog-profile",
        default="configs/research/multimodal_catalog_acquisition_v2.json",
    )
    reuse_plan.add_argument("--qualitative", help="Saved qualitative result bound to this input")
    reuse_plan.add_argument(
        "--memory-index",
        help="Read-only exported Memory procedure index; no private credentials are read",
    )
    reuse_plan.add_argument(
        "--installed-memory",
        action="store_true",
        help="Read the installed Memory owner through its reader port; never writes or dispatches",
    )
    bilateral = commands.add_parser(
        "baseline-bilateral-scope",
        help="Compose two immutable GRAPE baseline eye preparations for planning only",
    )
    bilateral.add_argument("--input", action="append", required=True, help="OD and OS preparation.json")
    bilateral.add_argument("--output", required=True, help="New scope directory")
    task = commands.add_parser("task-plan")
    task.add_argument("--input", required=True)
    task.add_argument("--task", required=True)
    task.add_argument("--output", required=True)
    task.add_argument("--registry", default="configs/research/multimodal_tasks_v1.json")
    pilot = commands.add_parser("pilot-prepare")
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--manifest", default="configs/research/multimodal_pilot_v1.json")
    from packages.multimodal_research.stages import STAGES

    for name in ("status", "check", "preview", "prepare", "run", "verify", "resume", "export"):
        stage_command = commands.add_parser("stage-" + name)
        stage_command.add_argument("--input", required=True)
        if name in {"run", "verify", "resume"}:
            stage_command.add_argument("--stage", required=True, choices=STAGES)
        if name == "resume":
            stage_command.add_argument("--definition-repair-proof")
            stage_command.add_argument(
                "--include-unstarted-acquisition",
                action="store_true",
                help="Explicit one saved-definition acquisition before Review",
            )
            stage_command.add_argument("--authorization")
    args = parser.parse_args()
    if args.command == "finalize-input-scope":
        from packages.multimodal_research.campaign_renewal import assemble
        from packages.multimodal_research.finalization import finalize_scope

        result = finalize_scope(
            assemble(args.previous),
            args.target,
            args.output,
            authorization=args.authorization,
            check_only=args.check_only,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "capability-scope-attempt-prepare":
        from packages.multimodal_research.scope_unknown_successor import prepare

        print(
            json.dumps(prepare(**{k: v for k, v in vars(args).items() if k != "command"})),
            flush=True,
        )
        return 0
    if args.command == "capability-current-prepare":
        from packages.multimodal_research.capability_window import prepare_current

        print(
            json.dumps(prepare_current(**{k: v for k, v in vars(args).items() if k != "command"})),
            flush=True,
        )
        return 0
    if args.command == "capability-scope-successor-prepare":
        from packages.multimodal_research.scope_successor import prepare

        print(
            json.dumps(prepare(**{k: v for k, v in vars(args).items() if k != "command"})),
            flush=True,
        )
        return 0
    if args.command == "capability-window-recover":
        from packages.multimodal_research.capability_recovery import prepare

        print(json.dumps(prepare(**{k: v for k, v in vars(args).items() if k != "command"})))
        return 0
    if args.command == "capability-review-origin-prepare":
        from packages.multimodal_research.capability_review_origin import prepare

        print(json.dumps(prepare(**{k: v for k, v in vars(args).items() if k != "command"})))
        return 0
    if args.command == "capability-window-prepare":
        from packages.multimodal_research.capability_window import prepare

        print(
            json.dumps(prepare(**{k: v for k, v in vars(args).items() if k != "command"})),
            flush=True,
        )
        return 0
    if args.command == "requirements-plan":
        if args.previous:
            if args.task:
                parser.error("historical requirements-plan retains the original task")
            from packages.multimodal_research.requirement_catalog import prepare_transition

            print(json.dumps(prepare_transition(args.previous, args.output)), flush=True)
            return 0
        if not args.task:
            parser.error("requirements-plan --input requires --task")
        from packages.multimodal_research.requirement_catalog import prepare_selection

        print(
            json.dumps(
                prepare_selection(
                    args.input,
                    args.task,
                    args.output,
                    registry=args.registry,
                )
            ),
            flush=True,
        )
        return 0
    if args.command == "capability-verify":
        from packages.dataset_adapters.multimodal.capabilities import read_snapshot

        value = read_snapshot(args.input, args.snapshot)
        print(json.dumps(dict(checksum=value.checksum, matches=True, execution_authorized=False)))
        return 0
    if args.command == "capability-plan":
        from packages.dataset_adapters.multimodal.capabilities import write_snapshot

        value = write_snapshot(args.input, args.output)
        print(
            json.dumps(
                dict(
                    output=args.output,
                    checksum=value.checksum,
                    input_checksum=value.input_checksum,
                    components=len(value.components),
                    qualification=value.qualification,
                    execution_authorized=False,
                    api_requests=0,
                    model_runs=0,
                    database_writes=0,
                )
            ),
            flush=True,
        )
        return 0
    if args.command == "coverage":
        from packages.multimodal_research.demand_coverage import inspect_saved

        if args.history_root:
            from packages.multimodal_research.demand_history import inspect_history

            result = inspect_history(args.report, args.history_root)
        else:
            result = inspect_saved(args.report)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "assembly-repair-prepare":
        from packages.multimodal_research.assembly_recovery import authorize

        print(json.dumps(authorize(args.input, args.test_proof)), flush=True)
        return 0
    if args.command == "phase-budget-repair-prepare":
        from packages.multimodal_research.context_recovery import authorize_budget_repair

        print(
            json.dumps(
                authorize_budget_repair(
                    args.input, args.test_proof, supersede=args.supersede_prewire_proof
                )
            )
        )
        return 0
    if args.command == "review-context-repair-prepare":
        from packages.multimodal_research.context_recovery import authorize

        print(json.dumps(authorize(args.input, args.test_proof), ensure_ascii=False))
        return 0
    if args.command == "pending-review-preview":
        from packages.multimodal_research.campaign_renewal import completed_parent, stopped_tip
        from packages.multimodal_research.pending_review import inventory

        parent, _ = completed_parent(args.facts)
        value = inventory(
            parent,
            stopped_tip(parent, args.facts),
            args.acknowledge_unknown_request,
            args.acknowledge_format_request,
            historical_origin=args.historical_review_origin,
        )
        print(
            json.dumps(dict(inventory=value, api_requests=0, database_writes=0), ensure_ascii=False)
        )
        return 0
    if args.command == "renewal-execution-lease":
        from packages.multimodal_research.execution_lease import authorize

        print(json.dumps(authorize(args.input, args.proof, args.authorization)))
        return 0
    if args.command == "renewal-repair":
        from packages.multimodal_research.campaign_renewal import authorize_prewire_repair

        print(json.dumps(authorize_prewire_repair(args.input, args.proof, args.authorization)))
        return 0
    if args.command == "aggregate-accepted":
        from packages.multimodal_research.accepted_aggregation import aggregate_entry

        result = aggregate_entry(
            args.plan,
            args.output,
            retry_from=args.retry_from,
            progress=lambda value: print(json.dumps(value, ensure_ascii=False), flush=True),
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    if args.command.startswith("definition-trial-"):
        from packages.multimodal_research.definition_trial import (
            initialize,
        )
        from packages.multimodal_research.definition_trial import (
            prepare as trial_prepare,
        )

        if args.command == "definition-trial-initialize":
            result = initialize(args.input)
        else:
            result = trial_prepare(
                args.input,
                expansion_policy=args.expansion_policy,
                authorization=args.authorization,
                max_new_requests=args.max_new_requests,
                seconds=args.seconds,
                entry_time_window=args.entry_time_window,
            )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    if args.command.startswith("stage-"):
        from packages.multimodal_research.stages import cli

        result = cli(args)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 2 if result.get("complete") is False else 0
    if args.command == "context-review-prepare":
        from packages.multimodal_research.context_review import prepare as prepare_context

        result = prepare_context(**{k: v for k, v in vars(args).items() if k != "command"})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    if args.command == "source-checkpoint-prepare":
        from packages.multimodal_research.source_checkpoint import prepare as prepare_checkpoint

        result = prepare_checkpoint(**{k: v for k, v in vars(args).items() if k != "command"})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    if args.command in {"renewal-window-prepare", "renewal-window-run"}:
        from packages.multimodal_research.unstarted_window import prepare as prepare_window
        from packages.multimodal_research.unstarted_window import prepare_and_run

        action = prepare_and_run if args.command == "renewal-window-run" else prepare_window
        result = action(**{k: v for k, v in vars(args).items() if k != "command"})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if args.command == "renewal-window-prepare" or result["complete"] else 2
    if args.command == "renewal-local-review-run":
        from packages.multimodal_research.local_review_recovery import prepare_and_run

        result = prepare_and_run(**{k: v for k, v in vars(args).items() if k != "command"})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result["complete"] else 2
    if args.command == "provider-transition-prepare":
        from packages.multimodal_research.provider_transition import prepare as prepare_transition

        result = prepare_transition(**{k: v for k, v in vars(args).items() if k != "command"})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    if args.command == "unscored-progress-repair":
        from packages.multimodal_research.unscored_acquisition import authorize_progress_repair

        print(json.dumps(authorize_progress_repair(args.input, args.test_proof)), flush=True)
        return 0
    if args.command == "unscored-acquisition-prepare":
        from packages.multimodal_research.unscored_acquisition import prepare as prepare_unscored

        result = prepare_unscored(**{k: v for k, v in vars(args).items() if k != "command"})
        print(json.dumps(result), flush=True)
        return 0
    if args.command == "definition-alternative-prepare":
        from packages.multimodal_research.definition_alternative import prepare

        result = prepare(
            previous=args.previous,
            output=args.output,
            target=args.target,
            authorization=args.authorization,
            request_limit=args.request_limit,
            check_only=args.check_only,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "renewal-prepare":
        from packages.multimodal_research.campaign_renewal import prepare as prepare_renewal

        print(
            json.dumps(prepare_renewal(**{k: v for k, v in vars(args).items() if k != "command"})),
            flush=True,
        )
        return 0
    if args.command == "definition-format-successor-prepare":
        from packages.multimodal_research.confirmed_definition_failure import prepare

        print(
            json.dumps(prepare(**{k: v for k, v in vars(args).items() if k != "command"})),
            flush=True,
        )
        return 0
    if args.command == "definition-interruption-prepare":
        from packages.multimodal_research.interrupted_definition import (
            prepare as prepare_interrupted,
        )

        print(
            json.dumps(
                prepare_interrupted(**{k: v for k, v in vars(args).items() if k != "command"})
            ),
            flush=True,
        )
        return 0
    if args.command == "truncated-review-inspect":
        from packages.multimodal_research.truncated_review import inspect

        print(json.dumps(inspect(args.input, args.request_id), ensure_ascii=False), flush=True)
        return 0
    if args.command == "renewal-run":
        from packages.multimodal_research.campaign_renewal import run as run_renewal

        result = run_renewal(args.input)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result["complete"] else 2
    if args.command in {"review-status-prepare", "review-status-run"}:
        from packages.multimodal_research.review_status_recovery import prepare as prepare_review
        from packages.multimodal_research.review_status_recovery import run

        result = (
            prepare_review(args.previous, args.output)
            if args.command == "review-status-prepare"
            else run(args.input)
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result.get("complete", True) else 2
    if args.command == "restore-report":
        from packages.multimodal_research.report_recovery import restore, restore_context_report

        action = restore_context_report if args.retained_context else restore
        print(json.dumps(action(args.input), ensure_ascii=False), flush=True)
        return 0
    if args.command == "authorize-local-search-repair":
        from packages.multimodal_research.search_recovery import authorize

        print(
            json.dumps(authorize(args.input, failure_log=args.failure_log, tests=args.test_proof)),
            flush=True,
        )
        return 0
    if args.command == "authorize-material-filter-repair":
        from packages.multimodal_research.archive_recovery import authorize_material_filter
        from packages.multimodal_research.campaign_renewal import assemble

        print(
            json.dumps(
                authorize_material_filter(
                    assemble(args.input), diagnostic=args.diagnostic, tests=args.test_proof
                )
            ),
            flush=True,
        )
        return 0
    if args.command == "authorize-definition-projection-repair":
        from packages.multimodal_research.campaign_renewal import assemble
        from packages.multimodal_research.pending_review import (
            authorize_partition_projection_repair,
        )

        campaign = assemble(args.input)
        if campaign.protocol.get("schema_version") != "multimodal-campaign-renewal/1.56.0":
            raise ValueError("definition_projection_repair_protocol")
        if (campaign.output / "prewire-repair.json").exists():
            from packages.multimodal_research.archive_recovery import (
                authorize_definition_projection_guard,
            )

            print(
                json.dumps(
                    authorize_definition_projection_guard(
                        campaign, proof=args.proof, authorization=args.authorization
                    )
                ),
                flush=True,
            )
            return 0
        print(
            json.dumps(
                authorize_partition_projection_repair(campaign, args.proof, args.authorization)
            ),
            flush=True,
        )
        return 0
    if args.command == "proposition-trees":
        from packages.multimodal_research.proposition_trees import export_saved

        print(json.dumps(export_saved(args.source, args.output), ensure_ascii=False), flush=True)
        return 0
    if args.command in {"finalize-partial", "finalize-recorded"}:
        from packages.multimodal_research.finalization import finalize_entry

        print(
            json.dumps(
                finalize_entry(
                    args.previous,
                    args.output,
                    complete=args.command == "finalize-recorded",
                    lossless_explanations=getattr(args, "lossless_explanations", False),
                    citation_rejection=getattr(args, "citation_rejection", False),
                    recover_confirmed_raw=getattr(args, "recover_confirmed_raw", False),
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    if args.command == "live-prepare":
        from packages.multimodal_research.entry import prepare_entry

        print(
            json.dumps(prepare_entry(**{k: v for k, v in vars(args).items() if k != "command"})),
            flush=True,
        )
        return 0
    if args.command in {"run", "preflight"}:
        from packages.multimodal_research.entry_preflight import command

        result = command(args.input, preflight_only=args.command == "preflight")
        print(json.dumps(result), flush=True)
        return (
            0 if result.get("complete") or result.get("status") == "local_preflight_passed" else 2
        )
    if args.command == "task-plan":
        from packages.dataset_adapters.multimodal.tasks import write_projection

        value = write_projection(args.input, args.task, args.output, registry=args.registry)
        print(
            json.dumps(
                dict(
                    output=args.output,
                    checksum=value["checksum"],
                    task=value["task"]["code"],
                    required_classifier=None,
                    execution_authorized=False,
                    api_requests=0,
                )
            ),
            flush=True,
        )
        return 0
    if args.command == "plan":
        from packages.dataset_adapters.multimodal.analysis import from_preparation

        value = from_preparation(args.input)
        with Path(args.output).open("x") as stream:
            stream.write(value.model_dump_json(indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    output=args.output,
                    checksum=value.checksum,
                    inference_ready=False,
                    api_requests=0,
                )
            ),
            flush=True,
        )
        return 0
    if args.command == "reuse-plan":
        from packages.multimodal_research.node_planner import run

        result = run(
            input_path=args.input,
            task_id=args.task,
            output=args.output,
            registry=args.registry,
            catalog_profile=args.catalog_profile,
            qualitative_path=args.qualitative,
            memory_index=args.memory_index,
            installed_memory=args.installed_memory,
            scope_input=args.scope_input,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    if args.command == "baseline-bilateral-scope":
        from packages.multimodal_research.scope import create_baseline_bilateral

        print(json.dumps(create_baseline_bilateral(args.input, args.output), ensure_ascii=False), flush=True)
        return 0
    if args.command == "pilot-prepare":
        config = json.loads(Path(args.manifest).read_bytes())
        if (
            config.get("schema_version") != "multimodal-pilot-selection/1.0.0"
            or config.get("purpose") != "input_preparation_only"
            or config.get("real_api_authorization") is not False
            or not 1 <= len(config.get("cases", [])) <= 9
        ):
            raise AdapterError("multimodal_pilot_manifest")
        for index, case in enumerate(config["cases"]):
            value = prepare_input(**case)
            path = persist(value, Path(args.output) / (str(index) + "-" + case["dataset"]))
            print(json.dumps(dict(**summary(verify(path)), output=str(path))), flush=True)
        return 0
    if args.command == "verify":
        result = summary(verify(args.input))
    else:
        value = prepare_input(
            args.dataset,
            args.root,
            args.sample,
            selection=args.selection,
            cutoff_visit=args.cutoff_visit,
            semantic_profile=args.input_semantics,
        )
        result = summary(value.manifest)
        if args.command == "prepare":
            path = persist(value, args.output)
            verify(path)
            result["output"] = str(path)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AdapterError as error:
        print(json.dumps({"error": str(error), "llm_requests": 0}), flush=True)
        raise SystemExit(2) from None
