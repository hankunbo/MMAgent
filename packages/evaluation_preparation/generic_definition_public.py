"""Original-journal accounting and source bindings for the public definition stage.

No HTTP client, independent ledger or new authorization port lives here. The
existing connector reserves each wire attempt in DefinitionExpansion, while
Source and Review retain their existing RuntimeAgentJournal reservations.
"""

import json

from packages.evidence_reuse.definition_sources import LiteratureDefinitionConnector
from packages.grounding.models import digest
from packages.multimodal_research.definition_expansion import GenericPublicInterpretationPolicy
from packages.research_runtime.agent_records import RuntimeDocuments


def public_root(view, policy):
    # The current input owner reconstructed this public title from the original
    # source-driven root. Case-specific hashed GOAL codes are never sent as
    # staging vocabulary, and a general glaucoma question is never relabelled
    # early/moderate/advanced just to fit the targeted search builder.
    goal = view.public_goal
    if (
        type(policy) is not GenericPublicInterpretationPolicy
        or goal.get("schema_version") != "generic-original-public-proposition/1.0.0"
        or goal.get("title") != "glaucoma"
        or goal.get("original_root_key") != "glaucoma"
        or goal.get("target") != policy.public_target
        or digest(goal) != policy.public_proposition_checksum
    ):
        raise ValueError("generic_definition_public_query_root_scope")


class GenericPublicDefinitionConnector(LiteratureDefinitionConnector):
    """Only the public goal projection differs; original search/fetch/governor."""

    def search_target(self):
        self.expansion.campaign.base.check()
        public_root(self.expansion.campaign.base, self.expansion.policy)
        if super().search_target() != self.expansion.binding["target"]:
            raise ValueError("generic_definition_public_query_root_binding")
        return None  # The existing broad glaucoma plan; no patient code on wire.


def operations(documents, policy, expected_binding):
    keys = tuple(f"definition:operation:{n}" for n in range(policy.max_operations))
    values = (
        documents.read_many(keys)
        if type(documents) is RuntimeDocuments
        else tuple(documents.read(key) for key in keys)
    )
    result, seen, gap = [], set(), False
    for slot, saved in enumerate(values):
        if saved is None:
            gap = True
            continue
        if gap or len(saved) != 1:
            raise ValueError("generic_definition_public_operation_sequence")
        row = saved[0]
        if (
            not isinstance(row, dict)
            or row.get("binding") != expected_binding
            or not isinstance(row.get("key"), str)
            or row["key"] in seen
        ):
            raise ValueError("generic_definition_public_operation_binding")
        seen.add(row["key"])
        if row.get("kind") == "wire":
            # An unconfirmed reservation is still charged. Absence of outcome
            # does not authorize a resend or turn this into an unsent request.
            if row["key"] != f"definition:wire:{slot}" or row.get("wire_kind") not in {
                "search",
                "fetch",
            }:
                raise ValueError("generic_definition_public_wire_binding")
        result.append(row)
    for kind in ("search", "fetch"):
        if sum(r.get("wire_kind") == kind for r in result) > getattr(
            policy, "max_" + kind + "_requests"
        ):
            raise ValueError("generic_definition_public_wire_budget")
    return result


def wire_usage(campaign):
    policy = campaign.checked.policy
    if type(policy) is not GenericPublicInterpretationPolicy:
        return 0
    path = campaign.output / "definition" / "request.json"
    if not path.is_file():
        return 0
    request = json.loads(path.read_bytes())
    run = request["lock"]["run_id"]
    context = campaign.contexts.read_research_run(run)
    if (
        context is None
        or context.run_id != run
        or context.request.model_dump(mode="json") != request
    ):
        raise ValueError("generic_definition_public_request_binding")
    docs = campaign.documents(run)
    saved = docs.read("generic-definition-stage-input")
    if (
        not saved
        or len(saved) != 1
        or saved[0].get("binding") != campaign.checked.binding
        or saved[0].get("request_checksum") != digest(context.request)
        or saved[0].get("protocol_checksum") != digest(campaign.protocol)
    ):
        raise ValueError("generic_definition_public_stage_binding")
    view = campaign.checked.view
    expected = dict(
        campaign_id=campaign.spec["key"],
        input_checksum=view.manifest.checksum,
        target=view.task.target_code,
        protocol_checksum=digest(campaign.protocol),
        parent_requirements_checksum=view.original_rows_checksum,
        policy_checksum=digest(policy),
    )
    return sum(r["kind"] == "wire" for r in operations(docs, policy, expected))


def check_sources(stage):
    sources, policy = stage.sources, stage.policy
    if type(policy) is not GenericPublicInterpretationPolicy:
        if (
            sources.index_keys != policy.index_keys
            or sources.search_public is not None
            or sources.fetch_hit is not None
        ):
            raise ValueError("generic_definition_stage_frozen_accounting_binding")
        return
    connector = stage.public_connector
    if connector is None:
        if sources.search_public is not None or sources.fetch_hit is not None:
            raise ValueError("generic_definition_public_connector_binding")
    elif (
        type(connector) is not GenericPublicDefinitionConnector
        or connector.repository is not sources.repository
        or connector.expansion is not stage.expansion
        or connector.topics != policy.public_topic_sets
        or sources.search_public != connector.search
        or sources.fetch_hit != connector.fetch
    ):
        raise ValueError("generic_definition_public_connector_binding")
    # The index pool may grow only from this campaign's confirmed original
    # query operations, after Knowledge.restore_indexes checked the receipts.
    # Current material reads still check withdrawal/purpose at the original owner.
    confirmed = {}
    if stage.expansion is not None:
        for row in operations(stage.documents, policy, stage.expansion.binding):
            if row["kind"] != "local_search":
                continue
            saved = stage.documents.read(row["key"] + ":result")
            if saved is None:
                continue
            for receipt in saved[0].get("acquired_indexes", ()):
                key = receipt["key"]
                if key in confirmed and confirmed[key] != receipt:
                    raise ValueError("generic_definition_public_index_conflict")
                confirmed[key] = receipt
    for key, receipt in sources.index_receipts.items():
        if confirmed.get(key) != receipt:
            raise ValueError("generic_definition_public_index_not_confirmed")
    expected = tuple(dict.fromkeys((*policy.index_keys, *sources.index_receipts)))
    if sources.index_keys != expected:
        raise ValueError("generic_definition_public_index_pool_binding")


def accounting(campaign):
    public = wire_usage(campaign)
    return dict(
        schema_version="generic-public-definition-accounting/1.0.0",
        public_wire_reservations=public,
        agent_reservations=campaign.usage() - public,
        all_reservations=campaign.usage(),
        prior_reservations=campaign.protocol["previous"]["reservations"],
        unknown_is_charged=True,
        public_requests_are_model_calls=False,
    )
