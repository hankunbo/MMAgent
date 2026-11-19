"""Report projections of accepted facts; no medical inference or graph mutations."""

from packages.graph_engine import accepted_delta_event_from_payload, canonical_checksum

REPORT_VERSION = "deterministic-case-report/2.0.0"


def accepted_validations(graph, events, audit_events, read_snapshot):
    """Join the saved bound input to its exact accepted 03 event and Runtime snapshots.

    An older source version is legitimate when its accepted result remains current.
    A later status transition on that target supersedes it, including a reset back to
    the same eventual status. Unaccepted/stale bound inputs stay in trace only.
    """
    if graph is None:
        return []
    scope = (graph.case_code, graph.scope.run_id, graph.graph_code)
    snapshots = {graph.graph_version: graph}

    def snapshot(version):
        if version not in snapshots:
            snapshots[version] = read_snapshot(graph.graph_code, version)
        return snapshots[version]

    accepted, last_transition = {}, {}
    for event in sorted(audit_events, key=lambda e: e.event_sequence):
        if (
            (event.case_code, event.run_id, event.graph_code) != scope
            or event.event_type.value != "GRAPH_VERSION_APPENDED"
            or event.graph_version > graph.graph_version
        ):
            continue
        payload = event.payload
        if "accepted_delta_event" not in payload:
            continue
        commit = accepted_delta_event_from_payload(payload["accepted_delta_event"])
        if (
            commit.graph_code != graph.graph_code
            or event.graph_version != commit.result_version
            or canonical_checksum(snapshot(commit.base_version)) != commit.base_checksum
            or canonical_checksum(snapshot(commit.result_version)) != commit.result_checksum
        ):
            continue
        for operation in commit.delta.operations:
            raw = operation.model_dump(mode="json")
            if raw["op"] == "UPDATE_NODE_STATUS" and raw["from_status"] != raw["to_status"]:
                last_transition[raw["node_code"]] = event.event_sequence
        item = commit.validation_result
        if item is not None and (
            (item.scope.case_code, item.scope.run_id, item.scope.graph_code) == scope
            and payload.get("validation_checksum") == commit.validation_checksum
        ):
            accepted[commit.validation_checksum] = (event, commit)

    result, seen = [], set()
    for entry in events:
        if entry["kind"] != "validation":
            continue
        bound = entry["validation"]
        checksum = canonical_checksum(bound["item"])
        if checksum not in accepted or checksum in seen:
            continue
        event, commit = accepted[checksum]
        item = commit.validation_result
        source = dict(
            graph_code=commit.graph_code,
            graph_version=commit.base_version,
            graph_checksum=commit.base_checksum,
        )
        # Verify the existing envelope; never rebind or submit an old validation.
        envelope = dict(
            payload_version="03C_BOUND_INPUT_V1",
            kind="VALIDATION_RESULT",
            source_snapshot=source,
            item_checksum=checksum,
            item=bound["item"],
        )
        if (
            bound.get("payload_version") != envelope["payload_version"]
            or bound.get("source_snapshot") != source
            or bound.get("item_checksum") != checksum
            or bound.get("binding_checksum") != canonical_checksum(envelope)
            or last_transition.get(item.target_node_code) != event.event_sequence
        ):
            continue
        source_graph = snapshot(commit.base_version)
        node = next((n for n in graph.nodes if n.node_code == item.target_node_code), None)
        source_node = next(
            (n for n in source_graph.nodes if n.node_code == item.target_node_code), None
        )
        if (
            (source_graph.case_code, source_graph.scope.run_id, source_graph.graph_code) != scope
            or node is None
            or source_node is None
            or not set(item.evidence_bindings) <= set(node.evidence_bindings)
            or not set(item.evidence_bindings) <= set(source_node.evidence_bindings)
        ):
            continue
        seen.add(checksum)
        result.append(
            dict(
                item=item,
                reference=dict(
                    validation_code=item.validation_code,
                    validation_checksum=checksum,
                    binding_checksum=bound["binding_checksum"],
                    source_snapshot=source,
                    accepted_graph=dict(
                        graph_code=commit.graph_code,
                        graph_version=commit.result_version,
                        graph_checksum=commit.result_checksum,
                    ),
                    audit_event_id=event.event_id,
                    audit_event_sequence=event.event_sequence,
                    rule_refs=[r.model_dump(mode="json") for r in item.rule_refs],
                    evidence_bindings=list(item.evidence_bindings),
                ),
            )
        )
    return result


def observation_effects(graph, target, step, plan, validations, *, synthetic):
    """One target × one Artifact × accepted validation. No last-item aggregation."""
    eligible = (
        graph is not None
        and plan.document.purpose == "decisive_research"
        and step.result.status.value == "SUCCEEDED"
        and step.result.qc_status.value == "PASS"
        and (step.result.decisive_evidence or synthetic)
    )
    details = []
    for artifact in step.result.artifacts:
        matches = []
        if eligible and (artifact.scope.case_code, artifact.scope.run_id) == (
            graph.case_code,
            graph.scope.run_id,
        ):
            for value in validations:
                item = value["item"]
                if (
                    item.target_node_code == target
                    and (item.scope.case_code, item.scope.run_id, item.scope.graph_code)
                    == (graph.case_code, graph.scope.run_id, graph.graph_code)
                    and artifact.asset_code in item.evidence_bindings
                    and artifact.checksum in item.provenance.input_checksums
                    and canonical_checksum(step) in item.provenance.input_checksums
                    and item.qc_status.value == "PASS"
                    and item.tool_execution_status.value == "SUCCEEDED"
                ):
                    matches.append(
                        dict(**value["reference"], clinical_effect=item.clinical_effect.value)
                    )
        details.append(
            dict(
                target=target,
                asset_code=artifact.asset_code,
                artifact_checksum=artifact.checksum,
                validations=matches,
            )
        )
    effects = {v["clinical_effect"] for d in details for v in d["validations"]}
    if not effects:
        effect, status = "NONE", "unvalidated"
    elif len(details) == 1 and len(effects) == 1:
        effect, status = next(iter(effects)), "single_evidence_effect"
    else:
        # Even identical effects on several artifacts remain separate facts.
        effect, status = None, "per_evidence_only"
    return dict(clinical_effect=effect, clinical_effect_status=status, evidence_validations=details)
