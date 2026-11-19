"""Pure graph projection and joint admissibility, before creating any live run.

The registered graph owns identities. Policy nodes are historical local keys;
the root uses its actual graph code. Neither is inferred from an LLM reply.
"""

from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.review_tasks import project_review_tasks, review_payload
from packages.graph_engine.research_candidates import validate_research_target
from packages.grounding.models import canonical, digest, validate
from packages.knowledge_claims.research_candidates import validate_source_target


def graph_target_binding(view, policy, graph):
    nodes = view.get("nodes", [])
    if not nodes or any(
        not isinstance(n, dict) or not n.get("key") or not n.get("node_code") for n in nodes
    ):
        raise ValueError("case_graph_node_inventory_required")
    by_key = {n["key"]: n for n in nodes}
    by_code = {n["node_code"]: n for n in nodes}
    if len(by_key) != len(nodes) or len(by_code) != len(nodes):
        raise ValueError("case_graph_identity_ambiguous")
    if any(code in by_key and by_key[code] != n for code, n in by_code.items()):
        raise ValueError("case_graph_identity_ambiguous")
    roots = [n for n in nodes if n.get("parent") is None]
    if len(roots) != 1 or roots[0].get("node_type") != "GOAL":
        raise ValueError("case_graph_unique_goal_required")
    root = roots[0]
    keys = sorted({i.node for i in policy.items})
    for key in keys:
        if key not in by_key:
            raise ValueError("case_graph_policy_node_missing")
        seen, current = set(), key
        while current != root["key"]:
            if current in seen or current not in by_key:
                raise ValueError("case_graph_node_not_rooted")
            seen.add(current)
            current = by_key[current].get("parent")
    return dict(
        schema_version="case-graph-target-binding/1.0.0",
        graph=graph,
        root_key=root["key"],
        root_code=root["node_code"],
        policy_nodes=[dict(key=k, node_code=by_key[k]["node_code"]) for k in keys],
        node_inventory_checksum=digest(
            [{k: n.get(k) for k in ("key", "node_code", "node_type", "parent")} for n in nodes]
        ),
    )


def check_case_admissibility(state, config):
    """Reuse original owner gates; no candidate, medical answer, IO or permissions."""
    state = validate(DomainState, state)
    material = state.material
    validate_source_target(material.goal_code, material)
    validate_research_target(material.goal_code, scope=material.scope, node_codes=state.nodes)
    if state.config_checksum != digest(config):
        raise ValueError("case_preflight_config_binding")
    roles = {r.role: r for r in config.roles}
    if not {"source", "vision", "review"} <= roles.keys():
        raise ValueError("case_preflight_roles_required")
    if state.prompt_checksum != roles["review"].prompt_checksum:
        raise ValueError("case_preflight_prompt_binding")
    # Current single-case Vision acceptance requires exactly one explicitly bound image slot.
    if len([o for o in material.observations if o.image is not None]) != 1:
        raise ValueError("case_preflight_vision_slot")
    tasks = project_review_tasks(state)
    policy = getattr(config, "review_batching", None)
    if policy:
        from packages.evidence_details.review_batches import batch_payload, make_plan

        plan = make_plan(state, policy, key="preflight-only")
        payloads = [batch_payload(state, plan, batch) for batch in plan.batches]
    else:
        payloads = [review_payload(material, tasks, "A")]
    for role in ("source", "vision"):
        if len(canonical(dict(role=role, material=material))) > roles[role].max_input_bytes:
            raise ValueError("case_preflight_input_bound:" + role)
    if max(len(canonical(p)) for p in payloads) > roles["review"].max_input_bytes:
        raise ValueError("live_review_input_bound")
    return dict(
        schema_version="case-joint-admissibility/1.0.0",
        state_checksum=digest(state),
        goal_code=material.goal_code,
        review_tasks=len(tasks.tasks),
        review_batches=len(payloads),
        creation_authorized=False,
        network_requests=0,
    )
