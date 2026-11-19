"""Opt-in request dependencies; never a cache of current tool qualification.

Vision reads a selected image and definition, not every saved tool measurement.
Review and adopted definitions remain conservative full consumers. The same
selection is used by projection and all request guards.
"""

POLICY = "target-dependencies/1.0.0"


def enabled(base):
    policy = getattr(base, "acquisition_scope_policy", None)
    if policy not in {None, POLICY}:
        raise ValueError("multimodal_acquisition_scope_policy")
    return policy == POLICY


def resources(base, target):
    from .input_scope import current_entry, is_target

    if is_target(target):
        current_entry(base, target)
        return (base.resources[0],)  # Original input, no tool measurement consumed.
    if not enabled(base) or target is None:
        return base.resources
    rows = [r for r in base.ledger if r["id"] == target]
    if len(rows) != 1:
        raise ValueError("multimodal_acquisition_scope_target")
    row = rows[0]
    component = getattr(base, "component_observations", None)
    if (
        row.get("acquisition_method", "vision") != "vision"
        or row.get("definition_receipt_key")
        or (component is not None and target == component.target)
    ):
        return base.resources
    by_code = {r.code: r for r in base.resources}
    if not base.resources or len(by_code) != len(base.resources):
        raise ValueError("multimodal_acquisition_scope_resources")
    root = base.resources[0].code
    paths = row.get("dependency_paths", ((root,),))
    if not paths or any(not path for path in paths):
        raise ValueError("multimodal_acquisition_scope_dependencies")
    used = {root, *(code for path in paths for code in path)}
    quality = row.get("quality_source")
    if quality is not None:
        if by_code.get(quality.code) != quality:
            raise ValueError("multimodal_acquisition_scope_quality")
        used.add(quality.code)
    if not used <= by_code.keys():
        raise ValueError("multimodal_acquisition_scope_unknown_dependency")
    return tuple(r for r in base.resources if r.code in used)


def check(base, target=None):
    from .input_scope import current_entry, is_target

    if is_target(target):
        base.check_definition_materials()
        current_entry(base, target)
        base.view(target)
        return
    if not enabled(base) or target is None:
        return base.check()
    # Input, current medical sources, accepted definitions and local projection
    # integrity always apply. This is not permission to consume the saved tool.
    base.check_definition_materials()
    selected = resources(base, target)
    component = getattr(base, "component_observations", None)
    if component is not None and component.resource() in selected:
        component.check()
