"""Complete checkpoint composition; no silent random-parameter fallback.

Does not import a model framework. The runner validates tensor contents.
"""


def official_key(key):
    for block in range(12):
        prefix = f"image_encoder.blocks.{block}.attn."
        if key in (prefix + "rel_h", prefix + "rel_w"):
            return prefix + "rel_pos_" + key[-1]
    for layer in range(2):
        prefix = f"mask_decoder.transformer.layers.{layer}.mlp."
        for target, original in (("layers.0.0.", "lin1."), ("fc.", "lin2.")):
            if key.startswith(prefix + target):
                return prefix + original + key[len(prefix + target) :]
    return key


def complete_state(expected, base, adapter):
    if not adapter or not any("Adapter" in key for key in adapter):
        raise ValueError("cup_adapter_parameters_missing")
    state, origins, used_adapter = {}, {}, set()
    for key, template in expected.items():
        alias = official_key(key)
        candidates = [
            ("adapter", adapter, key),
            ("adapter", adapter, alias),
            ("base", base, key),
            ("base", base, alias),
        ]
        match = next(
            ((owner, values, name) for owner, values, name in candidates if name in values), None
        )
        if match is None:
            raise ValueError("cup_uninitialized_parameter")
        owner, values, name = match
        if values[name].shape != template.shape:
            raise ValueError("cup_parameter_shape_mismatch")
        state[key] = values[name]
        origins[key] = {"checkpoint": owner, "key": name}
        if owner == "adapter":
            used_adapter.add(name)
    if set(adapter) != used_adapter:
        raise ValueError("cup_unconsumed_adapter_parameters")
    return state, origins
