"""No GPU/weights: complete loading rather than random initialization."""

from dataclasses import dataclass

import pytest

from Seg_Model.medical_sam_cup_repaired.checkpoint import complete_state, official_key


@dataclass
class TensorShape:
    shape: tuple = (2, 3)


def inputs():
    target = "mask_decoder.transformer.layers.0.mlp.layers.0.0.weight"
    trained = "image_encoder.blocks.0.Adapter.weight"
    return target, trained, {target: TensorShape(), trained: TensorShape()}


def test_official_key_is_mapped_and_adapter_preserved():
    target, trained, expected = inputs()
    base = {official_key(target): TensorShape()}
    adapter = {trained: TensorShape()}
    state, origins = complete_state(expected, base, adapter)
    assert state[target] is base[official_key(target)]
    assert state[trained] is adapter[trained]
    assert origins[target]["key"].endswith("lin1.weight")


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("suffix", ["weight", "bias"])
@pytest.mark.parametrize("current,official", [("layers.0.0", "lin1"), ("fc", "lin2")])
def test_all_eight_aliases(layer, suffix, current, official):
    prefix = f"mask_decoder.transformer.layers.{layer}.mlp."
    assert official_key(prefix + current + "." + suffix) == prefix + official + "." + suffix


@pytest.mark.parametrize("block", range(12))
@pytest.mark.parametrize("axis", ["h", "w"])
def test_all_relative_position_parameters(block, axis):
    prefix = f"image_encoder.blocks.{block}.attn."
    assert official_key(prefix + "rel_" + axis) == prefix + "rel_pos_" + axis


def test_missing_base_is_not_random_fallback():
    _, trained, expected = inputs()
    with pytest.raises(ValueError, match="^cup_uninitialized_parameter$"):
        complete_state(expected, {}, {trained: TensorShape()})


def test_bad_shape_rejected():
    target, trained, expected = inputs()
    with pytest.raises(ValueError, match="^cup_parameter_shape_mismatch$"):
        complete_state(
            expected, {official_key(target): TensorShape((3, 2))}, {trained: TensorShape()}
        )


def test_unknown_adapter_key_rejected():
    target, trained, expected = inputs()
    with pytest.raises(ValueError, match="^cup_unconsumed_adapter_parameters$"):
        complete_state(
            expected,
            {official_key(target): TensorShape()},
            {trained: TensorShape(), "unknown": TensorShape()},
        )


def test_task_trained_weight_not_overwritten_by_base():
    target, trained, expected = inputs()
    adapter = {trained: TensorShape(), target: TensorShape()}
    state, _ = complete_state(expected, {official_key(target): TensorShape()}, adapter)
    assert state[target] is adapter[target]


def test_initial_random_values_never_enter_composed_state():
    target, trained, expected = inputs()
    base, adapter = {official_key(target): TensorShape()}, {trained: TensorShape()}
    one, _ = complete_state(expected, base, adapter)
    two, _ = complete_state({k: TensorShape() for k in expected}, base, adapter)
    assert all(one[k] is two[k] for k in one)
