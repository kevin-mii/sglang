"""Unit tests for the checkpoint-mmap host-weight preflight and binding.

The preflight must be fail-closed: any tensor whose ordinary load path is not
provably identity (renames/merges, shape or dtype casts, custom weight_loaders
without a declared transform) makes the whole plan incompatible.
"""

import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.loader.checkpoint_mmap import (
    bind_checkpoint_mmap_views,
    build_checkpoint_mmap_plan,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)


def _identity_mapping(checkpoint_key):
    return checkpoint_key, None, None


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(4, 6, dtype=torch.bfloat16), requires_grad=False
        )


class _MmapModel(torch.nn.Module, LayerwiseOffloadableModuleMixin):
    layer_names = ["blocks"]

    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(num_layers)])
        self.head = torch.nn.Parameter(
            torch.randn(3, dtype=torch.float32), requires_grad=False
        )
        self.register_buffer("scale", torch.ones(2, dtype=torch.float32))
        self.register_buffer(
            "cache", torch.zeros(2, dtype=torch.float32), persistent=False
        )


def _checkpoint_for(model, tmp_path, mutate=None):
    state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if name != "cache"  # non-persistent buffers are not in checkpoints
    }
    if mutate is not None:
        mutate(state)
    path = str(tmp_path / "weights.safetensors")
    save_file(state, path)
    return path


def test_plan_and_bind_happy_path(tmp_path):
    model = _MmapModel()
    expected = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    path = _checkpoint_for(model, tmp_path)

    plan, reason = build_checkpoint_mmap_plan(model, [path], _identity_mapping)
    assert reason is None
    # Params + persistent buffers are bound; the non-persistent buffer is not.
    assert set(plan.bindings) == {
        "blocks.0.weight",
        "blocks.1.weight",
        "head",
        "scale",
    }
    assert plan.transforms == {}

    bind_checkpoint_mmap_views(model, plan)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, expected[name]), name
    assert not model.blocks[0].weight.requires_grad
    assert model._checkpoint_mmap_file_handles


def test_plan_rejects_missing_tensor(tmp_path):
    model = _MmapModel()
    path = _checkpoint_for(model, tmp_path, mutate=lambda s: s.pop("head"))
    plan, reason = build_checkpoint_mmap_plan(model, [path], _identity_mapping)
    assert plan is None
    assert "no checkpoint source" in reason


def test_plan_rejects_dtype_mismatch(tmp_path):
    model = _MmapModel()

    def cast_head(state):
        state["head"] = state["head"].to(torch.bfloat16)

    path = _checkpoint_for(model, tmp_path, mutate=cast_head)
    plan, reason = build_checkpoint_mmap_plan(model, [path], _identity_mapping)
    assert plan is None
    assert "dtype mismatch" in reason


def test_plan_rejects_shape_mismatch(tmp_path):
    model = _MmapModel()

    def reshape_head(state):
        state["head"] = torch.randn(4, dtype=torch.float32)

    path = _checkpoint_for(model, tmp_path, mutate=reshape_head)
    plan, reason = build_checkpoint_mmap_plan(model, [path], _identity_mapping)
    assert plan is None
    assert "shape mismatch" in reason


def test_plan_rejects_merged_parameter_mappings(tmp_path):
    model = _MmapModel()
    path = _checkpoint_for(model, tmp_path)

    def merge_mapping(checkpoint_key):
        if checkpoint_key == "head":
            return "head", 0, 2
        return checkpoint_key, None, None

    plan, reason = build_checkpoint_mmap_plan(model, [path], merge_mapping)
    assert plan is None
    assert "merged-parameter" in reason


def test_plan_rejects_custom_loader_without_transform(tmp_path):
    model = _MmapModel()

    def custom_loader(param, loaded):  # a closure, not a bound module method
        param.copy_(loaded.flip(0))

    model.blocks[0].weight.weight_loader = custom_loader
    path = _checkpoint_for(model, tmp_path)
    plan, reason = build_checkpoint_mmap_plan(model, [path], _identity_mapping)
    assert plan is None
    assert "custom weight_loader" in reason


def test_declared_transforms_defer_for_streamed_and_bind_eagerly_otherwise(
    tmp_path,
):
    class _TransformModel(_MmapModel):
        def get_checkpoint_mmap_transforms(self):
            return {
                "blocks.0.weight": lambda t: t + 1.0,  # streamed -> deferred
                "head": lambda t: t * 2.0,  # resident -> eager at bind
            }

    model = _TransformModel()
    original_block = model.blocks[0].weight.detach().clone()
    original_head = model.head.detach().clone()

    def custom_loader(param, loaded):
        param.copy_(loaded)

    # Custom loaders are acceptable when covered by a declared transform.
    model.blocks[0].weight.weight_loader = custom_loader
    model.head.weight_loader = custom_loader

    path = _checkpoint_for(model, tmp_path)
    plan, reason = build_checkpoint_mmap_plan(model, [path], _identity_mapping)
    assert reason is None
    assert set(plan.transforms) == {"blocks.0.weight"}

    bind_checkpoint_mmap_views(model, plan)
    # Streamed tensor keeps RAW checkpoint bytes (transform deferred to the
    # offload manager); the resident tensor is transformed eagerly.
    assert torch.equal(model.blocks[0].weight.data, original_block)
    assert torch.equal(model.head.data, original_head * 2.0)
