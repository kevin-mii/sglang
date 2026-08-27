"""Checkpoint-mmap host weights for distributed layerwise offload (DLO).

When DLO is enabled and the checkpoint is provably compatible, the DiT skips
ordinary weight materialization: parameters and persistent buffers are bound
directly to safetensors file-backed (mmap) CPU views. Every same-node rank maps
the same files, so the OS page cache keeps ~one physical host copy per node —
independent of DP/SP/CFG topology and with no process groups or synchronized
waves. This mirrors vLLM-Omni's loader-owned ``HostWeightPlan``
(``vllm_omni/diffusion/model_loader/host_weight_plan.py``).

The preflight is fail-closed: any required tensor without exactly one matching
checkpoint source, any shape/dtype mismatch (the ordinary loader would silently
cast), any rename/merge mapping, and any custom ``weight_loader`` that is
neither a standard parallel-linear loader (identity at tp=1) nor covered by a
model-declared deferred transform makes the whole plan incompatible, and the
ordinary loader runs instead.

Deferred transforms (e.g. MiniMax-H3's grouped-QKV reorder) are applied one
block at a time by the layerwise offload manager while staging or sharding, so
no private full-model copy is ever materialized. Tensors outside the streamed
block lists (e.g. H3's resident token_refiner) get their transforms applied
eagerly at bind time, materializing only those few tensors privately.

Mapped sources are immutable: the offload manager detaches a layer to private
writable host storage on the first writeback (LoRA merge / refit) and reports
that sharing was lost for it.
"""

from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable

import msgspec
import torch
from safetensors import safe_open

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_SAFETENSORS_DTYPES: Dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


class TensorBinding(msgspec.Struct, frozen=True):
    checkpoint_key: str
    file_path: str


class CheckpointMmapPlan(msgspec.Struct):
    """Loader-proved bindings for direct checkpoint-mmap weight backing.

    ``transforms`` holds only the DEFERRED transforms — those on streamed-block
    tensors, applied by the offload manager at staging/sharding time. Eagerly
    transformed (non-streamed) tensors are already in runtime layout after
    ``bind_checkpoint_mmap_views`` and do not appear here.
    """

    bindings: Dict[str, TensorBinding]
    transforms: Dict[str, Any]
    # Final-layout artifacts store post-load tensors under runtime names, so
    # binding must not re-apply model-declared checkpoint transforms.
    final_layout: bool = False


@runtime_checkable
class SupportsCheckpointMmapTransforms(Protocol):
    """Models whose checkpoint layout needs per-tensor runtime transforms."""

    def get_checkpoint_mmap_transforms(self) -> Dict[str, Any]: ...


class _PlanIncompatible(Exception):
    pass


def _required_tensors(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """All parameters plus persistent buffers — what the checkpoint must cover."""
    required = dict(model.named_parameters())
    persistent = {}
    for module_name, module in model.named_modules():
        for buffer_name, buffer in module._buffers.items():
            if buffer is None or buffer_name in module._non_persistent_buffers_set:
                continue
            full_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
            persistent[full_name] = buffer
    required.update(persistent)
    return required


def _param_owner_map(model: torch.nn.Module) -> Dict[int, torch.nn.Module]:
    owners: Dict[int, torch.nn.Module] = {}
    for module in model.modules():
        for param in module._parameters.values():
            if param is not None:
                owners[id(param)] = module
    return owners


def _streamed_name_prefixes(model: torch.nn.Module) -> Tuple[str, ...]:
    """Block-list prefixes whose tensors the offload manager will stream."""
    from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
        LayerwiseOffloadableModuleMixin,
    )

    if not isinstance(model, LayerwiseOffloadableModuleMixin):
        return ()
    return tuple(f"{layer_name}." for layer_name in model.layer_names)


def _is_streamed_name(name: str, streamed_prefixes: Tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in streamed_prefixes)


def _check_loader_policy(
    *,
    name: str,
    param: torch.Tensor,
    owners: Dict[int, torch.nn.Module],
    transforms: Dict[str, Any],
) -> None:
    """Reject tensors whose load path is not provably identity or adapted."""
    from sglang.multimodal_gen.runtime.loader.weight_utils import (
        default_weight_loader,
    )

    weight_loader = getattr(param, "weight_loader", None)
    if weight_loader is None or weight_loader is default_weight_loader:
        return
    if name in transforms:
        return
    # Standard parallel-linear loaders are bound methods of the owning module
    # and reduce to a full-size narrow + copy at tp=1, which the caller has
    # already verified. A custom closure (no matching __self__) needs a
    # model-declared transform instead.
    bound_owner = getattr(weight_loader, "__self__", None)
    if bound_owner is not None and bound_owner is owners.get(id(param)):
        return
    raise _PlanIncompatible(
        f"{name} has a custom weight_loader without a declared mmap transform"
    )


def build_checkpoint_mmap_plan(
    model: torch.nn.Module,
    safetensors_files: List[str],
    param_names_mapping_fn,
) -> Tuple[CheckpointMmapPlan | None, str | None]:
    """Preflight: prove the checkpoint can back the model via mmap views.

    The caller must already have gated topology and load-path features that
    rewrite weights (tp>1, FSDP/HSDP, quantization, bitsandbytes, post-load
    hooks, ``preprocess_loaded_state_dict``). Returns ``(plan, None)`` on
    success or ``(None, reason)`` for the ordinary-loader fallback.
    """
    try:
        return _build_plan(model, safetensors_files, param_names_mapping_fn), None
    except _PlanIncompatible as exc:
        return None, str(exc)


def _build_plan(
    model: torch.nn.Module,
    safetensors_files: List[str],
    param_names_mapping_fn,
) -> CheckpointMmapPlan:
    required = _required_tensors(model)
    owners = _param_owner_map(model)
    streamed_prefixes = _streamed_name_prefixes(model)

    declared_transforms: Dict[str, Any] = {}
    if isinstance(model, SupportsCheckpointMmapTransforms):
        declared_transforms = dict(model.get_checkpoint_mmap_transforms())

    bindings: Dict[str, TensorBinding] = {}
    for file_path in safetensors_files:
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for checkpoint_key in handle.keys():
                runtime_name, merge_index, _ = param_names_mapping_fn(checkpoint_key)
                if merge_index is not None:
                    raise _PlanIncompatible(
                        f"{checkpoint_key} participates in a merged-parameter "
                        "mapping, which cannot be file-backed"
                    )
                if runtime_name not in required:
                    continue
                if runtime_name in bindings:
                    raise _PlanIncompatible(
                        f"{runtime_name} matches more than one checkpoint tensor"
                    )
                target = required[runtime_name]
                tensor_slice = handle.get_slice(checkpoint_key)
                if tuple(tensor_slice.get_shape()) != tuple(target.shape):
                    raise _PlanIncompatible(
                        f"{runtime_name} shape mismatch: checkpoint "
                        f"{tuple(tensor_slice.get_shape())} vs runtime "
                        f"{tuple(target.shape)}"
                    )
                checkpoint_dtype = _SAFETENSORS_DTYPES.get(tensor_slice.get_dtype())
                if checkpoint_dtype is None or checkpoint_dtype != target.dtype:
                    raise _PlanIncompatible(
                        f"{runtime_name} dtype mismatch: checkpoint "
                        f"{tensor_slice.get_dtype()} vs runtime {target.dtype} "
                        "(the ordinary loader would cast)"
                    )
                bindings[runtime_name] = TensorBinding(
                    checkpoint_key=checkpoint_key, file_path=file_path
                )

    missing = [name for name in required if name not in bindings]
    if missing:
        raise _PlanIncompatible(
            f"{len(missing)} required tensors have no checkpoint source "
            f"(first 5: {missing[:5]})"
        )

    for name, param in required.items():
        _check_loader_policy(
            name=name, param=param, owners=owners, transforms=declared_transforms
        )

    deferred = {
        name: transform
        for name, transform in declared_transforms.items()
        if _is_streamed_name(name, streamed_prefixes)
    }
    return CheckpointMmapPlan(bindings=bindings, transforms=deferred)


def build_final_layout_plan(
    model: torch.nn.Module, artifact_files: List[str]
) -> Tuple[CheckpointMmapPlan | None, str | None]:
    """Preflight a final-layout artifact: runtime names, post-load tensors.

    Loader-policy and transform checks do not apply — the artifact was
    produced from a fully finalized model, so a matching name IS the final
    tensor. Dtypes are taken from the artifact rather than checked against the
    meta-built model: ``post_load_weights`` dtype adjustments already happened
    before publication, and re-running them on bound views is an idempotent
    no-op.
    """
    required = _required_tensors(model)
    bindings: Dict[str, TensorBinding] = {}
    for file_path in artifact_files:
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name not in required:
                    continue
                if name in bindings:
                    return None, f"{name} appears in more than one artifact shard"
                shape = tuple(handle.get_slice(name).get_shape())
                if shape != tuple(required[name].shape):
                    return None, (
                        f"{name} shape mismatch: artifact {shape} vs runtime "
                        f"{tuple(required[name].shape)}"
                    )
                bindings[name] = TensorBinding(checkpoint_key=name, file_path=file_path)
    missing = [name for name in required if name not in bindings]
    if missing:
        return None, (
            f"{len(missing)} required tensors missing from the artifact "
            f"(first 5: {missing[:5]})"
        )
    return (
        CheckpointMmapPlan(bindings=bindings, transforms={}, final_layout=True),
        None,
    )


def bind_checkpoint_mmap_views(
    model: torch.nn.Module, plan: CheckpointMmapPlan
) -> None:
    """Bind every planned tensor to its file-backed view (meta -> mmap).

    File handles are retained on the model so the mappings stay alive for the
    process lifetime. Non-streamed tensors with declared transforms are
    materialized in runtime layout here (private, but only those few tensors);
    streamed tensors keep their raw views — the offload manager applies
    ``plan.transforms`` at staging/sharding time.
    """
    from safetensors import safe_open as _safe_open

    declared_transforms: Dict[str, Any] = {}
    if not plan.final_layout and isinstance(model, SupportsCheckpointMmapTransforms):
        declared_transforms = dict(model.get_checkpoint_mmap_transforms())

    file_handles: Dict[str, Any] = {}
    bound = 0
    for runtime_name, binding in plan.bindings.items():
        parent_path, _, leaf_name = runtime_name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        if binding.file_path not in file_handles:
            file_handles[binding.file_path] = _safe_open(
                binding.file_path, framework="pt", device="cpu"
            )
        tensor = file_handles[binding.file_path].get_tensor(binding.checkpoint_key)

        transform = declared_transforms.get(runtime_name)
        if transform is not None and runtime_name not in plan.transforms:
            # Not streamed by the offload manager -> runtime layout now.
            transformed = transform(tensor)
            if transformed.dtype != tensor.dtype or transformed.shape != tensor.shape:
                raise ValueError(
                    f"mmap transform changed tensor metadata for {runtime_name}: "
                    f"{tensor.dtype}/{tuple(tensor.shape)} -> "
                    f"{transformed.dtype}/{tuple(transformed.shape)}"
                )
            tensor = transformed

        old_param = parent._parameters.get(leaf_name)
        if old_param is not None:
            new_param = torch.nn.Parameter(tensor, requires_grad=False)
            # Preserve loader/runtime attributes (weight_loader, output_dim, ...).
            new_param.__dict__.update(old_param.__dict__)
            parent._parameters[leaf_name] = new_param
        else:
            parent._buffers[leaf_name] = tensor
        bound += 1

    model._checkpoint_mmap_file_handles = file_handles
    logger.info(
        "Bound %d tensors to checkpoint-mmap views across %d file(s); host "
        "weight pages are node-shared via the OS page cache.",
        bound,
        len(file_handles),
    )
