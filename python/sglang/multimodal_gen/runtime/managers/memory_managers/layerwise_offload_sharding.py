"""Sharded host storage and bounded device buffers for distributed layerwise offload.

Used by :class:`LayerwiseOffloadManager` when ``--enable-distributed-layerwise-offload``
is set with AllGather enabled (the default): each rank keeps only ``1/world`` of every
offloaded layer's flat pinned host buffer, prefetch copies the local shard H2D on the
manager's copy stream, and the full flat layer buffer is reconstructed with
``all_gather_into_tensor`` on a dedicated comm stream.

The weight-shard layout mirrors vLLM-Omni's distributed layerwise offload
(``vllm_omni/diffusion/offloader/distributed_layerwise_backend.py``): equal-size
shards padded for the AllGather contract, with parameter metadata offsets kept
relative to the FULL flat buffer so the existing 32-byte alignment contract and
parameter rebinding logic are unchanged. Shards are additionally rounded up to the
32-byte alignment so every shard boundary stays aligned.
"""

from typing import Dict, List

import torch
import torch.distributed

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def resolve_sp_shard_group():
    """Return the sequence-parallel :class:`GroupCoordinator` for weight sharding.

    Sharding must only span ranks holding identical weight replicas. With the
    ``tp-sp-pp-cfg-dp`` rank layout, each SP group varies only the sp coordinate
    at fixed tp/cfg/dp coordinates, so every rank in it holds the same (possibly
    TP-sharded) replica and executes the same request in lockstep. TP is
    intentionally excluded (TP shards differ per rank), matching vLLM-Omni.

    Returns ``None`` when distributed or model parallel is not initialized, or
    when the SP group has a single rank — callers then fall back to rank-local
    layerwise offload.
    """
    if not torch.distributed.is_initialized():
        return None
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_sp_group,
        model_parallel_is_initialized,
    )

    if not model_parallel_is_initialized():
        return None
    sp_group = get_sp_group()
    if sp_group.world_size <= 1:
        return None
    return sp_group


def shard_numels(
    *,
    total_numel: int,
    world_size: int,
    dtype: torch.dtype,
    alignment_bytes: int = 32,
) -> tuple[int, int]:
    """Return ``(shard_numel, padded_numel)`` for one flat (layer, dtype) buffer.

    ``padded_numel = shard_numel * world_size`` satisfies the
    ``all_gather_into_tensor`` contract (output numel == world * input numel),
    and ``shard_numel`` is rounded up to the 32-byte alignment so every shard
    boundary inside the reconstructed buffer stays aligned.
    """
    element_size = torch.empty((), dtype=dtype).element_size()
    alignment_numel = max(1, alignment_bytes // element_size)
    shard = (total_numel + world_size - 1) // world_size
    remainder = shard % alignment_numel
    if remainder:
        shard += alignment_numel - remainder
    return shard, shard * world_size


def copy_overlap_into_shard(
    *,
    shard: torch.Tensor,
    shard_start: int,
    offset: int,
    flat_source: torch.Tensor,
) -> None:
    """Copy the part of one flattened weight that falls inside this rank's shard.

    ``offset`` is the weight's start inside the FULL flat buffer; the shard
    covers ``[shard_start, shard_start + shard.numel())`` of that buffer.
    """
    overlap_start = max(offset, shard_start)
    overlap_end = min(offset + flat_source.numel(), shard_start + shard.numel())
    if overlap_start >= overlap_end:
        return
    shard[overlap_start - shard_start : overlap_end - shard_start].copy_(
        flat_source[overlap_start - offset : overlap_end - offset]
    )


def gather_full_cpu_buffer(
    *,
    shard: torch.Tensor,
    total_numel: int,
    cpu_group,
    world_size: int,
) -> torch.Tensor:
    """Reconstruct one full flat CPU buffer from per-rank shards over gloo.

    Collective: every rank in the shard group must call this together. Shards
    are exchanged as uint8 views so gloo dtype support (e.g. for fp8) is not a
    constraint.
    """
    byte_shard = shard.contiguous().view(torch.uint8)
    gathered = [torch.empty_like(byte_shard) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, byte_shard, group=cpu_group)
    return torch.cat(gathered).view(shard.dtype)[:total_numel]


def allocate_device_slot_buffers(
    *,
    max_numel_by_dtype: Dict[torch.dtype, int],
    num_slots: int,
    device: torch.device,
) -> List[Dict[torch.dtype, torch.Tensor]]:
    """Allocate ``num_slots`` persistent flat device buffers per dtype.

    All layers of a manager share these slots (each AllGather slices a prefix
    down to the layer's exact size), so device weight residency is bounded by
    ``num_slots`` layers and the buffer addresses stay stable across the
    thousands of collectives in a request — letting NCCL reuse its registered
    buffers. This is vLLM-Omni's ``_allocate_shared_buffers`` /
    ``_allocate_shared_shard_buffers`` scheme; at the default prefetch depth 1
    (two slots) it is exactly its double-buffer layout.
    """
    slots = [
        {
            dtype: torch.empty(numel, dtype=dtype, device=device)
            for dtype, numel in max_numel_by_dtype.items()
        }
        for _ in range(num_slots)
    ]
    total_mb = sum(
        buffer.numel() * buffer.element_size()
        for slot in slots
        for buffer in slot.values()
    ) / (1024 * 1024)
    logger.info(
        "Allocated %d shared device buffer slot(s) for distributed layerwise "
        "offload (%.0f MB total)",
        num_slots,
        total_mb,
    )
    return slots
