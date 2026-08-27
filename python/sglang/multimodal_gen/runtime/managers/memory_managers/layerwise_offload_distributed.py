"""Distributed layerwise offload (DLO) manager.

The counterpart of vLLM-Omni's ``distributed_layerwise_backend.py``, built as a
subclass of SGLang's :class:`LayerwiseOffloadManager` so the hook schedule,
residency, metadata, and alignment contracts are inherited unchanged.

Two transfer modes (``--enable-distributed-layerwise-offload``):

* **AllGather** (default): each rank persists only ``1/world`` of every
  offloaded layer's flat pinned host buffer; prefetch H2D's the local shard on
  ``copy_stream`` and reconstructs the full flat buffer with
  ``all_gather_into_tensor`` on a dedicated ``comm_stream`` — PCIe traffic and
  pinned host memory drop ``world``-fold, NVLink carries the rest. The sharding
  group is the sequence-parallel group (identical replicas at a fixed tp
  coordinate, lockstep denoise schedule), so no scheduler coupling is needed.
* **Rank-local** (``--dlo-no-use-allgather``): complete layers stream with H2D
  only — any DP/SP topology, no collectives.

Both modes stream through ``max(2, 2*prefetch_size)`` persistent shared device
slots sized to the largest layer (vLLM-Omni's double-buffer scheme generalized
to SGLang's burst lookahead): deterministic HBM, no per-prefetch allocation,
and stable addresses for NCCL buffer reuse. Slots are assigned in fetch order,
which stays correct for any layer count including the wraparound prefetch.

With a loader-proved :class:`CheckpointMmapPlan`, host weights are immutable
safetensors mmap views shared across same-node ranks via the OS page cache
(~one physical host copy per node). Rank-local mode packs them into bounded
pinned staging slots at prefetch time, applying deferred checkpoint->runtime
transforms (e.g. MiniMax-H3's grouped-QKV reorder) one layer at a time; a
writeback (LoRA merge / refit) detaches the affected layer to private writable
host storage and logs that sharing was lost for it.
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.distributed

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadManager,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_sharding import (
    allocate_device_slot_buffers,
    copy_overlap_into_shard,
    gather_full_cpu_buffer,
    shard_numels,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class DistributedLayerwiseOffloadManager(LayerwiseOffloadManager):
    """Layerwise offload with sharded-AllGather or rank-local DLO transfers."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        shard_group=None,
        mmap_plan=None,
        **base_kwargs,
    ) -> None:
        # DLO state must exist before the base constructor runs _initialize().
        if shard_group is not None and shard_group.world_size <= 1:
            shard_group = None
        self._shard_group = shard_group
        self._shard_world = shard_group.world_size if shard_group is not None else 1
        self._shard_rank = (
            shard_group.rank_in_group if shard_group is not None else 0
        )
        self.comm_stream = None
        self._mmap_plan = mmap_plan
        self._mmap_transforms: Dict[str, Any] = (
            dict(mmap_plan.transforms) if mmap_plan is not None else {}
        )
        # layer_idx -> {dtype: [(name, mmap_view, transform), ...]} in metadata
        # order — the rank-local checkpoint-mmap host backing.
        self._cpu_sources: Dict[int, Dict[torch.dtype, List[Tuple]]] = {}
        # layer_idx -> {dtype: (total_numel, shard_numel, padded_numel)};
        # rank-local layers store (total, total, total).
        self._flat_sizes: Dict[int, Dict[torch.dtype, Tuple[int, int, int]]] = {}
        self._shared_out_buffers: List[Dict[torch.dtype, torch.Tensor]] | None = None
        self._shared_shard_buffers: List[Dict[torch.dtype, torch.Tensor]] | None = (
            None
        )
        # Bounded pinned host slots for mmap -> device staging (rank-local).
        self._staging_buffers: List[Dict[torch.dtype, torch.Tensor]] | None = None
        self._staging_events: List[Any] = []
        # Layers whose mmap pages are cudaHostRegister'ed read-only: prefetch
        # DMAs the views straight into the device slots, skipping the per-step
        # staging pack. Registered for the process lifetime.
        self._direct_h2d_layers: set = set()
        # Monotone counter assigning device slots in fetch order, so the slot a
        # prefetch reuses always belonged to a layer whose forward was already
        # enqueued — correct for any num_layers (including wraparound).
        self._ring_fetch_counter = 0
        super().__init__(model, **base_kwargs)

    # ------------------------------------------------------------------ #
    #  Storage (overrides of the base template methods)                   #
    # ------------------------------------------------------------------ #

    def _init_transfer_resources(self) -> None:
        super()._init_transfer_resources()
        if self.copy_stream is None:
            raise ValueError(
                "Distributed layerwise offload requires an asynchronous copy "
                "stream; it is not supported on this platform."
            )
        if self._shard_group is not None:
            self.comm_stream = torch.get_device_module().Stream()
        # The shared device slots hold at most 2*prefetch_size in-flight
        # layers; deeper prefetch than half the stack cannot help anyway.
        self.prefetch_size = min(self.prefetch_size, max(1, self.num_layers // 2))

    def _plan_layer_hosting(self, layer_groups: Dict) -> Dict[int, str]:
        """DLO owns its file-backed storage; never leave layers on the mapping.

        The budget's pinned/pageable decisions still apply — AllGather shards
        are 1/world of the plan's sizes, so they always fit when the full
        copies would have.
        """
        hosting = super()._plan_layer_hosting(layer_groups)
        return {
            layer_idx: ("pageable" if kind == "mapped" else kind)
            for layer_idx, kind in hosting.items()
        }

    def _validate_storage_support(self) -> None:
        if self._has_dtensor_weights:
            raise ValueError(
                "Distributed layerwise offload does not support DTensor "
                "(FSDP/HSDP) weights: the offloader would shard or share "
                "already-sharded parameters. Disable "
                "--enable-distributed-layerwise-offload or FSDP inference."
            )

    def _store_layer_group(
        self,
        *,
        layer_idx: int,
        dtype: torch.dtype,
        contiguous_weights: List[Tuple[str, torch.Tensor, torch.Tensor]],
        aligned_offsets: Dict[str, int],
        total_numel: int,
        pin_memory: bool,
    ) -> None:
        if self._shard_group is not None:
            self._shard_and_pin(
                layer_idx=layer_idx,
                dtype=dtype,
                contiguous_weights=contiguous_weights,
                aligned_offsets=aligned_offsets,
                total_numel=total_numel,
                pin_memory=pin_memory,
            )
            return

        self._flat_sizes.setdefault(layer_idx, {})[dtype] = (
            total_numel,
            total_numel,
            total_numel,
        )
        if self._mmap_plan is not None:
            # Retain the immutable file-backed views; they are packed into the
            # bounded pinned staging slots at prefetch time. Capture the
            # underlying tensors (`.data`) — the parameters themselves are
            # rebound to placeholders right after this.
            self._cpu_sources.setdefault(layer_idx, {})[dtype] = [
                (name, local_weight.data, self._mmap_transforms.get(name))
                for name, _, local_weight in contiguous_weights
            ]
            return
        super()._store_layer_group(
            layer_idx=layer_idx,
            dtype=dtype,
            contiguous_weights=contiguous_weights,
            aligned_offsets=aligned_offsets,
            total_numel=total_numel,
            pin_memory=pin_memory,
        )

    def _shard_and_pin(
        self,
        *,
        layer_idx: int,
        dtype: torch.dtype,
        contiguous_weights: List[Tuple[str, torch.Tensor, torch.Tensor]],
        aligned_offsets: Dict[str, int],
        total_numel: int,
        pin_memory: bool,
    ) -> None:
        """Persist only this rank's 1/world shard of the flat buffer (pinned).

        Never materializes the full flat host buffer; deferred mmap transforms
        are applied one tensor at a time while copying the overlap.
        """
        shard_numel, padded_numel = shard_numels(
            total_numel=total_numel, world_size=self._shard_world, dtype=dtype
        )
        self._flat_sizes.setdefault(layer_idx, {})[dtype] = (
            total_numel,
            shard_numel,
            padded_numel,
        )
        shard = torch.zeros(shard_numel, dtype=dtype, pin_memory=pin_memory)
        shard_start = self._shard_rank * shard_numel
        for name, _, local_weight in contiguous_weights:
            copy_overlap_into_shard(
                shard=shard,
                shard_start=shard_start,
                offset=aligned_offsets[name],
                flat_source=self._apply_mmap_transform(name, local_weight).flatten(),
            )
        self._consolidated_cpu_weights[layer_idx][dtype] = shard

    def _apply_mmap_transform(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Apply a deferred checkpoint->runtime transform (identity if none)."""
        transform = self._mmap_transforms.get(name)
        if transform is None:
            return tensor
        transformed = transform(tensor)
        if transformed.dtype != tensor.dtype or transformed.shape != tensor.shape:
            raise ValueError(
                f"mmap transform changed tensor metadata for {name}: "
                f"{tensor.dtype}/{tuple(tensor.shape)} -> "
                f"{transformed.dtype}/{tuple(transformed.shape)}"
            )
        return transformed

    def _has_layer_storage(self, layer_idx: int) -> bool:
        return (
            layer_idx in self._consolidated_cpu_weights
            or layer_idx in self._cpu_sources
        )

    def _post_initialize_storage(self) -> None:
        if self._shard_group is None and self._cpu_sources:
            self._register_mmap_sources_for_direct_h2d()
        self._allocate_shared_buffers()
        if self._shard_group is not None:
            self._warn_if_sharding_ineffective()
            if not self._cpu_sources:
                # AllGather mode consumed the checkpoint-mmap views only as the
                # source for each rank's pinned shard; dropping the loader's
                # handle anchor lets files with no remaining views unmap.
                # Tensors still bound to views (other managers' rank-local
                # sources, non-streamed params) hold their own buffer refs.
                self.model.__dict__.pop("_checkpoint_mmap_file_handles", None)

    def _allocate_shared_buffers(self) -> None:
        """Allocate the persistent device slots (and host staging) once.

        Sized to the largest layer across the stack, so device weight residency
        is bounded by the slot count regardless of num_layers, and the stable
        addresses let NCCL reuse registered buffers across per-layer AllGathers.
        """
        max_padded: Dict[torch.dtype, int] = {}
        max_shard: Dict[torch.dtype, int] = {}
        max_staging: Dict[torch.dtype, int] = {}
        for layer_idx, layer_sizes in self._flat_sizes.items():
            for dtype, (total_numel, shard_numel, padded_numel) in layer_sizes.items():
                max_padded[dtype] = max(max_padded.get(dtype, 0), padded_numel)
                max_shard[dtype] = max(max_shard.get(dtype, 0), shard_numel)
                if (
                    layer_idx in self._cpu_sources
                    and layer_idx not in self._direct_h2d_layers
                ):
                    max_staging[dtype] = max(max_staging.get(dtype, 0), total_numel)
        if not max_padded:
            return
        num_slots = max(2, 2 * self.prefetch_size)
        self._shared_out_buffers = allocate_device_slot_buffers(
            max_numel_by_dtype=max_padded, num_slots=num_slots, device=self.device
        )
        if self._shard_group is not None:
            # AllGather inputs; rank-local mode copies H2D straight into the
            # output slots and needs no separate shard slots.
            self._shared_shard_buffers = allocate_device_slot_buffers(
                max_numel_by_dtype=max_shard, num_slots=num_slots, device=self.device
            )
        if max_staging:
            self._staging_buffers = [
                {
                    dtype: torch.empty(
                        numel, dtype=dtype, pin_memory=self.pin_cpu_memory
                    )
                    for dtype, numel in max_staging.items()
                }
                for _ in range(num_slots)
            ]
            self._staging_events = [None] * num_slots

    def _warn_if_sharding_ineffective(self) -> None:
        """Warn when most bytes are stride-preserving and stay rank-local.

        Non-contiguous (e.g. ModelOpt FP8 CUTLASS) weights are kept as private
        full copies per rank, so a checkpoint dominated by them gains little
        from AllGather sharding.
        """
        strided_bytes = sum(
            tensor.numel() * tensor.element_size()
            for per_layer in self._strided_cpu_weights.values()
            for tensor in per_layer.values()
        )
        flat_bytes = sum(
            total_numel * torch.empty((), dtype=dtype).element_size()
            for layer_sizes in self._flat_sizes.values()
            for dtype, (total_numel, _, _) in layer_sizes.items()
        )
        total_bytes = strided_bytes + flat_bytes
        if total_bytes and strided_bytes > 0.1 * total_bytes:
            logger.warning(
                "Distributed layerwise offload: %.0f%% of offloaded bytes use "
                "stride-preserving layouts that stay rank-local, so weight "
                "sharding saves little host memory or H2D bandwidth for this "
                "checkpoint. Consider --dlo-no-use-allgather.",
                100.0 * strided_bytes / total_bytes,
            )

    # ------------------------------------------------------------------ #
    #  Transfer                                                            #
    # ------------------------------------------------------------------ #

    def _wait_transfer_streams(self) -> None:
        super()._wait_transfer_streams()
        if self.comm_stream is not None:
            torch.get_device_module().current_stream().wait_stream(self.comm_stream)

    def _register_mmap_sources_for_direct_h2d(self) -> None:
        """cudaHostRegister the mmap views so prefetch can DMA them directly.

        Registration is read-only (the mappings are immutable checkpoint
        pages) and all-or-nothing: any failure rolls back and every layer
        keeps the pinned-staging path. Only transform-free layers qualify —
        deferred transforms must run on the CPU at stage time. The registered
        bytes are pinned page cache, so they are charged to the host pin
        budget like any other pinned hosting.
        """
        if (
            not self.pin_cpu_memory
            or not torch.cuda.is_available()
            or self._mmap_plan is None
            or not getattr(self._mmap_plan, "final_layout", False)
        ):
            # Raw-checkpoint plans interleave views with eagerly-transformed
            # private tensors in the same pages; page-granular registration
            # would leave tensors partially registered and later copies fail.
            # Final-layout artifacts bind every tensor as a view, so whole
            # per-file spans are safely coverable.
            return
        # The plan (and therefore the registered spans) covers the whole
        # model; the first manager registers and later managers (other layer
        # groups of the same DiT) adopt, since re-registering overlapping
        # pages fails.
        if self.model.__dict__.get("_dlo_direct_h2d_registered", False):
            self._direct_h2d_layers = set(self._cpu_sources)
            return

        source_views = {
            name: view
            for by_dtype in self._cpu_sources.values()
            for sources in by_dtype.values()
            for name, view, _ in sources
        }
        model_tensors = dict(self.model.named_parameters())
        model_tensors.update(dict(self.model.named_buffers()))

        page = 4096
        spans: Dict[str, List[int]] = {}
        for name, binding in self._mmap_plan.bindings.items():
            view = source_views.get(name)
            if view is None:
                view = model_tensors.get(name)
            if view is None or view.device.type != "cpu" or view.is_meta:
                continue
            start = view.data_ptr()
            end = start + view.numel() * view.element_size()
            span = spans.setdefault(binding.file_path, [start, end])
            span[0] = min(span[0], start)
            span[1] = max(span[1], end)

        merged = [
            (start // page * page, -(-end // page) * page)
            for start, end in spans.values()
        ]
        if not merged:
            return
        eligible = set(self._cpu_sources)
        total_bytes = sum(end - start for start, end in merged)
        if self._pin_budget is not None and not self._pin_budget.request(
            component_name=f"{self._pin_component_name}.mmap_register",
            weight_bytes=total_bytes,
        ):
            return

        cudart = torch.cuda.cudart()
        registered: List[Tuple[int, int]] = []
        # cudaHostRegisterReadOnly: the only flag valid for PROT_READ mappings.
        flags = 0x08
        for start, end in merged:
            error = cudart.cudaHostRegister(start, end - start, flags)
            if error != 0:
                for done_start, _ in registered:
                    cudart.cudaHostUnregister(done_start)
                # The failed call leaves a sticky last-error that torch would
                # raise from the next unrelated CUDA op.
                cudart.cudaGetLastError()
                logger.info(
                    "Direct H2D unavailable for %s (cudaHostRegister error %s "
                    "on a %.1f MiB range); keeping the pinned-staging path.",
                    self._pin_component_name,
                    error,
                    (end - start) / (1 << 20),
                )
                return
            registered.append((start, end))

        self._direct_h2d_layers = eligible
        self.model._dlo_direct_h2d_registered = True
        logger.info(
            "Registered %.2f GiB of checkpoint-mmap pages read-only across %d "
            "range(s); %d/%d mmap layers prefetch by direct H2D.",
            total_bytes / (1 << 30),
            len(registered),
            len(eligible),
            len(self._cpu_sources),
        )

    def _stage_mmap_sources(
        self, layer_idx: int, slot: int
    ) -> Dict[torch.dtype, torch.Tensor]:
        """Pack one layer's mmap views into a bounded pinned staging slot.

        The slot may only be overwritten after its previous H2D finished; the
        per-slot event guards reuse.
        """
        previous_copy = self._staging_events[slot]
        if previous_copy is not None:
            previous_copy.synchronize()
            self._staging_events[slot] = None

        staged: Dict[torch.dtype, torch.Tensor] = {}
        for dtype, sources in self._cpu_sources[layer_idx].items():
            total_numel, _, _ = self._flat_sizes[layer_idx][dtype]
            staging = self._staging_buffers[slot][dtype]
            for name, view, transform in sources:
                meta = self._weight_metadata[layer_idx][name]
                flat = (transform(view) if transform is not None else view).flatten()
                staging[meta["offset"] : meta["offset"] + meta["numel"]].copy_(flat)
            staged[dtype] = staging[:total_numel]
        return staged

    def _copy_mmap_sources_to_device(
        self,
        layer_idx: int,
        dtype: torch.dtype,
        destination: torch.Tensor,
        non_blocking: bool,
    ) -> None:
        """Copy one dtype group's mmap views straight into a device buffer."""
        for name, view, transform in self._cpu_sources[layer_idx][dtype]:
            meta = self._weight_metadata[layer_idx][name]
            flat = (transform(view) if transform is not None else view).flatten()
            destination[meta["offset"] : meta["offset"] + meta["numel"]].copy_(
                flat, non_blocking=non_blocking and flat.is_pinned()
            )

    @torch.compiler.disable
    def prefetch_layer(
        self,
        layer_idx: int,
        non_blocking: bool = True,
        dedicated_buffers: bool = False,
    ) -> None:
        """Materialize one layer through the DLO bounded device-slot ring.

        AllGather mode: the local shard's H2D runs on ``copy_stream``, the
        AllGather reconstructs the full flat buffer on ``comm_stream``, and the
        recorded event lands there so the pre-hook ``wait_event`` covers the
        collective. Every rank of the shard group executes the same denoise
        schedule, so the collective is always invoked in lockstep; a
        rank-asymmetric caller would deadlock NCCL.

        Rank-local mode: the complete flat buffer (pinned private copy, or the
        staged pack of immutable mmap views) is copied H2D on ``copy_stream``.

        Layers that may become residents (``layer_idx < resident_layers``) and
        ``dedicated_buffers`` callers (``load_all_layers``, where every layer is
        live at once) get their own full-size buffers instead of ring slots;
        those are freed when the parameters are rebound to placeholders.
        """
        if not self.enabled or self.device is None or self.copy_stream is None:
            return
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if layer_idx in self._gpu_layers:
            return
        if not self._has_layer_storage(layer_idx):
            return

        # Residency is policy-aware upstream (leading or strided); resident-
        # designated layers persist across steps and must not sit in ring slots.
        dedicated = dedicated_buffers or layer_idx in self._resident_set
        if dedicated:
            slot = None
        else:
            slot = self._ring_fetch_counter % len(self._shared_out_buffers)
            self._ring_fetch_counter += 1

        mmap_layer = layer_idx in self._cpu_sources
        direct_h2d = layer_idx in self._direct_h2d_layers
        staged: Dict[torch.dtype, torch.Tensor] = {}
        if mmap_layer and not dedicated and not direct_h2d:
            staged = self._stage_mmap_sources(layer_idx, slot)

        self.copy_stream.wait_stream(torch.get_device_module().current_stream())

        gathered_buffers: Dict[torch.dtype, torch.Tensor] = {}
        staged_shards: Dict[torch.dtype, torch.Tensor] = {}
        strided_gpu: Dict[str, torch.Tensor] = {}
        with torch.inference_mode(False), torch.no_grad():
            with torch.get_device_module().stream(self.copy_stream):
                for dtype, sizes in self._flat_sizes[layer_idx].items():
                    total_numel, shard_numel, padded_numel = sizes
                    if dedicated:
                        gpu_out = torch.empty(
                            padded_numel, dtype=dtype, device=self.device
                        )
                    else:
                        gpu_out = self._shared_out_buffers[slot][dtype][:padded_numel]
                    if self._shard_group is not None:
                        cpu_shard = self._consolidated_cpu_weights[layer_idx][dtype]
                        if dedicated:
                            gpu_shard = torch.empty(
                                shard_numel, dtype=dtype, device=self.device
                            )
                        else:
                            gpu_shard = self._shared_shard_buffers[slot][dtype][
                                :shard_numel
                            ]
                        gpu_shard.copy_(cpu_shard, non_blocking=non_blocking)
                        staged_shards[dtype] = gpu_shard
                    elif mmap_layer:
                        if direct_h2d and not dedicated:
                            # Registered pages: DMA each view straight into the
                            # slot, no host pack.
                            self._copy_mmap_sources_to_device(
                                layer_idx, dtype, gpu_out, non_blocking=non_blocking
                            )
                        elif dedicated:
                            self._copy_mmap_sources_to_device(
                                layer_idx, dtype, gpu_out, non_blocking=False
                            )
                        else:
                            gpu_out[:total_numel].copy_(
                                staged[dtype], non_blocking=non_blocking
                            )
                    else:
                        gpu_out[:total_numel].copy_(
                            self._consolidated_cpu_weights[layer_idx][dtype],
                            non_blocking=non_blocking,
                        )
                    gathered_buffers[dtype] = gpu_out

                # Stride-preserving tensors stay rank-local full copies (see
                # base _initialize) and transfer on the copy stream as usual.
                for name, meta in self._weight_metadata[layer_idx].items():
                    if not meta.get("preserve_strides", False):
                        continue
                    gpu_tensor = torch.empty_strided(
                        size=meta["shape"],
                        stride=meta["stride"],
                        dtype=meta["dtype"],
                        device=self.device,
                    )
                    gpu_tensor.copy_(
                        self._strided_cpu_weights[layer_idx][name],
                        non_blocking=non_blocking,
                    )
                    strided_gpu[name] = gpu_tensor

            if self._shard_group is not None:
                self.comm_stream.wait_stream(self.copy_stream)
                with torch.get_device_module().stream(self.comm_stream):
                    for dtype, gpu_shard in staged_shards.items():
                        # Slice down to this layer's exact AllGather output size
                        # (out.numel() must equal world * in.numel()).
                        torch.distributed.all_gather_into_tensor(
                            gathered_buffers[dtype],
                            gpu_shard,
                            group=self._shard_group.device_group,
                        )

            for name, meta in self._weight_metadata[layer_idx].items():
                target = self.get_target_with_name(name)
                if meta.get("preserve_strides", False):
                    target.data = self._wrap_for_target(target, strided_gpu[name])
                    continue
                gpu_buffer = gathered_buffers[meta["dtype"]]
                local_tensor = gpu_buffer[
                    meta["offset"] : meta["offset"] + meta["numel"]
                ].view(meta["shape"])
                target.data = self._wrap_for_target(target, local_tensor)

        event = torch.get_device_module().Event()
        event.record(
            self.comm_stream if self._shard_group is not None else self.copy_stream
        )
        self._prefetch_events[layer_idx] = event
        if mmap_layer and not dedicated and not direct_h2d:
            self._staging_events[slot] = event

        self._gpu_layers.add(layer_idx)

    @torch.compiler.disable
    def load_all_layers(self) -> None:
        """Load all layers to GPU (every layer live at once -> dedicated buffers)."""
        if not self.enabled or self.device is None:
            return
        self._wait_transfer_streams()

        for layer_idx in range(self.num_layers):
            if layer_idx not in self._gpu_layers:
                self.prefetch_layer(
                    layer_idx, non_blocking=False, dedicated_buffers=True
                )
        self._wait_transfer_streams()

    # ------------------------------------------------------------------ #
    #  Writeback / iteration                                               #
    # ------------------------------------------------------------------ #

    def _detach_layer_to_private_host(
        self, layer_idx: int, *, fill_from_sources: bool
    ) -> None:
        """Replace a layer's immutable mmap sources with private pinned storage.

        Mapped checkpoint pages are never written; the first writeback (LoRA
        merge / refit) materializes a private flat buffer instead — host weight
        sharing is lost for this layer only.
        """
        sources_by_dtype = self._cpu_sources.pop(layer_idx)
        for dtype, sources in sources_by_dtype.items():
            total_numel, _, _ = self._flat_sizes[layer_idx][dtype]
            cpu_buffer = torch.empty(
                total_numel, dtype=dtype, pin_memory=self.pin_cpu_memory
            )
            if fill_from_sources:
                for name, view, transform in sources:
                    meta = self._weight_metadata[layer_idx][name]
                    flat = (
                        transform(view) if transform is not None else view
                    ).flatten()
                    cpu_buffer[
                        meta["offset"] : meta["offset"] + meta["numel"]
                    ].copy_(flat)
            self._consolidated_cpu_weights.setdefault(layer_idx, {})[dtype] = (
                cpu_buffer
            )
        logger.warning(
            "Layer %d detached from the node-shared checkpoint mapping for a "
            "writeback; host weight sharing is lost for this layer.",
            layer_idx,
        )

    @torch.compiler.disable
    def sync_layer_to_cpu(self, layer_idx: int) -> None:
        """Sync a layer's weights from GPU back to host storage.

        AllGather mode persists only this rank's shard slice, so every rank
        must run the same writeback (true for LoRA merge and refit, which
        produce identical results on every replica). Rank-local mmap layers
        detach to private storage first.
        """
        if not self.enabled or layer_idx not in self._gpu_layers:
            return
        if not self._has_layer_storage(layer_idx):
            return

        if layer_idx in self._cpu_sources:
            # The GPU tensors hold the complete layer; the base writeback below
            # fills the fresh private buffer entirely.
            self._detach_layer_to_private_host(layer_idx, fill_from_sources=False)

        if self._shard_group is None:
            super().sync_layer_to_cpu(layer_idx)
            return

        self._wait_transfer_streams()
        for name, meta in self._weight_metadata.get(layer_idx, {}).items():
            target = self.get_target_with_name(name)
            target_local = self._to_local_tensor(target)
            if meta.get("preserve_strides", False):
                self._strided_cpu_weights[layer_idx][name].copy_(target_local.cpu())
                continue
            dtype = meta["dtype"]
            _, shard_numel, _ = self._flat_sizes[layer_idx][dtype]
            copy_overlap_into_shard(
                shard=self._consolidated_cpu_weights[layer_idx][dtype],
                shard_start=self._shard_rank * shard_numel,
                offset=meta["offset"],
                flat_source=target_local.flatten().cpu(),
            )

    @torch.compiler.disable
    def update_cpu_weights(self, weight_dict: Dict[str, torch.Tensor]):
        """Update host storage with new weights (refit).

        AllGather mode persists each rank's shard slice; rank-local mmap layers
        detach to private storage (filled from the sources first, since the
        update may be partial).
        """
        if not self.enabled:
            return None

        touched_layers = sorted(
            {
                layer_idx
                for layer_idx in (
                    self._match_layer_idx(name) for name in weight_dict
                )
                if layer_idx is not None and layer_idx in self._cpu_sources
            }
        )
        for layer_idx in touched_layers:
            self._detach_layer_to_private_host(layer_idx, fill_from_sources=True)

        if self._shard_group is None:
            return super().update_cpu_weights(weight_dict)

        updated_names = set()
        for name, loaded_weight in weight_dict.items():
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None:
                continue
            meta_layer = self._weight_metadata.get(layer_idx)
            if meta_layer is None or name not in meta_layer:
                continue

            meta = meta_layer[name]
            local_loaded_weight = self._to_local_tensor(loaded_weight)
            if tuple(meta["shape"]) != tuple(local_loaded_weight.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"expected={tuple(meta['shape'])}, "
                    f"loaded={tuple(local_loaded_weight.shape)}"
                )

            dtype = meta["dtype"]
            if meta.get("preserve_strides", False):
                self._strided_cpu_weights[layer_idx][name].copy_(
                    local_loaded_weight.to(dtype=dtype)
                )
            else:
                _, shard_numel, _ = self._flat_sizes[layer_idx][dtype]
                copy_overlap_into_shard(
                    shard=self._consolidated_cpu_weights[layer_idx][dtype],
                    shard_start=self._shard_rank * shard_numel,
                    offset=meta["offset"],
                    flat_source=local_loaded_weight.to(dtype=dtype).flatten().cpu(),
                )

            # If this layer is currently on GPU, update the live parameter.
            if layer_idx in self._gpu_layers:
                target = self.get_target_with_name(name)
                target_local = self._to_local_tensor(target)
                target_local.copy_(local_loaded_weight.to(dtype=target_local.dtype))

            updated_names.add(name)

        return updated_names

    def iter_cpu_weights(self):
        """Yield (name, tensor) pairs with real weights from host storage.

        With AllGather sharding this is a collective: every rank of the shard
        group must iterate together (each layer's full buffer is reconstructed
        from the per-rank shards over the group's gloo cpu_group). Existing
        callers — refit checksums — already run on every rank. Rank-local mmap
        layers yield transform-applied views of the immutable sources.
        """
        if self._shard_group is None and self._mmap_plan is None:
            yield from super().iter_cpu_weights()
            return

        for layer_idx in sorted(self._weight_metadata):
            source_tensors: Dict[str, torch.Tensor] = {}
            full_buffers: Dict[torch.dtype, torch.Tensor] = {}
            if layer_idx in self._cpu_sources:
                for sources in self._cpu_sources[layer_idx].values():
                    for name, view, transform in sources:
                        source_tensors[name] = (
                            transform(view) if transform is not None else view
                        )
            elif self._shard_group is not None:
                for dtype, shard in self._consolidated_cpu_weights[
                    layer_idx
                ].items():
                    total_numel, _, _ = self._flat_sizes[layer_idx][dtype]
                    full_buffers[dtype] = gather_full_cpu_buffer(
                        shard=shard,
                        total_numel=total_numel,
                        cpu_group=self._shard_group.cpu_group,
                        world_size=self._shard_world,
                    )
            for name, meta in self._weight_metadata[layer_idx].items():
                if meta.get("preserve_strides", False):
                    yield name, self._strided_cpu_weights[layer_idx][name]
                    continue
                if name in source_tensors:
                    yield name, source_tensors[name]
                    continue
                dtype = meta["dtype"]
                if dtype in full_buffers:
                    cpu_buffer = full_buffers[dtype]
                else:
                    cpu_buffer = self._consolidated_cpu_weights[layer_idx][dtype]
                yield name, cpu_buffer[
                    meta["offset"] : meta["offset"] + meta["numel"]
                ].reshape(meta["shape"])
