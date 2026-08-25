"""Unit tests for distributed layerwise offload (DLO).

Covers the shard math, the AllGather-sharded DistributedLayerwiseOffloadManager
(multi-rank emulation with a fake collective hub on the fake CPU device), the
rank-local transfer path with and without checkpoint-mmap host backing, and a
real 2-process gloo end-to-end check.
"""

import copy
import multiprocessing
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.managers.memory_managers import (
    layerwise_offload as layerwise_offload_mod,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadManager,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_distributed import (
    DistributedLayerwiseOffloadManager,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_sharding import (
    copy_overlap_into_shard,
    shard_numels,
)


class _FakeStream:
    def wait_stream(self, _stream) -> None:
        return None

    def wait_event(self, _event) -> None:
        return None


class _FakeEvent:
    def record(self, _stream) -> None:
        return None

    def synchronize(self) -> None:
        return None


class _FakeDeviceModule:
    Stream = _FakeStream
    Event = _FakeEvent

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def current_stream() -> _FakeStream:
        return _FakeStream()

    @staticmethod
    def stream(_stream):
        return nullcontext()

    @staticmethod
    def synchronize() -> None:
        return None


def _patch_fake_device(monkeypatch):
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")


class _DistBlock(torch.nn.Module):
    """One contiguous 2D weight, a bias, and a non-contiguous (transposed) proj.

    Parameters are requires_grad=False, matching what the loader produces for
    inference weights (fsdp_load sets requires_grad=False after loading).
    """

    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(
            torch.randn(4, 6, generator=generator, dtype=torch.float32),
            requires_grad=False,
        )
        self.bias = torch.nn.Parameter(
            torch.randn(4, generator=generator, dtype=torch.float32),
            requires_grad=False,
        )
        self.proj = torch.nn.Parameter(
            torch.randn(3, 5, generator=generator, dtype=torch.float32).t(),
            requires_grad=False,
        )


class _DistModel(torch.nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [_DistBlock(seed) for seed in range(num_layers)]
        )


class _FakeShardGroup(SimpleNamespace):
    pass


def _fake_shard_group(world_size: int, rank: int) -> _FakeShardGroup:
    return _FakeShardGroup(
        world_size=world_size,
        rank_in_group=rank,
        device_group=None,
        cpu_group=None,
        ranks=list(range(world_size)),
    )


class _FakeAllGatherHub:
    """Pairs the k-th all_gather_into_tensor call of every emulated rank.

    Ranks run sequentially in tests, so outputs are filled at flush() time —
    valid because parameters are rebound as views into persistent buffers and
    assertions only run after the flush.
    """

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.current_rank = 0
        self._calls = {rank: [] for rank in range(world_size)}

    def all_gather_into_tensor(self, out, shard, group=None) -> None:
        self._calls[self.current_rank].append((out, shard.detach().clone()))

    def flush(self) -> None:
        counts = {rank: len(calls) for rank, calls in self._calls.items()}
        assert len(set(counts.values())) == 1, f"collective divergence: {counts}"
        for call_idx in range(next(iter(counts.values()))):
            shards = [
                self._calls[rank][call_idx][1] for rank in range(self.world_size)
            ]
            gathered = torch.cat(shards)
            for rank in range(self.world_size):
                out = self._calls[rank][call_idx][0]
                out.copy_(gathered[: out.numel()])
        for calls in self._calls.values():
            calls.clear()


def _build_rank_managers(
    monkeypatch,
    *,
    world_size: int,
    num_layers: int,
    prefetch_size: int = 1,
    resident_layers: int = 0,
):
    """Build one sharded manager per emulated rank over deep-copied models."""
    _patch_fake_device(monkeypatch)
    hub = _FakeAllGatherHub(world_size)
    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", hub.all_gather_into_tensor
    )

    reference_model = _DistModel(num_layers)
    originals = {
        name: tensor.detach().clone()
        for name, tensor in reference_model.named_parameters()
    }
    models = [copy.deepcopy(reference_model) for _ in range(world_size)]
    managers = []
    for rank, model in enumerate(models):
        hub.current_rank = rank
        managers.append(
            DistributedLayerwiseOffloadManager(
                model=model,
                layers_attr_str="blocks",
                num_layers=num_layers,
                enabled=True,
                pin_cpu_memory=False,
                prefetch_size=prefetch_size,
                resident_layers=resident_layers,
                shard_group=_fake_shard_group(world_size, rank),
            )
        )
    hub.flush()
    return hub, models, managers, originals


def _for_each_rank(hub, managers, op) -> None:
    """Run one manager operation on every rank in lockstep, then flush."""
    for rank, manager in enumerate(managers):
        hub.current_rank = rank
        op(manager)
    hub.flush()


def _assert_layer_matches(models, originals, layer_idx: int) -> None:
    for model in models:
        block = model.blocks[layer_idx]
        for suffix in ("weight", "bias", "proj"):
            name = f"blocks.{layer_idx}.{suffix}"
            restored = getattr(block, suffix).data
            assert restored.shape == originals[name].shape, name
            assert torch.equal(restored, originals[name]), name


# ---------------------------------------------------------------------------
# Shard math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("world_size", [1, 2, 3, 8])
@pytest.mark.parametrize("total_numel", [0, 1, 5, 7, 8, 63, 64, 100, 1023])
def test_shard_numels_covers_and_aligns(total_numel, world_size, dtype):
    shard, padded = shard_numels(
        total_numel=total_numel, world_size=world_size, dtype=dtype
    )
    element_size = torch.empty((), dtype=dtype).element_size()
    alignment_numel = max(1, 32 // element_size)
    assert padded == shard * world_size
    assert padded >= total_numel
    assert shard % alignment_numel == 0 or shard == 0
    assert shard >= (total_numel + world_size - 1) // world_size


@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_shard_overlap_copies_reassemble_full_buffer(world_size):
    total_numel = 103
    full = torch.arange(total_numel, dtype=torch.float32)
    shard_numel, _ = shard_numels(
        total_numel=total_numel, world_size=world_size, dtype=torch.float32
    )
    shards = []
    for rank in range(world_size):
        shard = torch.zeros(shard_numel, dtype=torch.float32)
        # Simulate several params at arbitrary offsets covering the buffer.
        for offset, numel in ((0, 40), (40, 3), (43, 60)):
            copy_overlap_into_shard(
                shard=shard,
                shard_start=rank * shard_numel,
                offset=offset,
                flat_source=full[offset : offset + numel],
            )
        shards.append(shard)
    reassembled = torch.cat(shards)[:total_numel]
    assert torch.equal(reassembled, full)


# ---------------------------------------------------------------------------
# Sharded manager: multi-rank emulation
# ---------------------------------------------------------------------------


def test_sharded_metadata_matches_rank_local_manager(monkeypatch):
    _, _, managers, _ = _build_rank_managers(monkeypatch, world_size=2, num_layers=2)

    local_manager = LayerwiseOffloadManager(
        model=_DistModel(2),
        layers_attr_str="blocks",
        num_layers=2,
        enabled=True,
        pin_cpu_memory=False,
        prefetch_size=1,
    )

    for manager in managers:
        assert manager._weight_metadata == local_manager._weight_metadata
        for layer_metadata in manager._weight_metadata.values():
            for meta in layer_metadata.values():
                if meta["preserve_strides"]:
                    continue
                element_size = torch.empty((), dtype=meta["dtype"]).element_size()
                assert (meta["offset"] * element_size) % 32 == 0


def test_sharded_reconstruction_is_bit_exact_on_every_rank(monkeypatch):
    hub, models, managers, originals = _build_rank_managers(
        monkeypatch, world_size=3, num_layers=2
    )
    # Layer 0 was prefetched during _initialize (flushed by the builder).
    _assert_layer_matches(models, originals, 0)

    _for_each_rank(hub, managers, lambda m: m.prefetch_layer(1, non_blocking=False))
    _assert_layer_matches(models, originals, 1)

    # Each rank persists only 1/world of the flat host bytes (plus the private
    # stride-preserving proj), instead of a full copy.
    for manager in managers:
        for layer_idx, sizes in manager._flat_sizes.items():
            for dtype, (total, shard, padded) in sizes.items():
                assert (
                    manager._consolidated_cpu_weights[layer_idx][dtype].numel()
                    == shard
                )
                assert padded == shard * 3
                assert padded >= total


def test_sharded_hook_schedule_never_corrupts_live_layer(monkeypatch):
    """Emulate the pre/post hook order for 2 full passes over an ODD stack."""
    num_layers = 5
    hub, models, managers, originals = _build_rank_managers(
        monkeypatch, world_size=2, num_layers=num_layers, prefetch_size=1
    )

    for _step in range(2):
        for i in range(num_layers):

            def pre_hook(manager, i=i):
                if i == 0:
                    manager._activate_residency()
                    manager.prepare_for_next_req(non_blocking=False)
                if i not in manager._gpu_layers:
                    manager.prefetch_layer(i, non_blocking=False)
                if i % manager.prefetch_size == 0:
                    for j in range(
                        i + manager.prefetch_size, i + 2 * manager.prefetch_size
                    ):
                        manager.prefetch_layer(j % num_layers, non_blocking=True)

            _for_each_rank(hub, managers, pre_hook)
            # The live layer must hold exact weights on every rank, even after
            # the lookahead prefetch reused ring slots (incl. wraparound).
            _assert_layer_matches(models, originals, i)
            _for_each_rank(hub, managers, lambda m, i=i: m.release_layer(i))


def test_sharded_resident_layers_survive_ring_reuse(monkeypatch):
    num_layers = 4
    hub, models, managers, originals = _build_rank_managers(
        monkeypatch, world_size=2, num_layers=num_layers, resident_layers=2
    )

    def arm(manager):
        manager._activate_residency()
        manager.prepare_for_next_req(non_blocking=False)

    _for_each_rank(hub, managers, arm)

    # Stream the tail repeatedly so the shared ring slots are reused many times.
    for _ in range(3):
        for i in range(2, num_layers):
            _for_each_rank(
                hub, managers, lambda m, i=i: m.prefetch_layer(i, non_blocking=False)
            )
            _for_each_rank(hub, managers, lambda m, i=i: m.release_layer(i))

    # Residents were gathered once into dedicated buffers and must be intact.
    _assert_layer_matches(models, originals, 0)
    _assert_layer_matches(models, originals, 1)
    for manager in managers:
        assert {0, 1} <= manager._gpu_layers

    _for_each_rank(hub, managers, lambda m: m.release_all())
    for manager in managers:
        assert not manager._gpu_layers


def test_sharded_load_all_layers_materializes_everything(monkeypatch):
    num_layers = 3
    hub, models, managers, originals = _build_rank_managers(
        monkeypatch, world_size=2, num_layers=num_layers
    )
    _for_each_rank(hub, managers, lambda m: m.load_all_layers())
    for layer_idx in range(num_layers):
        _assert_layer_matches(models, originals, layer_idx)


def test_sharded_writeback_roundtrip(monkeypatch):
    hub, models, managers, originals = _build_rank_managers(
        monkeypatch, world_size=2, num_layers=2
    )

    new_weight = originals["blocks.0.weight"] + 1.0
    _for_each_rank(
        hub,
        managers,
        lambda m: m.update_cpu_weights({"blocks.0.weight": new_weight}),
    )
    # The live layer-0 parameter was updated in place on every rank.
    for model in models:
        assert torch.equal(model.blocks[0].weight.data, new_weight)

    # The persisted shards were updated too: release and re-gather.
    _for_each_rank(hub, managers, lambda m: m.release_layer(0, force=True))
    _for_each_rank(hub, managers, lambda m: m.prefetch_layer(0, non_blocking=False))
    for model in models:
        assert torch.equal(model.blocks[0].weight.data, new_weight)
        assert torch.equal(model.blocks[0].bias.data, originals["blocks.0.bias"])


def test_sharded_sync_layer_to_cpu_persists_shard_overlap(monkeypatch):
    hub, models, managers, originals = _build_rank_managers(
        monkeypatch, world_size=2, num_layers=2
    )
    # Mutate the live layer-0 weights identically on every rank, then persist.
    for model in models:
        model.blocks[0].weight.data.add_(2.0)
        model.blocks[0].proj.data.add_(3.0)
    _for_each_rank(hub, managers, lambda m: m.sync_layer_to_cpu(0))
    _for_each_rank(hub, managers, lambda m: m.release_layer(0, force=True))
    _for_each_rank(hub, managers, lambda m: m.prefetch_layer(0, non_blocking=False))
    for model in models:
        assert torch.equal(
            model.blocks[0].weight.data, originals["blocks.0.weight"] + 2.0
        )
        assert torch.equal(
            model.blocks[0].proj.data, originals["blocks.0.proj"] + 3.0
        )
        assert not model.blocks[0].proj.data.is_contiguous()


def test_sharded_rejects_dtensor_weights(monkeypatch):
    _patch_fake_device(monkeypatch)
    monkeypatch.setattr(
        layerwise_offload_mod, "DTensor", torch.nn.Parameter
    )  # every param now "is" a DTensor
    monkeypatch.setattr(
        LayerwiseOffloadManager, "_to_local_tensor", staticmethod(lambda tensor: tensor)
    )
    with pytest.raises(ValueError, match="DTensor"):
        DistributedLayerwiseOffloadManager(
            model=_DistModel(1),
            layers_attr_str="blocks",
            num_layers=1,
            enabled=True,
            pin_cpu_memory=False,
            prefetch_size=1,
            shard_group=_fake_shard_group(2, 0),
        )


def test_base_manager_keeps_legacy_layout(monkeypatch):
    """Regression: the plain manager keeps full private flat buffers."""
    _patch_fake_device(monkeypatch)
    manager = LayerwiseOffloadManager(
        model=_DistModel(2),
        layers_attr_str="blocks",
        num_layers=2,
        enabled=True,
        pin_cpu_memory=False,
        prefetch_size=1,
    )
    assert type(manager) is LayerwiseOffloadManager
    for layer_idx, per_dtype in manager._consolidated_cpu_weights.items():
        for dtype, cpu_buffer in per_dtype.items():
            covered = max(
                meta["offset"] + meta["numel"]
                for meta in manager._weight_metadata[layer_idx].values()
                if not meta["preserve_strides"] and meta["dtype"] == dtype
            )
            assert cpu_buffer.numel() == covered


# ---------------------------------------------------------------------------
# Rank-local DLO (--dlo-no-use-allgather): complete blocks stream through the
# bounded device slots without any process group or collective.
# ---------------------------------------------------------------------------


def _rank_local_manager(monkeypatch, *, num_layers, prefetch_size=1, mmap_plan=None):
    _patch_fake_device(monkeypatch)
    model = _DistModel(num_layers)
    originals = {
        name: tensor.detach().clone() for name, tensor in model.named_parameters()
    }
    manager = DistributedLayerwiseOffloadManager(
        model=model,
        layers_attr_str="blocks",
        num_layers=num_layers,
        enabled=True,
        pin_cpu_memory=False,
        prefetch_size=prefetch_size,
        shard_group=None,
        mmap_plan=mmap_plan,
    )
    return model, manager, originals


def test_rank_local_uses_bounded_slots_without_group(monkeypatch):
    model, manager, originals = _rank_local_manager(monkeypatch, num_layers=2)
    assert manager._shard_group is None
    assert manager.comm_stream is None
    assert manager._shared_out_buffers is not None
    assert manager._shared_shard_buffers is None
    assert manager._staging_buffers is None  # no mmap plan -> no staging

    manager.prefetch_layer(1, non_blocking=False)
    for layer_idx in (0, 1):
        block = model.blocks[layer_idx]
        for suffix in ("weight", "bias", "proj"):
            assert torch.equal(
                getattr(block, suffix).data,
                originals[f"blocks.{layer_idx}.{suffix}"],
            )


def test_rank_local_hook_schedule_never_corrupts_live_layer(monkeypatch):
    num_layers = 5
    model, manager, originals = _rank_local_manager(monkeypatch, num_layers=num_layers)
    for _step in range(2):
        for i in range(num_layers):
            if i == 0:
                manager._activate_residency()
                manager.prepare_for_next_req(non_blocking=False)
            if i not in manager._gpu_layers:
                manager.prefetch_layer(i, non_blocking=False)
            if i % manager.prefetch_size == 0:
                for j in range(
                    i + manager.prefetch_size, i + 2 * manager.prefetch_size
                ):
                    manager.prefetch_layer(j % num_layers, non_blocking=True)
            block = model.blocks[i]
            for suffix in ("weight", "bias", "proj"):
                assert torch.equal(
                    getattr(block, suffix).data,
                    originals[f"blocks.{i}.{suffix}"],
                ), f"corruption at layer {i} ({suffix})"
            manager.release_layer(i)


def test_rank_local_writeback_roundtrip(monkeypatch):
    model, manager, originals = _rank_local_manager(monkeypatch, num_layers=2)
    model.blocks[0].weight.data.add_(1.0)
    manager.sync_layer_to_cpu(0)
    manager.release_layer(0, force=True)
    manager.prefetch_layer(0, non_blocking=False)
    assert torch.equal(
        model.blocks[0].weight.data, originals["blocks.0.weight"] + 1.0
    )
    assert torch.equal(model.blocks[0].bias.data, originals["blocks.0.bias"])


# ---------------------------------------------------------------------------
# Rank-local DLO with checkpoint-mmap host backing: immutable sources, bounded
# pinned staging, deferred transforms, detach-on-write.
# ---------------------------------------------------------------------------


def _plus_one_transform(tensor: torch.Tensor) -> torch.Tensor:
    return tensor + 1.0


def test_mmap_backing_stages_and_applies_deferred_transforms(monkeypatch):
    plan = SimpleNamespace(
        transforms={"blocks.0.weight": _plus_one_transform},
        bindings={},
    )
    model, manager, originals = _rank_local_manager(
        monkeypatch, num_layers=3, mmap_plan=plan
    )
    assert manager._staging_buffers is not None
    assert set(manager._cpu_sources) == {0, 1, 2}
    # No private flat host buffers were materialized for mmap-backed layers.
    assert all(not per for per in manager._consolidated_cpu_weights.values())

    # Stream layer by layer (release before a slot can be reused, as the
    # forward hooks do). The deferred transform applies to layer 0 only.
    for layer_idx in range(3):
        manager.prefetch_layer(layer_idx, non_blocking=False)
        block = model.blocks[layer_idx]
        expected_weight = originals[f"blocks.{layer_idx}.weight"]
        if layer_idx == 0:
            expected_weight = expected_weight + 1.0
        assert torch.equal(block.weight.data, expected_weight)
        assert torch.equal(block.bias.data, originals[f"blocks.{layer_idx}.bias"])
        # Strided tensors stay private per rank and keep their layout.
        assert not block.proj.data.is_contiguous()
        assert torch.equal(block.proj.data, originals[f"blocks.{layer_idx}.proj"])
        manager.release_layer(layer_idx)


def test_mmap_backing_iter_cpu_weights_yields_runtime_layout(monkeypatch):
    plan = SimpleNamespace(
        transforms={"blocks.0.weight": _plus_one_transform},
        bindings={},
    )
    _, manager, originals = _rank_local_manager(
        monkeypatch, num_layers=2, mmap_plan=plan
    )
    iterated = dict(manager.iter_cpu_weights())
    assert torch.equal(
        iterated["blocks.0.weight"], originals["blocks.0.weight"] + 1.0
    )
    assert torch.equal(iterated["blocks.1.weight"], originals["blocks.1.weight"])
    assert torch.equal(iterated["blocks.0.proj"], originals["blocks.0.proj"])


def test_mmap_backing_detaches_layer_on_writeback(monkeypatch):
    plan = SimpleNamespace(transforms={}, bindings={})
    model, manager, originals = _rank_local_manager(
        monkeypatch, num_layers=2, mmap_plan=plan
    )
    source_weight = originals["blocks.0.weight"].clone()

    # sync path: layer 0 detaches, the mapped source stays untouched.
    manager.prefetch_layer(0, non_blocking=False)
    model.blocks[0].weight.data.add_(2.0)
    manager.sync_layer_to_cpu(0)
    assert 0 not in manager._cpu_sources
    assert 1 in manager._cpu_sources
    manager.release_layer(0, force=True)
    manager.prefetch_layer(0, non_blocking=False)
    assert torch.equal(model.blocks[0].weight.data, source_weight + 2.0)

    # update path: layer 1 detaches with source fill + partial update.
    new_bias = originals["blocks.1.bias"] + 5.0
    manager.update_cpu_weights({"blocks.1.bias": new_bias})
    assert 1 not in manager._cpu_sources
    manager.release_layer(1, force=True)
    manager.prefetch_layer(1, non_blocking=False)
    assert torch.equal(model.blocks[1].bias.data, new_bias)
    assert torch.equal(model.blocks[1].weight.data, originals["blocks.1.weight"])


# ---------------------------------------------------------------------------
# Real 2-process gloo end-to-end (skips cleanly where spawn/gloo unavailable)
# ---------------------------------------------------------------------------


def _gloo_worker(rank: int, world_size: int, port: int, queue) -> None:
    try:
        import sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload as mod

        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
        )
        mod.torch.get_device_module = lambda: _FakeDeviceModule
        mod.current_platform.device_type = "cpu"

        model = _DistModel(3)
        originals = {
            name: tensor.detach().clone()
            for name, tensor in model.named_parameters()
        }
        group = SimpleNamespace(
            cpu_group=torch.distributed.group.WORLD,
            device_group=torch.distributed.group.WORLD,
            world_size=world_size,
            rank_in_group=rank,
            ranks=list(range(world_size)),
        )
        manager = DistributedLayerwiseOffloadManager(
            model=model,
            layers_attr_str="blocks",
            num_layers=3,
            enabled=True,
            pin_cpu_memory=False,
            prefetch_size=1,
            shard_group=group,
        )
        for layer_idx in range(3):
            manager.prefetch_layer(layer_idx, non_blocking=False)
            block = model.blocks[layer_idx]
            for suffix in ("weight", "bias", "proj"):
                name = f"blocks.{layer_idx}.{suffix}"
                if not torch.equal(getattr(block, suffix).data, originals[name]):
                    queue.put((rank, f"mismatch at {name}"))
                    return
            manager.release_layer(layer_idx)
        # Collective full-weight reconstruction over the gloo cpu_group.
        iterated = dict(manager.iter_cpu_weights())
        for name, original in originals.items():
            if not torch.equal(iterated[name], original):
                queue.put((rank, f"iter_cpu_weights mismatch at {name}"))
                return
        queue.put((rank, "ok"))
    except Exception as exc:  # pragma: no cover - propagated to the parent
        queue.put((rank, f"error: {exc!r}"))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_two_process_gloo_end_to_end():
    if sys.platform == "win32":
        pytest.skip("spawn+gloo test is POSIX-only")
    context = multiprocessing.get_context("spawn")
    queue = context.SimpleQueue()
    workers = [
        context.Process(target=_gloo_worker, args=(rank, 2, 29612, queue))
        for rank in range(2)
    ]
    for worker in workers:
        worker.start()
    results = {}
    for worker in workers:
        worker.join(timeout=120)
    for worker in workers:
        if worker.is_alive():  # pragma: no cover - hang safety
            worker.terminate()
            pytest.fail("gloo worker hung")
    while not queue.empty():
        rank, message = queue.get()
        results[rank] = message
    if len(results) < 2 and all(w.exitcode != 0 for w in workers):
        pytest.skip(f"gloo spawn unavailable in this environment: {results}")
    assert results == {0: "ok", 1: "ok"}, results
