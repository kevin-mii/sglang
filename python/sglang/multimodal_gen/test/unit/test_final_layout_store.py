# SPDX-License-Identifier: Apache-2.0
"""Final-layout artifact store: publish/bind roundtrip and trust rules."""

import torch

from sglang.multimodal_gen.runtime.loader import final_layout_store
from sglang.multimodal_gen.runtime.loader.checkpoint_mmap import (
    bind_checkpoint_mmap_views,
    build_final_layout_plan,
)


class _TinyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Linear(8, 16)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(3)])
        self.register_buffer("scale", torch.full((16,), 2.0))
        self.register_buffer("cache", torch.zeros(4), persistent=False)


def _sources(tmp_path):
    source = tmp_path / "model-00001.safetensors"
    source.write_bytes(b"x" * 128)
    return [str(source)]


def test_publish_then_bind_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        final_layout_store.envs, "SGLANG_DIFFUSION_CACHE_ROOT", str(tmp_path)
    )
    torch.manual_seed(0)
    producer = _TinyDiT()
    identity = final_layout_store.build_identity(producer, _sources(tmp_path))

    assert final_layout_store.artifact_files(identity) is None
    assert final_layout_store.publish_final_layout_artifact(producer, identity)
    files = final_layout_store.artifact_files(identity)
    assert files is not None

    with torch.device("meta"):
        consumer = _TinyDiT()
    plan, reason = build_final_layout_plan(consumer, files)
    assert reason is None, reason
    assert plan.final_layout
    bind_checkpoint_mmap_views(consumer, plan)

    for name, tensor in producer.state_dict().items():
        if name == "cache":
            continue  # non-persistent: not stored, stays for post-load init
        bound = consumer.state_dict()[name]
        assert bound.device.type == "cpu"
        assert torch.equal(bound, tensor), name


def test_publish_is_idempotent_and_identity_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(
        final_layout_store.envs, "SGLANG_DIFFUSION_CACHE_ROOT", str(tmp_path)
    )
    model = _TinyDiT()
    identity = final_layout_store.build_identity(model, _sources(tmp_path))
    assert final_layout_store.publish_final_layout_artifact(model, identity)
    assert final_layout_store.publish_final_layout_artifact(model, identity)

    # A different source fingerprint must not trust the existing artifact.
    other = tmp_path / "model-00002.safetensors"
    other.write_bytes(b"y" * 256)
    other_identity = final_layout_store.build_identity(model, [str(other)])
    assert final_layout_store.artifact_files(other_identity) is None


def test_incomplete_artifact_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        final_layout_store.envs, "SGLANG_DIFFUSION_CACHE_ROOT", str(tmp_path)
    )
    model = _TinyDiT()
    identity = final_layout_store.build_identity(model, _sources(tmp_path))
    assert final_layout_store.publish_final_layout_artifact(model, identity)
    files = final_layout_store.artifact_files(identity)

    class _Bigger(_TinyDiT):
        def __init__(self):
            super().__init__()
            self.extra = torch.nn.Linear(2, 2)

    with torch.device("meta"):
        consumer = _Bigger()
    plan, reason = build_final_layout_plan(consumer, files)
    assert plan is None
    assert "missing from the artifact" in reason


def test_lease_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(
        final_layout_store.envs, "SGLANG_DIFFUSION_CACHE_ROOT", str(tmp_path)
    )
    model = _TinyDiT()
    identity = final_layout_store.build_identity(model, _sources(tmp_path))
    first = final_layout_store.FinalLayoutLease(identity)
    second = final_layout_store.FinalLayoutLease(identity)
    assert first.acquire()
    try:
        # flock is per-fd, so a second open descriptor contends even within
        # one process.
        assert not second.acquire()
    finally:
        first.release()
    assert second.acquire()
    second.release()
