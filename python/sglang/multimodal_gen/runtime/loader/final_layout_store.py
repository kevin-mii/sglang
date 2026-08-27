# SPDX-License-Identifier: Apache-2.0
"""Node-local store of finalized DiT weights for DLO host backing.

Direct checkpoint-mmap can only represent renames and per-tensor view
transforms, so models whose load path changes bytes or fuses tensors (merged
qkv/gate_up, ``preprocess_loaded_state_dict``) fall back to the ordinary
loader on EVERY rank — each one materializes the full checkpoint in anonymous
host memory next to its offload copies. This store runs that canonical load
once per node: the flock winner publishes the model's finalized tensors as a
safetensors artifact, and every other rank (and every later process) binds
file-backed views of the artifact through the existing checkpoint-mmap
machinery. Artifact tensors are in runtime layout under runtime names, so the
plan is 1:1 by construction. Mirrors vLLM-Omni's Host Weight Runtime
(``vllm_omni/host_weight_runtime/``) reduced to the diffusion-DiT case.

Publication is atomic: shards are written to a temp directory that is renamed
into place before the READY marker is written. Readers only trust artifacts
whose READY.json matches the expected identity.
"""

import fcntl
import hashlib
import json
import os
import time
from typing import Dict, List

import msgspec
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Identity salt; bump when the artifact layout or trust rules change.
_STORE_VERSION = "v1"
_SHARD_BYTES = 8 << 30
_READY_NAME = "READY.json"


class FinalLayoutIdentity(msgspec.Struct, frozen=True):
    model_class: str
    source_files: tuple
    store_version: str = _STORE_VERSION

    def digest(self) -> str:
        payload = json.dumps(
            {
                "model_class": self.model_class,
                "source_files": list(self.source_files),
                "store_version": self.store_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:24]


def build_identity(
    model: torch.nn.Module, safetensors_files: List[str]
) -> FinalLayoutIdentity:
    """Source paths carry the snapshot revision; sizes catch same-path edits."""
    sources = tuple(
        (os.path.abspath(path), os.path.getsize(path))
        for path in sorted(safetensors_files)
    )
    return FinalLayoutIdentity(
        model_class=type(model).__qualname__, source_files=sources
    )


def _store_root() -> str:
    return os.path.join(envs.SGLANG_DIFFUSION_CACHE_ROOT, "final_layout_store")


def artifact_dir(identity: FinalLayoutIdentity) -> str:
    return os.path.join(_store_root(), identity.digest())


def artifact_files(identity: FinalLayoutIdentity) -> List[str] | None:
    """The artifact's shard paths, or None unless READY and identity-matched."""
    directory = artifact_dir(identity)
    ready_path = os.path.join(directory, _READY_NAME)
    try:
        with open(ready_path) as f:
            ready = json.load(f)
    except (OSError, ValueError):
        return None
    if ready.get("identity_digest") != identity.digest():
        return None
    files = [os.path.join(directory, name) for name in ready.get("shards", [])]
    if not files or not all(os.path.exists(path) for path in files):
        return None
    return files


def publish_final_layout_artifact(
    model: torch.nn.Module, identity: FinalLayoutIdentity
) -> bool:
    """Write the model's finalized tensors as a READY artifact (idempotent).

    Tensors are copied to CPU one shard at a time, so the publisher's host
    transient stays bounded by the shard size rather than the model size.
    """
    from sglang.multimodal_gen.runtime.loader.checkpoint_mmap import (
        _required_tensors,
    )

    if artifact_files(identity) is not None:
        return True
    directory = artifact_dir(identity)
    tmp_dir = f"{directory}.tmp.{os.getpid()}"
    os.makedirs(tmp_dir, exist_ok=True)
    started_at = time.perf_counter()

    sources = _required_tensors(model)
    shards: List[str] = []
    shard: Dict[str, torch.Tensor] = {}
    shard_bytes = 0

    def flush() -> None:
        nonlocal shard, shard_bytes
        if not shard:
            return
        name = f"final-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(tmp_dir, name))
        shards.append(name)
        shard = {}
        shard_bytes = 0

    for name in sorted(sources):
        tensor = sources[name].detach().to("cpu").contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if shard and shard_bytes + nbytes > _SHARD_BYTES:
            flush()
        shard[name] = tensor
        shard_bytes += nbytes
    flush()

    ready = {
        "identity_digest": identity.digest(),
        "model_class": identity.model_class,
        "shards": shards,
        "tensor_count": len(sources),
    }
    with open(os.path.join(tmp_dir, f"{_READY_NAME}.tmp"), "w") as f:
        json.dump(ready, f)
        f.flush()
        os.fsync(f.fileno())

    # Rename the populated directory into place, then arm READY inside it; a
    # crash in between leaves a directory without READY, which readers ignore.
    if os.path.isdir(directory):
        # A concurrent publisher won the rename; trust its artifact.
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return artifact_files(identity) is not None
    os.rename(tmp_dir, directory)
    os.rename(
        os.path.join(directory, f"{_READY_NAME}.tmp"),
        os.path.join(directory, _READY_NAME),
    )
    logger.info(
        "Published final-layout artifact for %s: %d tensors, %d shard(s) in "
        "%.1fs at %s",
        identity.model_class,
        len(sources),
        len(shards),
        time.perf_counter() - started_at,
        directory,
    )
    return True


class FinalLayoutLease:
    """flock-based single-publisher lease for one artifact identity.

    ``acquire`` returns True for the rank that should run the canonical load
    and publish; every other caller polls ``artifact_files`` until READY.
    """

    def __init__(self, identity: FinalLayoutIdentity):
        os.makedirs(_store_root(), exist_ok=True)
        self._path = os.path.join(_store_root(), f"{identity.digest()}.lock")
        self._fd: int | None = None

    def acquire(self) -> bool:
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


def wait_for_artifact(
    identity: FinalLayoutIdentity, timeout_s: float
) -> List[str] | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        files = artifact_files(identity)
        if files is not None:
            return files
        time.sleep(2.0)
    return None
