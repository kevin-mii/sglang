"""The fused fp8 MiniMax-M3 sparse-cache store must write the unfused stores' bytes."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.kvcache.minimax_store_kv_index_quant import (
    can_store_kv_index_quant,
)
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

DEV = "cuda"
SLOTS = 256
HEAD_DIM = 128
DENSE_LAYER, K_ONLY_LAYER, KV_INDEX_LAYER = 0, 1, 2
FP8_DTYPES = [torch.float8_e4m3fn] + ([torch.float8_e4m3fnuz] if is_hip() else [])
# Unit K/V scales and no index scales are what M3 checkpoints without KV scales
# pass; the last set is not powers of two, so the division's rounding order shows.
SCALE_SETS = [
    (None, None, None, None),
    (1.0, 1.0, None, None),
    (0.37, 1.7, 0.61, 3.1),
]


def _make_pool(dtype, index_dtype, head_num, fused):
    pool = MiniMaxSparseKVPool(
        size=SLOTS,
        page_size=1,
        dtype=dtype,
        index_dtype=index_dtype,
        head_num=head_num,
        head_dim=HEAD_DIM,
        idx_head_dim=HEAD_DIM,
        dense_layer_ids=[DENSE_LAYER],
        sparse_layer_ids=[K_ONLY_LAYER, KV_INDEX_LAYER],
        disable_value_sparse_layer_ids=[K_ONLY_LAYER],
        device=DEV,
        start_layer=0,
        end_layer=3,
    )
    pool.use_minimax_fused_kv_index_store = fused
    return pool


def _row(num_tokens, head_num, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    width = 3 * head_num * HEAD_DIM + 2 * HEAD_DIM
    row = torch.randn(num_tokens, width, generator=g, device=DEV) * 8
    return row.to(torch.bfloat16)


def _split(row, head_num):
    # Strided views of one wide row, like the qkv / index projection splits;
    # V gets its own token stride so a K/V stride mix-up shows.
    T, hd = row.shape[0], head_num * HEAD_DIM
    k = row[:, :hd].view(T, head_num, HEAD_DIM)
    v = row[:, hd : 2 * hd].reshape(T, head_num, HEAD_DIM).contiguous()
    idx_k = row[:, 3 * hd : 3 * hd + HEAD_DIM].view(T, 1, HEAD_DIM)
    idx_v = row[:, 3 * hd + HEAD_DIM :].view(T, 1, HEAD_DIM)
    return k, v, idx_k, idx_v


def _store(pool, layer_id, loc, row, head_num, scales):
    # The unfused stores divide K/V in place, so each pool gets its own copy.
    k, v, idx_k, idx_v = _split(row.clone(), head_num)
    pool.set_fused_kv_index_buffer(
        SimpleNamespace(layer_id=layer_id),
        loc,
        k,
        v,
        idx_k,
        None if layer_id == K_ONLY_LAYER else idx_v,
        *scales,
    )


def _caches(pool, layer_id):
    k_cache, v_cache = pool.get_kv_buffer(layer_id)
    if layer_id == K_ONLY_LAYER:
        return [k_cache, v_cache, pool.get_index_k_buffer(layer_id)]
    return [k_cache, v_cache, *pool.get_index_kv_buffer(layer_id)]


@pytest.mark.parametrize(
    "main_dtype,index_dtype",
    [(d, torch.float8_e4m3fn) for d in FP8_DTYPES]
    + [(d, torch.bfloat16) for d in FP8_DTYPES]
    + [(torch.bfloat16, torch.float8_e4m3fn)],
)
@pytest.mark.parametrize("head_num", [1, 2])
@pytest.mark.parametrize("layer_id", [K_ONLY_LAYER, KV_INDEX_LAYER])
@pytest.mark.parametrize("scales", SCALE_SETS)
@pytest.mark.parametrize(
    "num_tokens,num_pad,loc_dtype", [(1, 0, torch.int64), (200, 7, torch.int32)]
)
def test_fused_store_matches_unfused_pool(
    main_dtype, index_dtype, head_num, layer_id, scales, num_tokens, num_pad, loc_dtype
):
    """A wrong stride, slot, scale or rounding order would corrupt cache rows silently."""
    fused = _make_pool(main_dtype, index_dtype, head_num, fused=True)
    unfused = _make_pool(main_dtype, index_dtype, head_num, fused=False)

    total = num_tokens + num_pad
    row = _row(total, head_num, seed=total * 7 + head_num)
    k, _, idx_k, _ = _split(row, head_num)
    assert fused._can_fuse_kv_index_store_quant(
        fused.index_k_pool if layer_id == K_ONLY_LAYER else fused.index_kv_pool,
        k,
        idx_k,
    )
    # Scattered real slots, then cuda-graph padding tokens that all write slot 0.
    real = torch.randperm(SLOTS, device=DEV)[:num_tokens] + 1
    pad = torch.zeros(num_pad, dtype=real.dtype, device=DEV)
    loc = torch.cat([real, pad]).to(loc_dtype)

    _store(fused, layer_id, loc, row, head_num, scales)
    _store(unfused, layer_id, loc, row, head_num, scales)
    torch.cuda.synchronize()

    # Several padding tokens race on slot 0, so its bytes are only defined for one.
    first_slot = 1 if num_pad > 1 else 0
    for name, got, want in zip(
        ("k", "v", "idx_k", "idx_v"),
        _caches(fused, layer_id),
        _caches(unfused, layer_id),
    ):
        got_bytes = got[first_slot:].view(torch.uint8)
        want_bytes = want[first_slot:].view(torch.uint8)
        mismatched = (got_bytes != want_bytes).any(dim=-1).sum().item()
        assert mismatched == 0, (name, mismatched)
    # Untouched layers stay zero.
    for cache in _caches(fused, KV_INDEX_LAYER + K_ONLY_LAYER - layer_id):
        assert not cache.view(torch.uint8).any()


def test_layouts_the_kernel_cannot_address_fall_back():
    """V and index-V need the same layout checks as K, or the kernel misreads them."""
    T, H, D = 4, 2, HEAD_DIM
    bf16 = dict(dtype=torch.bfloat16, device=DEV)
    fp8 = dict(dtype=torch.float8_e4m3fn, device=DEV)
    k = torch.randn(T, H, D, **bf16)
    v = torch.randn(T, H, D, **bf16)
    idx_k = torch.randn(T, 1, D, **bf16)
    cache = torch.zeros(SLOTS, H, D, **fp8)
    idx_cache = torch.zeros(SLOTS, 1, D, **fp8)

    def storable(v_=v, vc=cache, iv=None, ivc=None):
        return can_store_kv_index_quant(k, v_, cache, vc, idx_k, idx_cache, iv, ivc)

    assert storable()
    assert storable(iv=idx_k, ivc=idx_cache)
    assert not storable(
        v_=torch.randn(T, H, 96, **bf16), vc=torch.zeros(SLOTS, H, 96, **fp8)
    )
    assert not storable(v_=torch.randn(T, D, H, **bf16).transpose(1, 2))
    assert not storable(v_=v[:, :1], vc=cache[:, :1])
    assert not storable(iv=idx_k)
    assert not storable(ivc=idx_cache)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
