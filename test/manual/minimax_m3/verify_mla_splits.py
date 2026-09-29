"""Split-count sweep for the grouped-head split-KV verify kernel on MiniMax-M3 dense-layer shapes.

TP4 rank shapes: 16 query heads, 1 KV head, head_dim 128, fp8 KV, 4 verify rows per request,
uneven long contexts. Compares (target_programs, max_splits) budgets against the default and checks
outputs match it.

    python test/manual/minimax_m3/verify_mla_splits.py
"""

import torch

from sglang.kernels.ops.attention.verify_mla import VerifyMLA, block_config


def make_inputs(lengths, h_q=16, d=128, l_ext=4, device="cuda"):
    bs = len(lengths)
    total = int(sum(lengths))
    k_buf = (torch.randn(total, 1, d, device=device) * 0.5).to(torch.float8_e4m3fn)
    v_buf = (torch.randn(total, 1, d, device=device) * 0.5).to(torch.float8_e4m3fn)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(torch.tensor(lengths, device=device), 0)
    kv_indices = torch.randperm(total, device=device, dtype=torch.int32)
    qo_indptr = torch.arange(0, (bs + 1) * l_ext, l_ext, dtype=torch.int32, device=device)
    q = torch.randn(bs * l_ext, h_q, d, device=device, dtype=torch.bfloat16)
    k_ext = torch.randn(bs * l_ext, 1, d, device=device, dtype=torch.bfloat16)
    v_ext = torch.randn(bs * l_ext, 1, d, device=device, dtype=torch.bfloat16)
    return q, k_ext, v_ext, k_buf, v_buf, qo_indptr, kv_indptr, kv_indices


def time_us(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(4):
            fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000 / (iters * 4)


def main():
    torch.manual_seed(0)
    block_h, block_n, warps = block_config(128)
    configs = [(4096, 64), (8192, 128), (16384, 256), (32768, 512)]
    for bs in (8, 16, 24, 32, 48):
        g = torch.Generator().manual_seed(bs)
        lengths = (torch.exp(torch.randn(bs, generator=g) * 0.6) * 150_000).clamp(16_384, 400_000).int().tolist()
        data = make_inputs(lengths)
        q, k_ext, v_ext, k_buf, v_buf, qo, kvp, kvi = data
        outs, row = {}, {"bs": bs, "ctx_mean_k": sum(lengths) / bs / 1000, "ctx_max_k": max(lengths) / 1000}
        for tp, ms in configs:
            vk = VerifyMLA(64, 16, 128, 128, 4, block_h=block_h, block_n=block_n, num_warps=warps,
                           kv_group_num=16, target_programs=tp, max_splits=ms)
            fn = lambda: vk(q, k_ext, v_ext, k_buf, v_buf, qo, kvp, kvi, 128**-0.5)
            outs[(tp, ms)] = fn().float()
            row[f"{tp}/{ms}"] = round(time_us(fn), 1)
        ref = outs[configs[0]]
        row["max_abs_diff"] = max(float((o - ref).abs().max()) for o in outs.values())
        print(row, flush=True)


if __name__ == "__main__":
    main()
