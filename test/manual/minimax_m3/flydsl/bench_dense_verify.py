"""AITER FlyDSL paged attention on MiniMax-M3 dense-layer EAGLE verify shapes (query_length = draft tokens).

TP4 rank: 16 query heads, 1 KV head, head_dim 128, fp8 KV in the SHUFFLE layout, page 16, uneven long
contexts (same lengths as verify_mla_splits.py). Times static partitions and the GPU work planner, and
checks one batch against an fp32 reference. Run on the AITER #4332/#5546 stack.
"""

import torch
from aiter.ops.flydsl.pa_decode import get_recommended_splits, pa_decode, plan_pa_decode

H_Q, D, PAGE, X, NDT = 16, 128, 16, 16, 4


def lengths_for(bs):
    g = torch.Generator().manual_seed(bs)
    return (torch.exp(torch.randn(bs, generator=g) * 0.6) * 150_000).clamp(16_384, 400_000).int().tolist()


def build(bs, prefix_lens, device="cuda"):
    lens = [p + NDT for p in prefix_lens]
    pages = [(l + PAGE - 1) // PAGE for l in lens]
    nblocks = sum(pages)
    k = (torch.randn(nblocks * PAGE, 1, D, device=device) * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(nblocks * PAGE, 1, D, device=device) * 0.5).to(torch.float8_e4m3fn)
    key_cache = k.view(nblocks, PAGE, 1, D // X, X).permute(0, 2, 3, 1, 4).contiguous()
    value_cache = v.view(nblocks, PAGE // X, X, 1, D).permute(0, 3, 1, 4, 2).contiguous()
    perm = torch.randperm(nblocks, device=device, dtype=torch.int32)
    table = torch.zeros(bs, max(pages), dtype=torch.int32, device=device)
    off = 0
    for i, n in enumerate(pages):
        table[i, :n] = perm[off : off + n]
        off += n
    q = torch.randn(bs * NDT, H_Q, D, device=device, dtype=torch.bfloat16)
    ctx = torch.tensor(lens, dtype=torch.int32, device=device)
    return q, key_cache, value_cache, ctx, table, k, v


def reference(q, k, v, lens, table, row):
    # request `row`, causal over the NDT query tokens at the end of its context
    L = lens[row]
    pages = table[row, : (L + PAGE - 1) // PAGE].long()
    tok = (pages[:, None] * PAGE + torch.arange(PAGE, device=q.device)).reshape(-1)[:L]
    kk, vv = k[tok, 0].float(), v[tok, 0].float()
    out = []
    for t in range(NDT):
        qt = q[row * NDT + t].float()
        s = (qt @ kk.T) * D**-0.5
        s[:, L - NDT + t + 1 :] = float("-inf")
        out.append(torch.softmax(s, -1) @ vv)
    return torch.stack(out)


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
    for bs in (8, 16, 24, 32, 48):
        prefix = lengths_for(bs)
        q, kc, vc, ctx, table, k, v = build(bs, prefix)
        out = torch.empty_like(q)
        splits = get_recommended_splits(bs, 1)
        static = lambda: pa_decode(output=out, query=q, key_cache=kc, value_cache=vc, context_lengths=ctx,
                                   block_tables=table, softmax_scale=D**-0.5, query_length=NDT,
                                   max_context_partition_num=splits, compute_type=kc.dtype)
        plan = plan_pa_decode(ctx, 1, query_length=NDT)
        planned = lambda: pa_decode(output=out, query=q, key_cache=kc, value_cache=vc, context_lengths=ctx,
                                    block_tables=table, softmax_scale=D**-0.5, query_length=NDT,
                                    max_context_partition_num=plan.max_partitions, compute_type=kc.dtype,
                                    work_plan=plan)
        row = {"bs": bs, "ctx_mean_k": sum(prefix) / bs / 1000}
        for name, fn in (("static", static), ("planned", planned)):
            fn(); torch.cuda.synchronize()
            ref = reference(q, k, v, ctx.tolist(), table, 0)
            got = out[:NDT].float().view(NDT, H_Q, D)
            row[f"{name}_maxdiff"] = float((got - ref).abs().max())
            row[f"{name}_us"] = round(time_us(fn), 1)
        print(row, flush=True)


if __name__ == "__main__":
    main()
