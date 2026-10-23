import torch, mla_fp8
DEV = "cuda"; FP8 = torch.float8_e4m3fn
def check(nq, nk, causal=False, zero_q=False):
    torch.manual_seed(0)
    q = torch.randn(nq, 1, 192, device=DEV).to(torch.bfloat16) * (0 if zero_q else 0.3)
    k = torch.randn(nk, 1, 192, device=DEV).to(FP8); v = torch.randn(nk, 1, 128, device=DEV).to(FP8)
    cq = torch.tensor([0, nq], dtype=torch.int32); ck = torch.tensor([0, nk], dtype=torch.int32)
    o = mla_fp8.fwd_varlen(q, k, v, cq.to(DEV), ck.to(DEV), nq, nk, causal, 192**-0.5, host_cu_q=cq, host_cu_k=ck).float()[:, 0]
    s = (q.float()[:, 0] @ k.float()[:, 0].T) * 192**-0.5
    if causal: s.masked_fill_(torch.arange(nk, device=DEV)[None] > torch.arange(nq, device=DEV)[:, None] + nk - nq, float("-inf"))
    ref = torch.softmax(s, -1) @ v.float()[:, 0]
    err = ((o - ref).abs().max() / ref.abs().max()).item()
    bad_rows = ((o - ref).abs().amax(1) > 0.1 * ref.abs().max()).nonzero().flatten().tolist()
    print(f"nq={nq} nk={nk} causal={causal} zero_q={zero_q}: err {err:.3e}  bad rows {bad_rows[:8]}{'...' if len(bad_rows) > 8 else ''} ({len(bad_rows)})")
for nq, nk in [(32, 128), (32, 192), (32, 256), (64, 64), (256, 64), (32, 1024)]:
    check(nq, nk, zero_q=True); check(nq, nk)
