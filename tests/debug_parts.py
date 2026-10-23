import torch, mla_fp8
DEV = "cuda"; FP8 = torch.float8_e4m3fn
def call(q, k, v, nq, nk, causal=False):
    cq = torch.tensor([0, nq], dtype=torch.int32); ck = torch.tensor([0, nk], dtype=torch.int32)
    return mla_fp8.fwd_varlen(q, k, v, cq.to(DEV), ck.to(DEV), nq, nk, causal, 192**-0.5, host_cu_q=cq, host_cu_k=ck).float()
torch.manual_seed(0)
nq, nk, H = 32, 64, 1
# (a) Q = 0: uniform P; O = mean over keys of V.
q = torch.zeros(nq, H, 192, device=DEV, dtype=torch.bfloat16)
k = torch.randn(nk, H, 192, device=DEV).to(FP8)
v = torch.randn(nk, H, 128, device=DEV).to(FP8)
o = call(q, k, v, nq, nk)
ref = v.float().mean(0, keepdim=True).expand(nq, H, 128)
print("(a) uniform P: max err", (o - ref).abs().max().item(), " o[0,0,:6]", o[0, 0, :6].tolist(), " ref", ref[0, 0, :6].tolist())
# (b) V = identity (d = kv), Q/K random: O[q][kv] = P[q][kv].
v = torch.zeros(nk, H, 128, device=DEV); v[torch.arange(nk), 0, torch.arange(nk)] = 1.0; v = v.to(FP8)
q = torch.randn(nq, H, 192, device=DEV).to(torch.bfloat16) * 0.3
k = torch.randn(nk, H, 192, device=DEV).to(FP8)
o = call(q, k, v, nq, nk)
s = (q.float()[:, 0] @ k.float()[:, 0].T) * 192**-0.5
p = torch.softmax(s, -1)
print("(b) P via identity V: max err", (o[:, 0, :nk] - p).abs().max().item())
print("  row 0 kernel:", [round(x, 3) for x in o[0, 0, :12].tolist()])
print("  row 0 ref   :", [round(x, 3) for x in p[0, :12].tolist()])
# best matching permutation hint: for each kernel column j, which ref column matches best on all rows
o_p = o[:, 0, :nk]
match = [int(((o_p[:, j:j+1] - p).abs().sum(0)).argmin()) for j in range(nk)]
print("  kernel col j ~ ref kv:", match[:32])
