import unittest

import torch

from sglang.kernels.ops.attention.mla_absorbed_extend import (
    mla_absorbed_extend,
    plan_absorbed_extend,
)
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Kimi-K3 TP8 MLA: 12 heads per rank, 512-wide latent, 64-wide rope.
H, R, P = 12, 512, 64
FP8 = torch.float8_e4m3fn


def _reference(q, cache, qo_indptr, kv_indptr, kv_indices, scale):
    out = torch.empty(q.shape[0], H, R, device="cuda")
    keys_all = cache.float()
    for b in range(qo_indptr.numel() - 1):
        q0, q1 = qo_indptr[b].item(), qo_indptr[b + 1].item()
        k0, k1 = kv_indptr[b].item(), kv_indptr[b + 1].item()
        keys = keys_all[kv_indices[k0:k1].long()]
        n_q, n_k = q1 - q0, k1 - k0
        s = torch.einsum("qhd,kd->hqk", q[q0:q1].float(), keys) * scale
        pos = torch.arange(n_k - n_q, n_k, device="cuda")[:, None]
        s = s.masked_fill(torch.arange(n_k, device="cuda")[None, :] > pos, float("-inf"))
        out[q0:q1] = torch.einsum("hqk,kd->qhd", torch.softmax(s, -1), keys[:, :R])
    return out


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx95_supported(), "requires gfx950"
)
class TestMLAAbsorbedExtend(unittest.TestCase):
    @torch.inference_mode()
    def test_matches_reference(self):
        """Ragged extends after cached prefixes (and one without) attend every
        prefix key and their own earlier positions only."""
        torch.manual_seed(0)
        scale = 1.0 / 192**0.5
        for prefix_lens, ext_lens in (
            ([1000], [37]),
            ([4000, 0, 2500], [256, 128, 1]),
            ([300, 7000], [5, 200]),
        ):
            kv_lens = [p + e for p, e in zip(prefix_lens, ext_lens)]
            cache = (torch.randn(sum(kv_lens) + 1, R + P, device="cuda") * 0.5).to(FP8)
            kv_indices = (torch.randperm(sum(kv_lens), device="cuda") + 1).int()
            kv_indptr = torch.tensor(
                [0, *torch.tensor(kv_lens).cumsum(0).tolist()],
                device="cuda",
                dtype=torch.int32,
            )
            qo_indptr = torch.tensor(
                [0, *torch.tensor(ext_lens).cumsum(0).tolist()],
                device="cuda",
                dtype=torch.int32,
            )
            q = (torch.randn(sum(ext_lens), H, R + P, device="cuda") * 0.3).to(FP8)
            plan = plan_absorbed_extend(ext_lens, kv_lens, q.device)
            out = mla_absorbed_extend(
                q, cache, kv_indices, kv_indptr, qo_indptr, plan, scale, R
            )
            ref = _reference(q, cache, qo_indptr, kv_indptr, kv_indices, scale)
            cos = torch.nn.functional.cosine_similarity(
                out.float().flatten(), ref.flatten(), dim=0
            ).item()
            with self.subTest(prefix_lens=prefix_lens, ext_lens=ext_lens):
                self.assertGreater(cos, 0.9995)
                torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
