import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Kimi-K3 TP8: 896 experts, top-16, latent 3584, hidden 7168, up to 8 bank rows.
E, K, LATENT, H, NB = 896, 16, 3584, 7168, 9
BLOCK_M = 32


def _routes(out, num_tokens):
    """{(token, slot): (row, weight)} over the valid rows; padding rows carry weight 0."""
    pad = (K << 24) | num_tokens
    ids, weights = out.sorted_ids.cpu(), out.sorted_weights.cpu()
    routes = {}
    for row in range(int(out.num_valid_ids[0])):
        v = int(ids[row])
        if v == pad:
            assert float(weights[row]) == 0.0
            continue
        routes[(v & 0xFFFFFF, v >> 24)] = (row, float(weights[row]))
    return routes


@unittest.skipIf(torch.version.hip is None, "ROCm route sort")
class TestKimiK3RouteSort(CustomTestCase):
    @torch.inference_mode()
    def test_route_direct_layout_carries_every_route_once(self):
        """The route-direct layout gives route t * K + k block t * K + k with the
        route's expert, token, slot and weight in its first row, and its fused
        stage-1 activation scales; the routes are those of the expert-grouped layout."""
        from sglang.kernels.ops.moe import kimi_k3_route_sort as rs

        g = torch.Generator(device="cuda").manual_seed(0)
        bias = torch.randn(E, generator=g, device="cuda") * 0.1
        for num_tokens in (1, 4, 8):
            logits = torch.randn(num_tokens, E, generator=g, device="cuda").bfloat16()
            x = torch.randn(num_tokens, LATENT, generator=g, device="cuda").bfloat16()
            outs = [
                rs.route_sort(
                    logits,
                    bias,
                    topk=K,
                    block_m=BLOCK_M,
                    model_dim=LATENT,
                    moe_buf=None,
                    routed_scaling_factor=1.0,
                    renormalize=True,
                    quant_input=x,
                    direct=direct,
                )
                for direct in (False, True)
            ]
            grouped, direct = outs
            with self.subTest(num_tokens=num_tokens):
                self.assertTrue(torch.equal(direct.topk_ids, grouped.topk_ids))
                self.assertEqual(
                    direct.num_valid_ids.tolist(),
                    [num_tokens * K * BLOCK_M, num_tokens],
                )
                got, want = _routes(direct, num_tokens), _routes(grouped, num_tokens)
                self.assertEqual(set(got), set(want))
                ids = direct.topk_ids.flatten().cpu()
                experts = direct.sorted_expert_ids.cpu()
                scales_d = direct.a1_scale.view(torch.uint8).cpu()
                scales_g = grouped.a1_scale.view(torch.uint8).cpu()
                for (t, k), (row, weight) in got.items():
                    r = t * K + k
                    self.assertEqual(row, r * BLOCK_M)
                    self.assertEqual(int(experts[r]), int(ids[r]))
                    self.assertEqual(weight, want[(t, k)][1])
                    self.assertTrue(
                        torch.equal(
                            _row_scales(scales_d, row),
                            _row_scales(scales_g, want[(t, k)][0]),
                        )
                    )

    @torch.inference_mode()
    def test_bank_partials(self):
        """The spare workgroups' partials are the bank's softmax partial sums the
        fused reads merge: per row and score vector, m = max score, sum =
        sum exp(score - m), acc = sum exp(score - m) * bank row, score =
        <bank row, cw> / rms(bank row); a second vector only when given."""
        from sglang.kernels.ops.moe import kimi_k3_route_sort as rs

        g = torch.Generator(device="cuda").manual_seed(1)
        bias = torch.randn(E, generator=g, device="cuda") * 0.1
        acc = torch.zeros(2, 64, H, device="cuda")
        ms = torch.zeros(2, 64, 2, device="cuda")
        eps = 1e-6
        for num_tokens, nvb, sides in ((1, 2, 2), (4, 8, 2), (16, 5, 1)):
            logits = torch.randn(num_tokens, E, generator=g, device="cuda").bfloat16()
            bank = (
                torch.randn(num_tokens, NB, H, generator=g, device="cuda") * 3
            ).bfloat16()
            cws = [
                torch.randn(H, generator=g, device="cuda") * 0.05 for _ in range(sides)
            ]
            acc.zero_()
            ms.zero_()
            rs.route_sort(
                logits,
                bias,
                topk=K,
                block_m=BLOCK_M,
                model_dim=LATENT,
                moe_buf=None,
                routed_scaling_factor=1.0,
                renormalize=True,
                partials=rs.BankPartials(
                    bank=bank,
                    nvb=nvb,
                    score_weights=(cws[0], cws[1] if sides == 2 else None),
                    score_eps=eps,
                    acc=acc,
                    ms=ms,
                ),
            )
            b = bank[:, :nvb].float()
            rms = torch.sqrt((b * b).mean(-1) + eps)
            with self.subTest(num_tokens=num_tokens, nvb=nvb, sides=sides):
                for side, cw in enumerate(cws):
                    score = (b * cw).sum(-1) / rms
                    m = score.max(-1).values
                    e = torch.exp(score - m[:, None])
                    want = (e[:, :, None] * b).sum(1)
                    torch.testing.assert_close(
                        acc[side, :num_tokens], want, rtol=1e-4, atol=1e-3
                    )
                    torch.testing.assert_close(
                        ms[side, :num_tokens],
                        torch.stack([m, e.sum(-1)], -1),
                        rtol=1e-4,
                        atol=1e-4,
                    )
                if sides == 1:
                    self.assertEqual(acc[1].abs().max().item(), 0.0)


def _row_scales(scales: torch.Tensor, row: int) -> torch.Tensor:
    """The e8m0 group scales of a sorted row in aiter's shuffled layout."""
    n_pad = scales.shape[1]
    flat = scales.flatten()
    idx = [
        (row // 32 * n_pad) * 32
        + (g // 8) * 256
        + (g % 4) * 64
        + (row % 16) * 4
        + (g % 8) // 4 * 2
        + (row % 32) // 16
        for g in range(LATENT // 32)
    ]
    return flat[idx]


if __name__ == "__main__":
    unittest.main()
