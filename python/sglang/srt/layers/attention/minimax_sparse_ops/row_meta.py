"""Per-row metadata for MiniMax-M3 sparse attention forwards that flatten a
batch into one query row per token (EAGLE chain verify, small extends).

Pure tensor helpers with no engine imports so they are unit-testable on CPU.
"""

from __future__ import annotations

from typing import Tuple

import torch


def chain_verify_row_meta(
    prefix_lens: torch.Tensor, req_pool_indices: torch.Tensor, num_draft_tokens: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row metadata for an EAGLE chain verify batch (``topk == 1``).

    Every request contributes ``num_draft_tokens`` draft rows; row ``j`` of a
    request attends ``KV[0 : prefix + j + 1]``. Returns ``(per_query_req,
    per_query_seq_lens)`` with ``bs * num_draft_tokens`` entries, request-major,
    built from device ops only so it is safe under graph capture.
    """
    ndt = int(num_draft_tokens)
    offsets = torch.arange(1, ndt + 1, device=prefix_lens.device, dtype=torch.long)
    per_query_seq_lens = (
        (prefix_lens.to(torch.long).unsqueeze(1) + offsets.unsqueeze(0))
        .reshape(-1)
        .to(torch.int32)
    )
    per_query_req = req_pool_indices.long().repeat_interleave(ndt)
    return per_query_req, per_query_seq_lens
