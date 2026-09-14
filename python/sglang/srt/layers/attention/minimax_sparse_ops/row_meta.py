"""Per-row metadata for MiniMax-M3 sparse forwards that flatten a batch into one query row per token."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple

import torch


def chain_verify_row_meta(
    prefix_lens: torch.Tensor, req_pool_indices: torch.Tensor, num_draft_tokens: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(per_query_req, per_query_seq_lens)`` for an EAGLE chain-verify batch.

    Row ``j`` of a request attends ``KV[0 : prefix + j + 1]``; device ops only,
    so the result is capture-safe.
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


def flattened_extend_row_meta(
    req_pool_indices: torch.Tensor,
    prefix_lens: list[int],
    extend_lens: list[int],
    seq_lens_dtype: torch.dtype,
) -> SimpleNamespace:
    """Per-row metadata for a small EXTEND served as flattened decode rows.

    Row ``j`` of a request attends ``KV[0 : prefix + j + 1]``; ``packed`` is the
    per-request row count when it is constant across the batch, else 1.
    """
    ext = [int(x) for x in extend_lens]
    prefix = [int(x) for x in prefix_lens]
    dev = req_pool_indices.device
    per_query_req = req_pool_indices.repeat_interleave(
        torch.tensor(ext, dtype=torch.int64, device=dev)
    )
    per_query_seq_lens = torch.tensor(
        [p + j + 1 for p, e in zip(prefix, ext) for j in range(e)],
        dtype=seq_lens_dtype,
        device=dev,
    )
    return SimpleNamespace(
        per_query_req=per_query_req,
        per_query_seq_lens=per_query_seq_lens,
        max_seqlen=max(p + e for p, e in zip(prefix, ext)),
        packed=ext[0] if all(e == ext[0] for e in ext) else 1,
        rows=sum(ext),
    )
