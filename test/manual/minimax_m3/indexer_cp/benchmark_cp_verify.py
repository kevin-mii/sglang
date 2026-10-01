"""Four-rank indexer CP vs the native packed EAGLE verify path: selected-ID parity and graph timing.

EAGLE chain verify scores ``ndt`` query rows per request: row ``i`` of a request sees the prefix plus
draft tokens ``0..i``. The native path scores them as one packed group per request
(``packed_queries=ndt``); indexer CP scores each row as an ordinary decode query.

    torchrun --nproc_per_node=4 test/manual/minimax_m3/indexer_cp/benchmark_cp_verify.py --output out.json
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from benchmark_cp import capture, exact, setup, timing


def verify_inputs(reqs, prefix, ndt, dtype, rank):
    torch.manual_seed(20260929)
    device = torch.device("cuda", rank)
    ctx = prefix + ndt
    padded_len = (ctx + 127) // 128 * 128
    nslots = reqs * padded_len
    pages = torch.randperm(nslots // 16, device=device, dtype=torch.int32)
    table = (
        (pages[:, None] * 16 + torch.arange(16, device=device))
        .reshape(reqs, padded_len)
        .to(torch.int32)
    )
    cache = torch.randn((nslots, 1, 128), device=device, dtype=torch.bfloat16).to(dtype)
    all_q = torch.randn((reqs * ndt, 4, 128), device=device, dtype=torch.bfloat16)
    lengths = (
        torch.full((reqs, 1), prefix, dtype=torch.int64, device=device)
        + torch.arange(1, ndt + 1, device=device)
    ).reshape(-1)
    slots = torch.arange(reqs, dtype=torch.int64, device=device).repeat_interleave(ndt)
    return all_q[:, rank : rank + 1].contiguous(), cache, table, slots, lengths, ctx


def functions(cp, data, ndt):
    from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
        flash_decode_with_topk_idx,
    )

    q, cache, table, slots, lengths, ctx = data

    def native(pack):
        return lambda: flash_decode_with_topk_idx(
            q=q,
            sink=None,
            k_cache=cache,
            v_cache=None,
            req_to_token=table,
            seq_lens=lengths,
            max_seqlen=ctx,
            slot_ids=slots,
            block_size=128,
            topk=16,
            init_blocks=1,
            local_blocks=2,
            disable_index_value=True,
            packed_queries=pack,
        )[1]

    return {
        "native_packed": native(ndt),
        "native_rows": native(1),
        "indexer_cp": lambda: cp(q, cache, table, slots, lengths, ctx, 1, 2),
        "indexer_cp_packed": lambda: cp(
            q, cache, table, slots, lengths, ctx, 1, 2, packed_queries=ndt
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ndt", type=int, default=4)
    args = parser.parse_args()
    rank, group = setup()
    from sglang.srt.layers.attention.minimax_sparse_ops.indexer_cp import (
        MiniMaxIndexerCP,
    )

    cp = MiniMaxIndexerCP(group)
    cp.warmup()
    rows = []
    for prefix in (8192, 32768, 65536, 131072, 196608):
        for reqs in (1, 8, 16, 24, 32, 48):
            data = verify_inputs(reqs, prefix, args.ndt, torch.float8_e4m3fn, rank)
            fns = functions(cp, data, args.ndt)
            ref = fns["native_packed"]().clone()
            for name in ("native_rows", "indexer_cp", "indexer_cp_packed"):
                got = fns[name]()
                exact(
                    ref.sort(dim=-1).values,
                    got.sort(dim=-1).values,
                    f"{name} {reqs}x{prefix}",
                )
            graphs, _ = capture(group, fns, calls=8)
            t = timing(graphs, calls=8, rounds=5, replays=50)
            row = {
                "reqs": reqs,
                "prefix": prefix,
                "ndt": args.ndt,
                **{k: v["median_us"] for k, v in t.items()},
            }
            rows.append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)
            del graphs, data, fns
            torch.cuda.empty_cache()
    if rank == 0:
        args.output.write_text(json.dumps(rows, indent=1))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
