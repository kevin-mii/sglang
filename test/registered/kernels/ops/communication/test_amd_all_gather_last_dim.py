"""aiter's last-dim all-gather must lay out each rank's columns as torch's gather does.

Run with ``python test_amd_all_gather_last_dim.py --num-gpu 2``.
"""

import os

import pytest
import torch
import torch.distributed as dist

from sglang.srt.distributed import parallel_state as ps
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

register_amd_ci(est_time=60, stage="sgl-kernel-unit", runner_config="2-gpu-amd")


@pytest.fixture(scope="module")
def group():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ps.set_custom_all_reduce(True)
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=world_size, world_rank=rank)
    ps.initialize_model_parallel()
    yield ps.get_tp_group()
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def _reference(group, shard):
    shards = [torch.empty_like(shard) for _ in range(group.world_size)]
    dist.all_gather(shards, shard, group=group.device_group)
    return torch.cat(shards, dim=-1)


# 32320 columns: a DeepSeek-V4 vocab shard at TP4
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 6, 48])
@pytest.mark.parametrize("use_graph", [False, True])
def test_last_dim_gather_matches_torch(group, dtype, rows, use_graph):
    torch.manual_seed(7 + group.rank_in_group)
    shard = torch.randn(rows, 32320, device="cuda", dtype=dtype)
    out = torch.full(
        (rows, 32320 * group.world_size), float("nan"), device="cuda", dtype=dtype
    )
    if use_graph:
        with group.graph_capture() as capture:
            for _ in range(3):
                group.all_gather_last_dim(shard, out=out)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture.stream):
                gathered = group.all_gather_last_dim(shard, out=out)
        graph.replay()
    else:
        gathered = group.all_gather_last_dim(shard, out=out)
    assert gathered is out, "aiter's last-dim gather did not apply"
    torch.cuda.synchronize()
    torch.testing.assert_close(out, _reference(group, shard), atol=0, rtol=0)
    # all_gather(dim=-1) takes the same path
    torch.testing.assert_close(
        group.all_gather(shard, dim=-1), _reference(group, shard), atol=0, rtol=0
    )


def test_last_dim_gather_declines_unaligned_rows(group):
    # 3 float32 columns: a rank's row chunk is not a 16-byte multiple
    shard = torch.randn(4, 3, device="cuda")
    assert group.all_gather_last_dim(shard) is None
    torch.testing.assert_close(
        group.all_gather(shard, dim=-1), _reference(group, shard), atol=0, rtol=0
    )


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2,))
