# 4-rank all-reduce check, eager and under graph capture: python qr_test.py QUANT
import os, sys, torch, torch.multiprocessing as mp
def run(rank, ws, port, q):
    os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = q
    from sglang.srt.distributed import init_distributed_environment
    from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce as ar
    from sglang.srt.distributed.parallel_state import initialize_model_parallel, graph_capture, get_tp_group
    from sglang.test.test_utils import publish_build_topology
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=ws, rank=rank, distributed_init_method=f"tcp://localhost:{port}", local_rank=rank)
    publish_build_topology(tp_size=ws, world_rank=rank); initialize_model_parallel()
    d = torch.zeros(1, device="cuda"); torch.distributed.all_reduce(d, group=get_tp_group().device_group); torch.cuda.synchronize()
    for sz in [1 << 12, 1 << 20, 1 << 23, 1 << 25, 1 << 26]:
        for mode in ("eager", "graph"):
            try:
                g = torch.Generator(device="cuda").manual_seed(sz)
                x = torch.randint(1, 23, (sz,), dtype=torch.bfloat16, device="cuda", generator=g)
                if mode == "eager":
                    y = ar(x.clone())
                else:
                    with graph_capture() as ctx:
                        xi = x.clone(); cg = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(cg, stream=ctx.stream): y = ar(xi)
                    cg.replay()
                torch.cuda.synchronize()
                err = (y.float() - x.float() * ws).abs().max().item()
                if rank == 0: print(f"{q} {mode} {sz*2/2**20:.2f} MB maxerr={err}", flush=True)
            except Exception as e:
                print(f"rank{rank} {q} {mode} {sz*2/2**20:.2f} MB FAIL {e}", flush=True); return
if __name__ == "__main__":
    mp.spawn(run, args=(4, 29511, sys.argv[1]), nprocs=4)
