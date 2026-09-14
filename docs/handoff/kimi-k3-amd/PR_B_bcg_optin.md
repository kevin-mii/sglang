# [ROCm] Let aiter MLA extend replay under breakable CUDA graphs with off-CUDA prefixes

## Motivation
Breakable CUDA graph prefill (`--enable-breakable-cuda-graph`-family) refuses to capture MLA extends whose prefix lives off-CUDA unless the attention backend opts in. The aiter MLA backend on ROCm can replay these captures (the prefix chunks are attended by kernels that take caller-provided K/V), but had no opt-in, so K3 prefill graphs were rejected for every radix-hit batch.

## Modifications
- `base_attn_backend.py`: `supports_breakable_cuda_graph_prefix_off_cuda` (default False).
- `aiter_backend.py`: opt in.
- `hybrid_linear_attn_backend.py`: forward the flag to the full-attention backend.
- `prefill_cuda_graph_runner.py`: consult the flag in the admission check.

## Results
8x MI350X Kimi-K3: prefill graphs capture and replay for 256/261 mixed and radix-hit batches; single-request 256-token prefill -39% latency; throughput on the fixed 1k/8k cells unchanged within noise (opt-in only changes admission).

## Checklist
- [x] Format your code according to the Code Formatting with Pre-Commit.
- [x] Add unit tests as outlined in the Running Unit Tests. (admission flag; covered by the existing breakable-graph prefill tests on CUDA, which keep the default)
- [x] Update documentation / docstrings / example tutorials as needed.
- [x] Provide throughput / latency benchmark results and accuracy evaluation results as needed.
