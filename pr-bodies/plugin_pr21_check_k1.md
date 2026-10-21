## Motivation

sgl-project/sglang#45112 regrouped SGLang's kernels. #17 moved the DeepSeek-V4.1 spec's and draft's reads (`unified_kv_gate`, `mhc_combine`); `check_k1` still imports the old modules, so on SGLang main `test/kernels/test_deepseek_v41_kernels.py` fails at TP2 and TP4 with `ImportError`.

## Modifications

`test/kernels/deepseek_v41/check_k1.py`: `fused_k_norm_rope_flashmla` from `attention/dsv4/norm_rope.py`, and `mxfp8_native_amd_gfx95` from `gemm/`.

## Accuracy Tests

MI350X, TP4, SGLang main f8fd41c36e, this tree (identical to #17 plus this commit):

- `test/unit/`: 120 passed.
- `test/kernels/test_deepseek_v41_kernels.py`: 4 passed.
- `test/kernels/test_deepseek_v41_k2i.py`: 14 passed.
- `test/e2e/test_deepseek_v41_server.py`: 9 passed (GSM8K 0.910 / 0.910).

## Checklist

- [x] `pre-commit run --all-files` passes.
- [x] Tests under `test/` cover the change.
- [x] The SGLang commit or PR this was tested against is named above.
- [ ] Accuracy and speed results are attached when kernel or forward code changes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
