"""ROCm raises the Triton decode split default, but an explicit value must survive.

``--triton-attention-num-kv-splits`` defaults to 8; on HIP the default becomes 16.
Any other value the operator passed (e.g. 64 for long-context decode on a single
KV head) has to reach the Triton backend unchanged.
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.arg_groups.platform_hook import handle_amd_specifics
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_STAGING_VAR = "GPU_PINNED_MIN_XFER_SIZE"


def _resolve(num_kv_splits: int, is_hip: bool) -> int:
    server_args = SimpleNamespace(triton_attention_num_kv_splits=num_kv_splits)
    with override_platform(is_hip=is_hip):
        handle_amd_specifics(server_args)
    return resolved_view(server_args).triton_attention_num_kv_splits


class TestRocmTritonNumKvSplits(CustomTestCase):
    def setUp(self):
        # handle_amd_specifics also seeds this env default on HIP.
        self._saved = os.environ.get(_STAGING_VAR)

    def tearDown(self):
        os.environ.pop(_STAGING_VAR, None)
        if self._saved is not None:
            os.environ[_STAGING_VAR] = self._saved

    def test_default_raised_on_hip(self):
        self.assertEqual(_resolve(8, is_hip=True), 16)

    def test_explicit_value_kept_on_hip(self):
        for value in (4, 16, 32, 64):
            with self.subTest(value=value):
                self.assertEqual(_resolve(value, is_hip=True), value)

    def test_untouched_off_hip(self):
        for value in (8, 64):
            with self.subTest(value=value):
                self.assertEqual(_resolve(value, is_hip=False), value)


if __name__ == "__main__":
    unittest.main()
