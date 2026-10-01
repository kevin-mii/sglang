"""Opt-in histogram of CUDA-graph replay shapes, for tuning the capture-size lists.

With ``SGLANG_GRAPH_SHAPE_STATS=<path prefix>``, TP rank 0 counts every graph replay by
(runner, forward mode, live rows, captured rows) and rewrites ``<prefix>.json`` every 30 s.
Rows are the tokens entering the graph, so padding = (captured - live) / captured.
"""

import json
import os
import time
from collections import Counter
from typing import Optional

from sglang.srt.environ import envs


class _GraphShapeStats:
    def __init__(self, path: str):
        self._path = path
        self._counts: Counter = Counter()
        self._last_flush = time.monotonic()

    def record(
        self, runner: str, mode: str, live_rows: int, captured_rows: int
    ) -> None:
        self._counts[(runner, mode, live_rows, captured_rows)] += 1
        now = time.monotonic()
        if now - self._last_flush >= 30.0:
            self._last_flush = now
            self._flush()

    def _flush(self) -> None:
        rows = [
            {
                "runner": r,
                "mode": m,
                "live_rows": live,
                "captured_rows": cap,
                "replays": n,
            }
            for (r, m, live, cap), n in sorted(self._counts.items())
        ]
        tmp = f"{self._path}.json.tmp"
        with open(tmp, "w") as f:
            json.dump(rows, f)
        os.replace(tmp, f"{self._path}.json")


_STATS: Optional[_GraphShapeStats] = None
_RESOLVED = False


def record_graph_replay(
    runner: str, mode: str, live_rows: int, captured_rows: int
) -> None:
    global _STATS, _RESOLVED
    if not _RESOLVED:
        _RESOLVED = True
        path = envs.SGLANG_GRAPH_SHAPE_STATS.get()
        if path:
            from sglang.srt.runtime_context import get_parallel

            if get_parallel().tp_rank == 0:
                _STATS = _GraphShapeStats(path)
    if _STATS is not None:
        _STATS.record(runner, mode, live_rows, captured_rows)
