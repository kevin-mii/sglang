"""Per-user decode tok/s at 1, 4, 8 and 16 concurrent requests on the e2e test's
DSpark server, simulated acceptance 3.51 (the median of 3 rounds, 1500 tokens a
request): one arm, the sgl-amd-mono on PYTHONPATH.
Usage: python decode_ab.py LABEL on|off [EXTRA_SERVER_ARGS...]"""

import concurrent.futures as cf
import json
import os
import statistics
import sys
import time

import requests

sys.path.insert(0, os.environ.get("OOT_E2E_DIR", ""))
try:
    from deepseek_v41_server import ARGS, ENV, MODEL
except ImportError:
    from sgl_amd_mono.test.deepseek_v41.server import ARGS, ENV, MODEL
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

PERF_ENV = {
    "SGLANG_SIMULATE_ACC_LEN": "3.51",
    "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
    "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
}
PROMPTS = (
    "Write a long detailed essay about the history of lighthouses and their keepers.",
    "Explain in great detail how TCP congestion control algorithms evolved.",
    "Describe the full lifecycle of a star from nebula to remnant, in depth.",
    "Give a thorough tutorial on writing a compiler front end in Python.",
)
CONCURRENCY = tuple(int(c) for c in os.environ.get("CONCURRENCY", "1,4,8,16").split(","))
TOKENS, ROUNDS = int(os.environ.get("AB_TOKENS", "1500")), int(os.environ.get("AB_ROUNDS", "3"))
# AB_CONTEXT=N: a shared ~N-token prefix ahead of each prompt (radix-cached after the
# first request); the rate is then decode only, from the first streamed token
CONTEXT = int(os.environ.get("AB_CONTEXT", "0"))
FILLER = (
    "Lighthouse keepers logged the weather, trimmed the wicks, polished the lenses and "
    "recorded every passing ship in ledgers that historians still read today. "
)
PREFIX = FILLER * (CONTEXT // 32) if CONTEXT else ""


def _decode_rate(body: dict) -> float:
    body = {**body, "stream": True}
    first = last = None
    tokens = 0
    with requests.post(
        DEFAULT_URL_FOR_TEST + "/generate", json=body, timeout=1800, stream=True
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                continue
            now = time.perf_counter()
            first = first or now
            last = now
            tokens = json.loads(line[5:])["meta_info"]["completion_tokens"]
    return (tokens - 1) / (last - first)


def _per_user_rate(i: int) -> float:
    body = {
        "text": PREFIX + PROMPTS[i % len(PROMPTS)],
        "sampling_params": {"max_new_tokens": TOKENS, "temperature": 0, "ignore_eos": True},
    }
    if CONTEXT:
        return _decode_rate(body)
    start = time.perf_counter()
    r = requests.post(DEFAULT_URL_FOR_TEST + "/generate", json=body, timeout=900)
    r.raise_for_status()
    return r.json()["meta_info"]["completion_tokens"] / (time.perf_counter() - start)


def main():
    label, mono, extra = sys.argv[1], sys.argv[2] == "on", sys.argv[3:]
    process = popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=3600,
        other_args=ARGS + (["--monokernel-backend", "amd"] if mono else []) + extra,
        # SIM_ACC=0: DSpark's real acceptance (the draft's numerics count too)
        env={**os.environ, **ENV, **(PERF_ENV if os.environ.get("SIM_ACC", "1") != "0" else {})},
    )
    try:
        rates = {}
        for n in CONCURRENCY:
            rounds = []
            for _ in range(ROUNDS):
                with cf.ThreadPoolExecutor(n) as pool:
                    rounds.append(statistics.mean(pool.map(_per_user_rate, range(n))))
            rates[n] = round(statistics.median(rounds), 1)
        info = requests.get(DEFAULT_URL_FOR_TEST + "/server_info", timeout=60).json()
        accept = (info.get("internal_states") or [{}])[0].get("avg_spec_accept_length")
        out = {"label": label, "mono": mono, "rates": rates, "accept_length": accept}
        if CONTEXT:
            out["context"] = CONTEXT
        print("DECODE_AB " + json.dumps(out), flush=True)
    finally:
        terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    main()
