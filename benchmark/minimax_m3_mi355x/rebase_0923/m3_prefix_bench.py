"""Closed-loop long-context benchmark: shared cached prefix + unique GSM8K suffix, fixed OSL.

Each request = one shared prefix (cache_ratio * ISL tokens, identical across requests, warmed before timing)
+ a unique suffix of GSM8K questions ((1 - cache_ratio) * ISL tokens). Output is forced to OSL tokens (ignore_eos).
For each concurrency c, 5*c requests run with at most c in flight. Uses sglang's native /generate with input_ids so the
ISL is exact; works against a single server or the sglang router.
"""

import argparse
import os
import asyncio
import json
import random
import time

import aiohttp
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def build_token_pool(tok, paths):
    rows = pd.concat([pd.read_parquet(p) for p in paths])
    texts = [f"Question: {q}\nAnswer: {a}\n\n" for q, a in zip(rows["question"], rows["answer"])]
    return texts


def tokens_from(tok, texts, start, n):
    out, i = [], start
    while len(out) < n:
        out.extend(tok.encode(texts[i % len(texts)], add_special_tokens=False))
        i += 1
    return out[:n], i


def build_requests(tok, texts, isl, cache_ratio, n_req, seed):
    prefix_len = int(round(isl * cache_ratio))
    suffix_len = isl - prefix_len
    header = tok.encode("You are a careful math tutor. Study the worked examples below, then solve the final problems.\n\n",
                        add_special_tokens=False)
    prefix_body, cursor = tokens_from(tok, texts, 0, prefix_len - len(header))
    prefix = header + prefix_body
    rng = random.Random(seed)
    reqs = []
    for r in range(n_req):
        # unique leading marker so no two suffixes share a first page, then GSM8K questions from a random offset
        marker = tok.encode(f"\n\n### Problem set {r} (seed {seed})\n", add_special_tokens=False)
        body, _ = tokens_from(tok, texts, rng.randrange(len(texts)), suffix_len - len(marker))
        reqs.append(prefix + marker + body)
    return prefix, reqs


async def one_request(session, url, ids, osl, results, t_bench0):
    payload = {"input_ids": ids, "stream": True,
               "sampling_params": {"max_new_tokens": osl, "ignore_eos": True, "temperature": 0.0}}
    t0 = time.perf_counter()
    ttft, last, itls, meta, n_chunks_tok = None, None, [], {}, 0
    prev_tokens = 0
    async with session.post(url + "/generate", json=payload) as resp:
        if resp.status != 200:
            results.append({"error": f"{resp.status} {await resp.text()}"})
            return
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            meta = obj.get("meta_info", meta)
            ntok = meta.get("completion_tokens", 0)
            now = time.perf_counter()
            if ntok > prev_tokens:
                if ttft is None:
                    ttft = now - t0
                else:
                    # a spec-decode chunk emits several tokens at once; spread the gap over them (vLLM/sglang convention)
                    gap = (now - last) / (ntok - prev_tokens)
                    itls.extend([gap] * (ntok - prev_tokens))
                last = now
                prev_tokens = ntok
    results.append({"ttft": ttft, "itls": itls, "e2e": time.perf_counter() - t0, "start": t0 - t_bench0,
                    "end": time.perf_counter() - t_bench0, "in": len(ids), "out": prev_tokens,
                    "cached": meta.get("cached_tokens"), "verify_ct": meta.get("spec_verify_ct"),
                    "accept_hist": meta.get("spec_accept_histogram")})


async def run_level(url, reqs, conc, osl):
    results = []
    sem = asyncio.Semaphore(conc)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=3600)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        t0 = time.perf_counter()

        async def guarded(ids):
            async with sem:
                await one_request(session, url, ids, osl, results, t0)

        await asyncio.gather(*(guarded(r) for r in reqs))
        wall = time.perf_counter() - t0
    return results, wall


async def warm(urls, prefix):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        for u in urls:
            for _ in range(2):
                async with s.post(u + "/generate", json={"input_ids": prefix + [11], "sampling_params":
                                  {"max_new_tokens": 8, "ignore_eos": True, "temperature": 0.0}}) as r:
                    await r.read()


def summarize(model, args, conc, res, wall, ngpu):
    ok = [r for r in res if "error" not in r and r["ttft"] is not None]
    errs = len(res) - len(ok)
    ttft = np.array([r["ttft"] for r in ok]) * 1000
    itl = np.concatenate([r["itls"] for r in ok]) * 1000 if ok else np.array([0.0])
    # throughput over the steady window: from first request end... keep it simple and honest: total tokens / wall time
    tin = sum(r["in"] for r in ok)
    tout = sum(r["out"] for r in ok)
    cached = [r["cached"] for r in ok if r["cached"] is not None]
    cache_pct = f"{100 * sum(cached) / tin:.1f}" if len(cached) == len(ok) and tin else "n/a"
    vct = [r["verify_ct"] for r in ok if r["verify_ct"]]
    if len(vct) == len(ok) and vct:
        acc_len = sum(r["out"] for r in ok) / sum(vct)
        acc_len_s, accept_s = f"{acc_len:.2f}", f"{100 * (acc_len - 1) / (args.draft_tokens - 1):.1f}"
    else:
        acc_len_s, accept_s = "n/a", "n/a"
    row = {"conc": conc, "reqs": len(ok), "errors": errs, "isl": int(np.mean([r["in"] for r in ok])), "cache": cache_pct,
           "ttft_p50": float(np.percentile(ttft, 50)), "ttft_p90": float(np.percentile(ttft, 90)),
           "itl_p50": float(np.percentile(itl, 50)), "itl_p90": float(np.percentile(itl, 90)),
           "in_s_gpu": tin / wall / ngpu, "out_s_gpu": tout / wall / ngpu,
           "tps_user_p50": float(np.percentile([r["out"] / (r["e2e"] - r["ttft"]) for r in ok], 50)),
           "accept": accept_s, "acc_len": acc_len_s, "wall_s": wall}
    return row


def print_report(model, args, rows, ngpu):
    print("\n=== report ===")
    print(f"model {model} | ISL {args.isl} OSL {args.osl} | cached prefix {int(args.cache_ratio * 100)}% | GPUs {ngpu} | prompts: GSM8K")
    print()
    hdr = (f"{'conc':>5} {'reqs':>5} {'ISL':>7}  {'cache%':<6} {'TTFT p50':>8}  {'TTFT p90':>8}  {'ITL p50':>7}  "
           f"{'in/s/gpu':>8}  {'out/s/gpu':>9}  {'TPS/usr':>7}  {'accept%':>7}  {'acc.len':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['conc']:>5} {r['reqs']:>5} {r['isl']:>7}  {r['cache']:<6} {r['ttft_p50']:>8.0f}  {r['ttft_p90']:>8.0f}  "
              f"{r['itl_p50']:>7.2f}  {r['in_s_gpu']:>8.1f}  {r['out_s_gpu']:>9.2f}  {r['tps_user_p50']:>7.1f}  "
              f"{r['accept']:>7}  {r['acc_len']:>7}" + (f"  ({r['errors']} errors)" if r["errors"] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--warm-urls", default=None, help="comma list of backend URLs to warm the prefix on (default: --url)")
    ap.add_argument("--model", default=os.environ.get("M3_WORK", "/scratch") + "/models/MiniMax-M3-MXFP4")
    ap.add_argument("--model-name", default="amd/MiniMax-M3-MXFP4")
    ap.add_argument("--isl", type=int, default=74176)
    ap.add_argument("--osl", type=int, default=650)
    ap.add_argument("--cache-ratio", type=float, default=0.9)
    ap.add_argument("--conc", default="64,80,128")
    ap.add_argument("--req-mult", type=int, default=5)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--draft-tokens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    texts = build_token_pool(tok, [os.environ.get("M3_WORK", "/scratch") + "/data/gsm8k/main/train-00000-of-00001.parquet",
                                   os.environ.get("M3_WORK", "/scratch") + "/data/gsm8k/main/test-00000-of-00001.parquet"])
    concs = [int(c) for c in args.conc.split(",")]
    rows = []
    for i, c in enumerate(concs):
        prefix, reqs = build_requests(tok, texts, args.isl, args.cache_ratio, c * args.req_mult, args.seed * 1000 + i)
        asyncio.run(warm((args.warm_urls or args.url).split(","), prefix))
        res, wall = asyncio.run(run_level(args.url, reqs, c, args.osl))
        row = summarize(args.model_name, args, c, res, wall, args.gpus)
        rows.append(row)
        print(json.dumps(row), flush=True)
        errs = [r["error"] for r in res if "error" in r][:3]
        if errs:
            print("sample errors:", errs, flush=True)
    print_report(args.model_name, args, rows, args.gpus)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
