"""HiCache robustness: cancel requests while their prefix is being restored from host, then check health.

1. Insert a long target prefix, flood the GPU pool so it is evicted to host.
2. Fire requests that reuse it and cancel each (client disconnect) after a short random delay, so aborts
   land before, during and after the host->GPU load.
3. Check /health, that a final request still hits the host tier, and the host pool gauges.
"""
import asyncio, json, random, sys, time, urllib.request
import aiohttp

URL = sys.argv[1]; V = 150000


def metric(name):
    txt = urllib.request.urlopen(URL + "/metrics", timeout=10).read().decode()
    return sum(float(l.split()[-1]) for l in txt.splitlines() if l.startswith(name + "{") and 'tp_rank="0"' in l)


async def gen(s, ids, n=16, timeout=None):
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": n, "temperature": 0}}
    async with s.post(URL + "/generate", json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
        return await r.json()


async def cancelled(s, ids, delay):
    try:
        await gen(s, ids, n=64, timeout=delay)
        return "completed"
    except asyncio.TimeoutError:
        return "cancelled"


async def main():
    rng = random.Random(3)
    target = [rng.randrange(1000, V) for _ in range(80000)]
    async with aiohttp.ClientSession() as s:
        await gen(s, target)
        floods = [[rng.randrange(1000, V) for _ in range(150000)] for _ in range(62)]
        sem = asyncio.Semaphore(8)

        async def one(ids):
            async with sem:
                await gen(s, ids, n=1)

        t = time.time()
        await asyncio.gather(*(one(f) for f in floods))
        print(f"flooded in {time.time() - t:.0f} s; host used {metric('sglang:hicache_host_used_tokens'):.0f}", flush=True)
        outcomes = await asyncio.gather(*(cancelled(s, target + [rng.randrange(1000, V)], rng.uniform(0.02, 1.5)) for _ in range(30)))
        print("aborts:", {o: outcomes.count(o) for o in set(outcomes)}, flush=True)
        await asyncio.sleep(5)
        ok = urllib.request.urlopen(URL + "/health", timeout=10).status
        final = await gen(s, target + [7])
        print("health", ok, "final cached", final["meta_info"].get("cached_tokens_details"), flush=True)
        print(f"host used {metric('sglang:hicache_host_used_tokens'):.0f} of {metric('sglang:hicache_host_total_tokens'):.0f}; "
              f"dropped {metric('sglang:hicache_dropped_tokens_total'):.0f}; running {metric('sglang:num_running_reqs'):.0f}", flush=True)


asyncio.run(main())
