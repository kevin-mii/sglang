import os, sys, time, asyncio, aiohttp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_prefix_bench import build_token_pool, build_requests
from transformers import AutoTokenizer
url, n = sys.argv[1], int(sys.argv[2])
tok = AutoTokenizer.from_pretrained(os.environ.get("M3_WORK", "/scratch") + "/models/MiniMax-M3-MXFP4", trust_remote_code=True)
texts = build_token_pool(tok, [os.environ.get("M3_WORK", "/scratch") + "/data/gsm8k/main/train-00000-of-00001.parquet", os.environ.get("M3_WORK", "/scratch") + "/data/gsm8k/main/test-00000-of-00001.parquet"])
async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        for trial in range(3):
            prefix, reqs = build_requests(tok, texts, 74176, 0.9, n, 900 + trial)
            async with s.post(url + "/generate", json={"input_ids": prefix + [11], "sampling_params": {"max_new_tokens": 1}}) as r:
                await r.read()
            t0 = time.perf_counter(); done = []
            async def one(ids):
                async with s.post(url + "/generate", json={"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}}) as r:
                    await r.read(); done.append(time.perf_counter() - t0)
            await asyncio.gather(*(one(x) for x in reqs))
            done.sort()
            print(f"burst {n}: last {done[-1]*1000:.0f} ms, median {done[len(done)//2]*1000:.0f} ms, per-req {done[-1]*1000/n:.0f} ms")
asyncio.run(main())
