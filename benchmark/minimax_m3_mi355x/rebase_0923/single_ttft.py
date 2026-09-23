import os, sys, time, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_prefix_bench import build_token_pool, build_requests
from transformers import AutoTokenizer
url = sys.argv[1]
tok = AutoTokenizer.from_pretrained(os.environ.get("M3_WORK", "/scratch") + "/models/MiniMax-M3-MXFP4", trust_remote_code=True)
texts = build_token_pool(tok, [os.environ.get("M3_WORK", "/scratch") + "/data/gsm8k/main/train-00000-of-00001.parquet", os.environ.get("M3_WORK", "/scratch") + "/data/gsm8k/main/test-00000-of-00001.parquet"])
prefix, reqs = build_requests(tok, texts, 74176, 0.9, 6, 777)
requests.post(url + "/generate", json={"input_ids": prefix + [11], "sampling_params": {"max_new_tokens": 1}})
for ids in reqs:
    t = time.perf_counter()
    r = requests.post(url + "/generate", json={"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}}).json()
    print(f"prefill 7.4K new on cached 66.8K: {1000*(time.perf_counter()-t):.0f} ms, cached={r['meta_info'].get('cached_tokens')}")
