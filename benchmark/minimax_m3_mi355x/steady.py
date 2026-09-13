"""Closed-loop steady-state decode throughput from server counters (natural EOS, no ignore_eos).
N worker threads each keep one request in flight (long-form prompts over cached ~70K docs, max_new 1200).
After WARM seconds, measure generation_tokens_total delta over SECONDS. Usage: python steady.py URL N SECONDS [WARM=15]"""
import requests, glob, time, sys, re, threading, concurrent.futures as cf, random, os
url=sys.argv[1]; N=int(sys.argv[2]); SEC=float(sys.argv[3]); WARM=float(sys.argv[4]) if len(sys.argv)>4 else 15.0
files=sorted(glob.glob("/sgl-workspace/sglang/python/sglang/srt/**/*.py", recursive=True))
corpus="\n".join(open(f).read() for f in files)
CTX=int(os.environ.get("CTX_CHARS","300000"))  # ~4.2 chars/token: 300000 -> ~70K tokens, 850000 -> ~200K
ASKS=["Write detailed documentation for every class and function in the dump above: one paragraph each, in order.",
      "Explain, function by function, what the code above does and list potential bugs with reasoning.",
      "Produce an exhaustive code review of the dump above, file by file, with concrete suggestions.",
      "Describe the control flow of the code above in depth, then write unit test plans for each module."]
def doc(i): off=i*(CTX+10000); return f"Code dump #{i}.\n\n"+corpus[off:off+CTX]+"\n\n"
def metric(name):
    m=requests.get(url+"/metrics", timeout=10).text
    return [float(x) for x in re.findall(r'sglang:%s\{[^}]*\} ([0-9.e+]+)'%name, m)]
def gen(i, ask, n):
    r=requests.post(url+"/generate", json=dict(text=doc(i)+ask, sampling_params=dict(max_new_tokens=n, temperature=0.7, top_p=0.95)), timeout=3600)
    return r.json()["meta_info"]["completion_tokens"]
if not os.environ.get("NOFLUSH"): requests.post(url+"/flush_cache")
with cf.ThreadPoolExecutor(N) as ex: list(ex.map(lambda i: gen(i, ASKS[0], 4), range(N)))
stop=threading.Event(); done=[0]*N; lens=[]
def worker(i):
    k=0
    while not stop.is_set():
        lens.append(gen(i, ASKS[(i+k)%len(ASKS)], 1200)); k+=1; done[i]+=1
ths=[threading.Thread(target=worker,args=(i,),daemon=True) for i in range(N)]
for t in ths: t.start()
time.sleep(WARM)
g0=sum(metric("generation_tokens_total")); s0=time.time(); runs=[]
while time.time()-s0<SEC:
    time.sleep(2.0); runs.append(max(metric("num_running_reqs")+[0]))
g1=sum(metric("generation_tokens_total")); s1=time.time()
stop.set()
tps=(g1-g0)/(s1-s0); a=metric("spec_accept_length")
print(f"STEADY N={N} gen_tok/s={tps:.0f} per_stream={tps/N:.1f} running(mean/min)={sum(runs)/len(runs):.1f}/{min(runs):.0f} accept_len(cum)={a[0] if a else 'na':.3} completions={sum(done)} mean_len={sum(lens)/max(1,len(lens)):.0f} window={s1-s0:.0f}s", flush=True)
