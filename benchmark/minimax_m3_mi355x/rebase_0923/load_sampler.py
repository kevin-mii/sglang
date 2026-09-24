"""Sample running/queued requests on each backend every 2 s: python load_sampler.py OUT.csv URL1 URL2 ..."""
import sys, time, re, urllib.request
out, urls = sys.argv[1], sys.argv[2:]
pat = {k: re.compile(rf"^sglang:{k}\{{[^}}]*\}} ([0-9.e+]+)", re.M) for k in ("num_running_reqs", "num_queue_reqs")}
with open(out, "w") as f:
    f.write("t," + ",".join(f"{u.rsplit(':',1)[1]}_{k}" for u in urls for k in pat) + "\n")
    while True:
        row = [f"{time.time():.1f}"]
        for u in urls:
            try:
                m = urllib.request.urlopen(u + "/metrics", timeout=2).read().decode()
                row += [str(sum(float(x) for x in p.findall(m))) for p in pat.values()]
            except Exception:
                row += ["", ""]
        f.write(",".join(row) + "\n"); f.flush(); time.sleep(2)
