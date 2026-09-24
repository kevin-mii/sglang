import json, os, sys, requests, subprocess, concurrent.futures as cf
repo = "lmsysorg/sglang-rocm"; dig = "sha256:02108de8f9418a12425fb56415794fe82d180cb985488725d485e96d97a9a26b"
out = os.environ.get("M3_IMAGE", "/scratch/m3/image")
def tok():
    return requests.get(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull").json()["token"]
H = lambda: {"Authorization": f"Bearer {tok()}", "Accept": ",".join([
    "application/vnd.oci.image.index.v1+json","application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json","application/vnd.docker.distribution.manifest.list.v2+json"])}
m = requests.get(f"https://registry-1.docker.io/v2/{repo}/manifests/{dig}", headers=H()).json()
if "manifests" in m:
    d = [x for x in m["manifests"] if x.get("platform",{}).get("architecture")=="amd64"][0]["digest"]
    m = requests.get(f"https://registry-1.docker.io/v2/{repo}/manifests/{d}", headers=H()).json()
json.dump(m, open(f"{out}/manifest.json","w"), indent=1)
layers = m["layers"]; print(len(layers), "layers", sum(l["size"] for l in layers)/1e9, "GB", flush=True)
def get(l):
    fn = f"{out}/blobs/{l['digest'].split(':')[1]}"
    if os.path.exists(fn) and os.path.getsize(fn)==l["size"]: return fn
    subprocess.check_call(["curl","-sSL","--retry","5","-o",fn,"-H",f"Authorization: Bearer {tok()}",
        f"https://registry-1.docker.io/v2/{repo}/blobs/{l['digest']}"])
    assert os.path.getsize(fn)==l["size"], fn
    print("got", fn, l["size"]/1e9, flush=True); return fn
os.makedirs(f"{out}/blobs", exist_ok=True)
with cf.ThreadPoolExecutor(8) as ex: list(ex.map(get, layers))
cfg = requests.get(f"https://registry-1.docker.io/v2/{repo}/blobs/{m['config']['digest']}", headers=H()).json()
json.dump(cfg, open(f"{out}/config.json","w"), indent=1)
print("PULLED", flush=True)
