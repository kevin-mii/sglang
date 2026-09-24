"""Write aiter's bf16 tuned-GEMM table without FlyDSL rows (FlyDSL hgemm kernels carry a hostcall buffer).

usage: python aiter_bf16_nohostcall_csv.py AITER_ROOT OUT.csv, then serve with AITER_CONFIG_GEMM_BF16=OUT.csv.
Merges configs/bf16_tuned_gemm.csv with model_configs/*bf16_tuned_gemm*.csv like aiter does; shapes whose only
tuned row was FlyDSL fall back to aiter's default (torch) solution.
"""

import glob
import sys

import pandas as pd

root, out = sys.argv[1], sys.argv[2]
files = [f"{root}/aiter/configs/bf16_tuned_gemm.csv"] + [
    f for f in sorted(glob.glob(f"{root}/aiter/configs/model_configs/*bf16_tuned_gemm*.csv")) if "untuned" not in f
]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
keys = ["gfx", "cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"]
kept = df[df["libtype"] != "flydsl"].sort_values("us").drop_duplicates(keys, keep="first")
kept.to_csv(out, index=False)
print(f"{len(df)} rows from {len(files)} files -> {len(kept)} (dropped {int((df['libtype'] == 'flydsl').sum())} flydsl)")
