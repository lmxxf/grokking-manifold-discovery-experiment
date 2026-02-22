"""
多 stride 邻接扫描 - 检测非标准环结构

对指定 results 目录的所有 checkpoint，扫描外层 stride=1,2 和内层 stride=1,4 的邻接得分。
用于发现 gcd(12,8)=4 扭转造成的非标准拓扑。

用法：
    python scan_strides.py --wd 1.5
    python scan_strides.py --tag wd2.0_5M
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
from scipy.spatial.distance import cdist

P = 97
G = 5


def compute_discrete_logs():
    dlog = {}
    val = 1
    for k in range(P - 1):
        dlog[val] = k
        val = (val * G) % P
    return dlog


def analyze_step(hidden, labels, dlog):
    # 外层陪集中心
    coset_vecs = {c: [] for c in range(12)}
    for i, lab in enumerate(labels):
        lab = int(lab)
        if lab == 0 or lab not in dlog:
            continue
        coset_vecs[dlog[lab] % 12].append(hidden[i])
    centers = {}
    for c in range(12):
        if coset_vecs[c]:
            centers[c] = np.mean(coset_vecs[c], axis=0)
    ids = sorted(centers.keys())
    mat = np.array([centers[i] for i in ids])
    dist = cdist(mat, mat, metric="euclidean")

    outer_scores = {}
    for stride in [1, 2]:
        correct = total = 0
        for idx_c, c in enumerate(ids):
            d = dist[idx_c].copy()
            d[idx_c] = np.inf
            nn = [ids[j] for j in np.argsort(d)[:2]]
            exp = {(c - stride) % 12, (c + stride) % 12}
            for n in nn:
                total += 1
                if n in exp:
                    correct += 1
        outer_scores[stride] = correct / total if total else 0

    # 内层代表点
    buckets = {}
    for i, lab in enumerate(labels):
        lab = int(lab)
        if lab == 0 or lab not in dlog:
            continue
        k = dlog[lab]
        key = (k % 12, k // 12)
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(hidden[i])
    reps = {k: np.mean(v, axis=0) for k, v in buckets.items()}

    inner_scores = {}
    for stride in [1, 4]:
        cs = []
        for c in range(12):
            iids = sorted([idx for (co, idx) in reps if co == c])
            if len(iids) < 3:
                continue
            pts = np.array([reps[(c, idx)] for idx in iids])
            dd = cdist(pts, pts, metric="euclidean")
            cor = tot = 0
            for ip, iid in enumerate(iids):
                dv = dd[ip].copy()
                dv[ip] = np.inf
                nnp = np.argmin(dv)
                nnid = iids[nnp]
                if nnid in {(iid - stride) % 8, (iid + stride) % 8}:
                    cor += 1
                tot += 1
            cs.append(cor / tot if tot else 0)
        inner_scores[stride] = np.mean(cs) if cs else 0

    return outer_scores, inner_scores


def main():
    parser = argparse.ArgumentParser(description="Multi-stride adjacency scan")
    parser.add_argument("--wd", type=str, default="2.0")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dir_name = args.tag if args.tag else f"wd_{args.wd}"
    act_dir = os.path.join(base_dir, "results", dir_name, "activations")

    if not os.path.isdir(act_dir):
        print(f"[ERROR] Not found: {act_dir}")
        sys.exit(1)

    dlog = compute_discrete_logs()

    files = sorted(glob.glob(os.path.join(act_dir, "step_*.npz")))
    print(f"Found {len(files)} checkpoints in {dir_name}")
    print()
    print("step       outer_s1  outer_s2  inner_s1  inner_s4")
    print("-" * 55)

    for f in files:
        m = re.search(r"step_(\d+)", os.path.basename(f))
        step = int(m.group(1))
        data = np.load(f)
        hidden, labels = data["hidden"], data["labels"]
        outer, inner = analyze_step(hidden, labels, dlog)
        print("%7d    %.3f     %.3f     %.3f     %.3f" % (
            step, outer[1], outer[2], inner[1], inner[4]))


if __name__ == "__main__":
    main()
