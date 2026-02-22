"""
嵌套 Grokking 分析 - 检测 Z_12 陪集内部的 Z_8 环结构涌现

核心思路：
- 乘法群 Z*_97 同构于 Z_96 = Z_12 x Z_8
- 外层：k mod 12 给出 12 个陪集 → Z_12 循环群
- 内层：同一陪集内的 8 个元素，按 k // 12 排序 → Z_8 循环群
- 如果内层 Z_8 结构也被学到，表现为陪集内部元素在 hidden 空间的邻接关系
  也符合 0→1→2→...→7→0 的环

分析流程：
1. 遍历所有 checkpoint 的 .npz 文件
2. 对每个 checkpoint 做两层邻接分析（外层 Z_12 + 内层 Z_8）
3. 做 PCA 维度分析（全局 + 陪集内部）
4. 输出 JSON 时间线 + 可视化 PNG

用法：
    python analyze_nested_structure.py --wd 2.0
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============ 常数 ============
P = 97
G = 5  # 原根
ORDER = P - 1  # 96
NUM_OUTER = 12  # Z_12
NUM_INNER = ORDER // NUM_OUTER  # 8, Z_8


def compute_discrete_logs(p=P, g=G):
    """计算所有非零元素的离散对数 g^k mod p -> k"""
    dlog = {}
    val = 1
    for k in range(p - 1):
        dlog[val] = k
        val = (val * g) % p
    return dlog


# ============ 数据加载 ============

def discover_checkpoints(activations_dir):
    """发现所有 step_XXXXXX.npz 文件并按 step 排序返回 [(step, path), ...]"""
    pattern = os.path.join(activations_dir, "step_*.npz")
    files = sorted(glob.glob(pattern))
    result = []
    for f in files:
        m = re.search(r"step_(\d+)\.npz", os.path.basename(f))
        if m:
            result.append((int(m.group(1)), f))
    result.sort(key=lambda x: x[0])
    return result


def load_train_log(results_dir):
    """加载 train_log.json，返回 {step: test_acc} 字典"""
    path = os.path.join(results_dir, "train_log.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        log = json.load(f)
    step_to_acc = {}
    for s, a in zip(log["steps"], log["test_acc"]):
        step_to_acc[int(s)] = float(a)
    return step_to_acc


# ============ 核心：外层 Z_12 邻接 ============

def compute_outer_adjacency(hidden, labels, dlog):
    """
    计算 Z_12 陪集中心的邻接得分（直接在原始 hidden 空间）。
    对每个陪集，聚合所有属于该陪集的样本（取均值）得到 12 个中心。
    检查每个中心的 2-最近邻是否是 mod 12 的 ±1 邻居。
    返回 score (0~1)。
    """
    # 聚合
    coset_vecs = {c: [] for c in range(NUM_OUTER)}
    for i, lab in enumerate(labels):
        lab = int(lab)
        if lab == 0 or lab not in dlog:
            continue
        k = dlog[lab]
        coset_vecs[k % NUM_OUTER].append(hidden[i])

    centers = {}
    for c in range(NUM_OUTER):
        if coset_vecs[c]:
            centers[c] = np.mean(coset_vecs[c], axis=0)

    if len(centers) < NUM_OUTER:
        # 某些陪集无数据
        pass

    if len(centers) < 3:
        return 0.0

    ids = sorted(centers.keys())
    mat = np.array([centers[i] for i in ids])
    dist = cdist(mat, mat, metric="euclidean")

    correct = 0
    total = 0
    for idx, c in enumerate(ids):
        d = dist[idx].copy()
        d[idx] = np.inf
        nn_idx = np.argsort(d)[:2]
        nn_cosets = [ids[j] for j in nn_idx]
        expected = {(c - 1) % NUM_OUTER, (c + 1) % NUM_OUTER}
        for n in nn_cosets:
            total += 1
            if n in expected:
                correct += 1

    return correct / total if total > 0 else 0.0


# ============ 核心：内层 Z_8 邻接 ============

def compute_inner_adjacency(hidden, labels, dlog):
    """
    对每个 Z_12 陪集，检查其内部 Z_8 环结构。

    步骤：
    1. 对每个陪集，按 k // 12 聚合同一 inner_index 的点（取均值）
       得到最多 8 个代表点。
    2. 对这 8 个代表点，检查每个点的最近邻是否是 Z_8 环上的 ±1 邻居。
    3. 返回 12 个陪集各自的得分 + 平均得分。

    Z_8 环：index 0→1→2→...→7→0
    """
    # 先按 (coset, inner_index) 聚合
    # coset = k % 12, inner_index = k // 12
    buckets = {}  # (coset, inner_idx) -> [vectors]
    for i, lab in enumerate(labels):
        lab = int(lab)
        if lab == 0 or lab not in dlog:
            continue
        k = dlog[lab]
        coset = k % NUM_OUTER
        inner_idx = k // NUM_OUTER
        key = (coset, inner_idx)
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(hidden[i])

    # 聚合为代表点
    representatives = {}  # (coset, inner_idx) -> mean_vector
    for key, vecs in buckets.items():
        representatives[key] = np.mean(vecs, axis=0)

    # 逐陪集分析
    per_coset_scores = {}
    for c in range(NUM_OUTER):
        # 取出该陪集的代表点
        inner_ids = sorted([idx for (co, idx) in representatives if co == c])
        if len(inner_ids) < 3:
            per_coset_scores[c] = 0.0
            continue

        pts = np.array([representatives[(c, idx)] for idx in inner_ids])
        dist = cdist(pts, pts, metric="euclidean")

        correct = 0
        total = 0
        for i_pos, inner_id in enumerate(inner_ids):
            d = dist[i_pos].copy()
            d[i_pos] = np.inf
            # 只看 1-最近邻（因为只有 8 个点，看 1-NN 更有意义）
            nn_pos = np.argmin(d)
            nn_inner_id = inner_ids[nn_pos]
            expected = {(inner_id - 1) % NUM_INNER, (inner_id + 1) % NUM_INNER}
            total += 1
            if nn_inner_id in expected:
                correct += 1

        per_coset_scores[c] = correct / total if total > 0 else 0.0

    valid_scores = [s for s in per_coset_scores.values() if s > 0 or True]
    mean_score = np.mean(valid_scores) if valid_scores else 0.0

    return float(mean_score), {int(k): float(v) for k, v in per_coset_scores.items()}


# ============ PCA 维度分析 ============

def pca_effective_dim(X, threshold=0.95):
    """
    计算 PCA 有效维度：需要多少主成分解释 threshold 比例的方差。
    X: (n_samples, n_features)
    """
    if X.shape[0] < 2:
        return 0
    n_components = min(X.shape[0], X.shape[1])
    if n_components < 1:
        return 0
    pca = PCA(n_components=n_components)
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dim = int(np.searchsorted(cumvar, threshold) + 1)
    return min(dim, n_components)


def compute_pca_dimensions(hidden, labels, dlog):
    """
    计算：
    1. 全局 PCA 维度（95% 方差）
    2. 每个陪集内部的 PCA 维度（对代表点做 PCA）
    """
    # 全局：只取有效样本
    valid_mask = []
    for lab in labels:
        lab = int(lab)
        valid_mask.append(lab != 0 and lab in dlog)
    valid_mask = np.array(valid_mask)
    valid_hidden = hidden[valid_mask]

    global_dim = pca_effective_dim(valid_hidden) if valid_hidden.shape[0] > 1 else 0

    # 陪集内部：用原始样本（不聚合），因为聚合后只有 8 个点
    # 但也可以用原始样本来算该陪集的内在维度
    coset_samples = {c: [] for c in range(NUM_OUTER)}
    for i, lab in enumerate(labels):
        lab = int(lab)
        if lab == 0 or lab not in dlog:
            continue
        k = dlog[lab]
        coset_samples[k % NUM_OUTER].append(hidden[i])

    inner_dims = {}
    for c in range(NUM_OUTER):
        if len(coset_samples[c]) < 2:
            inner_dims[c] = 0
        else:
            X = np.array(coset_samples[c])
            inner_dims[c] = pca_effective_dim(X)

    inner_dim_mean = float(np.mean(list(inner_dims.values())))
    return global_dim, inner_dim_mean, {int(k): int(v) for k, v in inner_dims.items()}


# ============ 单个 checkpoint 的完整分析 ============

def analyze_checkpoint(npz_path, dlog):
    """对一个 checkpoint 做完整的嵌套结构分析"""
    data = np.load(npz_path)
    hidden = data["hidden"]  # (N, D)
    labels = data["labels"]  # (N,)

    outer_score = compute_outer_adjacency(hidden, labels, dlog)
    inner_mean, inner_per_coset = compute_inner_adjacency(hidden, labels, dlog)
    global_dim, inner_dim_mean, inner_dims = compute_pca_dimensions(hidden, labels, dlog)

    return {
        "outer_adjacency_score": outer_score,
        "inner_adjacency_score_mean": inner_mean,
        "inner_adjacency_score_per_coset": inner_per_coset,
        "global_pca_dim": global_dim,
        "inner_pca_dim_mean": inner_dim_mean,
        "inner_pca_dim_per_coset": inner_dims,
    }


# ============ 可视化 ============

def plot_timeline(timeline, output_path):
    """
    上面板：test_acc + outer/inner adjacency score 随 step 变化
    下面板：global PCA dim + inner PCA dim mean 随 step 变化
    """
    steps = [e["step"] for e in timeline]
    test_accs = [e.get("test_acc", None) for e in timeline]
    outer_scores = [e["outer_adjacency_score"] for e in timeline]
    inner_scores = [e["inner_adjacency_score_mean"] for e in timeline]
    global_dims = [e["global_pca_dim"] for e in timeline]
    inner_dims = [e["inner_pca_dim_mean"] for e in timeline]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # --- 上面板 ---
    color_acc = "#2196F3"
    color_outer = "#E91E63"
    color_inner = "#4CAF50"

    ax1_acc = ax1
    ax1_adj = ax1.twinx()

    # test_acc
    valid_acc = [(s, a) for s, a in zip(steps, test_accs) if a is not None]
    if valid_acc:
        ss, aa = zip(*valid_acc)
        ax1_acc.plot(ss, aa, color=color_acc, linewidth=2, label="test_acc", alpha=0.7)
    ax1_acc.set_ylabel("Test Accuracy", color=color_acc)
    ax1_acc.tick_params(axis="y", labelcolor=color_acc)
    ax1_acc.set_ylim(-0.05, 1.05)

    # adjacency scores
    ax1_adj.plot(steps, outer_scores, color=color_outer, linewidth=2,
                 label=f"outer Z$_{{12}}$ adj (baseline={2/NUM_OUTER:.0%})", marker=".", markersize=3)
    ax1_adj.plot(steps, inner_scores, color=color_inner, linewidth=2,
                 label=f"inner Z$_{{8}}$ adj (baseline={2/NUM_INNER:.0%})", marker=".", markersize=3)
    # 随机基线
    ax1_adj.axhline(2 / NUM_OUTER, color=color_outer, linestyle="--", alpha=0.3,
                     label=f"Z$_{{12}}$ random = {2/NUM_OUTER:.1%}")
    ax1_adj.axhline(2 / NUM_INNER, color=color_inner, linestyle="--", alpha=0.3,
                     label=f"Z$_{{8}}$ random = {2/NUM_INNER:.1%}")
    ax1_adj.set_ylabel("Adjacency Score")
    ax1_adj.set_ylim(-0.05, 1.05)

    # 合并图例
    lines1, labels1 = ax1_acc.get_legend_handles_labels()
    lines2, labels2 = ax1_adj.get_legend_handles_labels()
    ax1_adj.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)

    ax1.set_title("Nested Grokking: Adjacency Score Timeline\n"
                   f"(P={P}, g={G}, Z*_{{97}} = Z_{{96}} = Z_{{12}} x Z_{{8}})")
    ax1.grid(True, alpha=0.3)

    # --- 下面板 ---
    color_global = "#FF9800"
    color_inner_dim = "#9C27B0"

    ax2.plot(steps, global_dims, color=color_global, linewidth=2,
             label="global PCA dim (95% var)", marker=".", markersize=3)
    ax2.plot(steps, inner_dims, color=color_inner_dim, linewidth=2,
             label="inner PCA dim mean (95% var)", marker=".", markersize=3)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("PCA Effective Dimension")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Dimensionality Evolution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {output_path}")


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="Nested Grokking Analysis: Z_12 x Z_8")
    parser.add_argument("--wd", type=str, default="2.0",
                        help="Weight decay value (determines results subdirectory, e.g. 2.0)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Custom results subdirectory name (overrides --wd)")
    args = parser.parse_args()

    # 路径构建
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dir_name = args.tag if args.tag else f"wd_{args.wd}"
    results_dir = os.path.join(base_dir, "results", dir_name)
    activations_dir = os.path.join(results_dir, "activations")
    output_dir = os.path.join(results_dir, "nested_analysis")

    # 如果 wd_XXX 目录不存在，尝试直接用 results/
    if not os.path.isdir(results_dir):
        alt = os.path.join(base_dir, "results")
        if os.path.isdir(alt):
            results_dir = alt
            activations_dir = os.path.join(results_dir, "activations")
            output_dir = os.path.join(results_dir, "nested_analysis")
            print(f"[INFO] wd dir not found, falling back to {results_dir}")
        else:
            print(f"[ERROR] Results directory not found: {results_dir}")
            sys.exit(1)

    if not os.path.isdir(activations_dir):
        print(f"[ERROR] Activations directory not found: {activations_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # 离散对数表
    dlog = compute_discrete_logs()
    print(f"Computed discrete logs for {len(dlog)} elements (P={P}, g={G})")
    print(f"Group structure: Z*_{P} ~ Z_{ORDER} = Z_{NUM_OUTER} x Z_{NUM_INNER}")

    # 加载 train_log
    step_to_acc = load_train_log(results_dir)
    print(f"Loaded train_log with {len(step_to_acc)} steps")

    # 发现 checkpoints
    checkpoints = discover_checkpoints(activations_dir)
    print(f"Found {len(checkpoints)} checkpoints in {activations_dir}")

    if not checkpoints:
        print("[ERROR] No checkpoint files found.")
        sys.exit(1)

    # 逐 checkpoint 分析
    timeline = []
    for idx, (step, path) in enumerate(checkpoints):
        print(f"  [{idx+1}/{len(checkpoints)}] step={step:>7d} ... ", end="", flush=True)
        try:
            result = analyze_checkpoint(path, dlog)
            result["step"] = step
            result["test_acc"] = step_to_acc.get(step, None)
            timeline.append(result)
            print(f"outer={result['outer_adjacency_score']:.3f}  "
                  f"inner={result['inner_adjacency_score_mean']:.3f}  "
                  f"pca_g={result['global_pca_dim']}  "
                  f"pca_i={result['inner_pca_dim_mean']:.1f}")
        except Exception as e:
            print(f"ERROR: {e}")

    # 输出 JSON
    json_path = os.path.join(output_dir, "nested_structure_timeline.json")
    with open(json_path, "w") as f:
        json.dump(timeline, f, indent=2)
    print(f"\nSaved timeline JSON to {json_path}")

    # 输出可视化
    png_path = os.path.join(output_dir, "nested_structure_timeline.png")
    plot_timeline(timeline, png_path)

    # 摘要
    print(f"\n{'='*60}")
    print("NESTED GROKKING ANALYSIS SUMMARY")
    print(f"{'='*60}")

    if timeline:
        last = timeline[-1]
        first = timeline[0]
        print(f"Steps analyzed: {first['step']} -> {last['step']} ({len(timeline)} checkpoints)")
        print(f"\nFinal checkpoint (step {last['step']}):")
        print(f"  test_acc              = {last.get('test_acc', 'N/A')}")
        print(f"  outer Z_12 adjacency  = {last['outer_adjacency_score']:.3f}  "
              f"(random baseline = {2/NUM_OUTER:.3f})")
        print(f"  inner Z_8  adjacency  = {last['inner_adjacency_score_mean']:.3f}  "
              f"(random baseline = {2/NUM_INNER:.3f})")
        print(f"  global PCA dim        = {last['global_pca_dim']}")
        print(f"  inner PCA dim mean    = {last['inner_pca_dim_mean']:.1f}")

        # 检测嵌套 grokking 信号
        # 寻找外层先涌现、内层后涌现的时间差
        outer_threshold = 0.5
        inner_threshold = 0.5
        outer_emerge_step = None
        inner_emerge_step = None
        for entry in timeline:
            if outer_emerge_step is None and entry["outer_adjacency_score"] >= outer_threshold:
                outer_emerge_step = entry["step"]
            if inner_emerge_step is None and entry["inner_adjacency_score_mean"] >= inner_threshold:
                inner_emerge_step = entry["step"]

        print(f"\nEmergence detection (threshold={outer_threshold}):")
        if outer_emerge_step:
            print(f"  outer Z_12 emerged at step {outer_emerge_step}")
        else:
            print(f"  outer Z_12 did NOT emerge (never exceeded {outer_threshold})")
        if inner_emerge_step:
            print(f"  inner Z_8  emerged at step {inner_emerge_step}")
        else:
            print(f"  inner Z_8  did NOT emerge (never exceeded {inner_threshold})")

        if outer_emerge_step and inner_emerge_step:
            if inner_emerge_step > outer_emerge_step:
                print(f"\n  >>> NESTED GROKKING DETECTED <<<")
                print(f"  Z_12 emerged first (step {outer_emerge_step}), "
                      f"then Z_8 (step {inner_emerge_step})")
                print(f"  Delay = {inner_emerge_step - outer_emerge_step} steps")
            elif inner_emerge_step == outer_emerge_step:
                print(f"\n  Both layers emerged simultaneously at step {outer_emerge_step}")
            else:
                print(f"\n  Inner Z_8 emerged BEFORE outer Z_12 (unexpected)")
        elif outer_emerge_step and not inner_emerge_step:
            print(f"\n  Only outer Z_12 structure emerged; inner Z_8 not detected")
            print(f"  (This could mean Z_8 structure has not yet grokked)")

    print(f"\n{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
