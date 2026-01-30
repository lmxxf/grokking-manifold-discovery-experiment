"""
群结构邻接关系分析 - 模乘法
验证"发现了结构"而非"分类变紧"

核心问题：
- 模乘法学到了 12 个大簇（k mod 12 的陪集结构）
- 这 12 个簇在 UMAP 空间中的邻接关系是否符合 Z_12 的循环群结构？
- 即：陪集 k 的簇是否与陪集 k±1 的簇空间相邻？

方法：
1. 加载 Grokking 后的 UMAP embedding
2. 计算每个 label 的离散对数 k（使用原根 g=5）
3. 按 k mod 12 分组，计算 12 个陪集的中心
4. 检查陪集之间的邻接关系是否符合 Z_12 结构

用法（在 Docker 容器内）：
    python /workspace/ai-theorys-study/arxiv/wechat67/exp_group2_multiplication/code/analyze_adjacency.py
"""

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from scipy.spatial.distance import cdist
import os
import json

# ============ 配置 ============
RESULTS_DIR = "/workspace/ai-theorys-study/arxiv/wechat67/exp_group2_multiplication/results"
ACTIVATIONS_DIR = os.path.join(RESULTS_DIR, "activations")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "adjacency_analysis")

# Grokking 后的时间点
STEP = 100000

P = 97  # 模数
G = 5   # 97 的原根


def compute_discrete_logs(p, g):
    """计算所有非零元素的离散对数"""
    # g^k mod p = y  =>  discrete_log[y] = k
    discrete_log = {}
    val = 1
    for k in range(p - 1):
        discrete_log[val] = k
        val = (val * g) % p
    return discrete_log


def load_activations(step):
    """加载指定 step 的激活"""
    path = os.path.join(ACTIVATIONS_DIR, f"step_{step:06d}.npz")
    if not os.path.exists(path):
        import glob
        files = sorted(glob.glob(os.path.join(ACTIVATIONS_DIR, "step_*.npz")))
        if not files:
            raise FileNotFoundError(f"No activation files found in {ACTIVATIONS_DIR}")
        path = files[-1]
        print(f"Using {path} instead of step {step}")
    data = np.load(path)
    return data["hidden"], data["labels"]


def compute_coset_centers(embedding, labels, discrete_log, num_cosets=12):
    """计算每个陪集（k mod 12）的中心"""
    # 按 k mod 12 分组
    coset_points = {i: [] for i in range(num_cosets)}

    for i, label in enumerate(labels):
        if label == 0:
            continue  # 跳过 0（乘法群不包含 0）
        if label not in discrete_log:
            continue
        k = discrete_log[label]
        coset = k % num_cosets
        coset_points[coset].append(embedding[i])

    # 计算中心
    centers = {}
    for coset, points in coset_points.items():
        if points:
            centers[coset] = np.mean(points, axis=0)

    return centers, coset_points


def compute_adjacency_score(centers, num_cosets=12):
    """
    计算陪集邻接关系得分

    对于 Z_12，陪集 k 应该与 k-1 和 k+1 相邻（mod 12）
    """
    center_labels = sorted(centers.keys())
    center_matrix = np.array([centers[l] for l in center_labels])

    dist_matrix = cdist(center_matrix, center_matrix, metric='euclidean')

    correct_neighbors = 0
    total_neighbors = 0

    neighbor_details = []

    for i, coset in enumerate(center_labels):
        distances = dist_matrix[i].copy()
        distances[i] = np.inf

        nearest_indices = np.argsort(distances)[:2]
        nearest_cosets = [center_labels[idx] for idx in nearest_indices]

        expected_neighbors = [(coset - 1) % num_cosets, (coset + 1) % num_cosets]

        for neighbor in nearest_cosets:
            total_neighbors += 1
            if neighbor in expected_neighbors:
                correct_neighbors += 1

        neighbor_details.append({
            "coset": int(coset),
            "nearest_neighbors": [int(n) for n in nearest_cosets],
            "expected_neighbors": expected_neighbors,
            "correct": sum(1 for n in nearest_cosets if n in expected_neighbors),
        })

    score = correct_neighbors / total_neighbors if total_neighbors > 0 else 0

    return score, neighbor_details, dist_matrix, center_labels


def visualize_adjacency(embedding, labels, discrete_log, centers, neighbor_details, output_path, num_cosets=12):
    """可视化陪集邻接关系"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 计算每个点的陪集
    coset_labels = []
    for label in labels:
        if label == 0 or label not in discrete_log:
            coset_labels.append(-1)  # 无效
        else:
            coset_labels.append(discrete_log[label] % num_cosets)
    coset_labels = np.array(coset_labels)

    # 左图：按陪集着色
    ax1 = axes[0]
    valid_mask = coset_labels >= 0
    scatter = ax1.scatter(
        embedding[valid_mask, 0], embedding[valid_mask, 1],
        c=coset_labels[valid_mask], cmap='tab20', s=3, alpha=0.5
    )

    # 画陪集中心和连线
    for detail in neighbor_details:
        coset = detail["coset"]
        if coset not in centers:
            continue
        center = centers[coset]
        ax1.scatter(center[0], center[1], c='black', s=100, marker='x', linewidths=2)
        ax1.annotate(str(coset), center, fontsize=8, ha='center', va='bottom')

        for neighbor in detail["nearest_neighbors"]:
            if neighbor in centers:
                neighbor_center = centers[neighbor]
                expected = detail["expected_neighbors"]
                color = 'green' if neighbor in expected else 'red'
                ax1.plot(
                    [center[0], neighbor_center[0]],
                    [center[1], neighbor_center[1]],
                    color=color, alpha=0.5, linewidth=2
                )

    ax1.set_title(f"Coset Adjacency (k mod {num_cosets})\ngreen=correct, red=incorrect")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    plt.colorbar(scatter, ax=ax1, label=f"Coset (k mod {num_cosets})")

    # 右图：正确邻接比例
    ax2 = axes[1]
    correct_counts = [d["correct"] for d in neighbor_details]
    ax2.hist(correct_counts, bins=[0, 1, 2, 3], align='left', rwidth=0.8, edgecolor='black')
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['0/2 correct', '1/2 correct', '2/2 correct'])
    ax2.set_xlabel("Correct neighbors (out of 2)")
    ax2.set_ylabel("Number of cosets")
    ax2.set_title(f"Distribution of Correct Adjacencies")

    total_correct = sum(correct_counts)
    total_possible = len(correct_counts) * 2
    score = total_correct / total_possible
    ax2.text(0.5, 0.9, f"Overall score: {score:.1%}",
             transform=ax2.transAxes, fontsize=14, ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Computing discrete logarithms...")
    discrete_log = compute_discrete_logs(P, G)
    print(f"Computed discrete logs for {len(discrete_log)} elements (g={G})")

    print("Loading activations...")
    hidden, labels = load_activations(STEP)
    print(f"Loaded {len(labels)} samples")

    print("Running UMAP...")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(hidden)

    print("Computing coset centers...")
    num_cosets = 12
    centers, coset_points = compute_coset_centers(embedding, labels, discrete_log, num_cosets)
    print(f"Found {len(centers)} cosets (expected {num_cosets})")

    for coset, points in coset_points.items():
        print(f"  Coset {coset}: {len(points)} points")

    print("Analyzing adjacency relations...")
    score, neighbor_details, dist_matrix, center_labels = compute_adjacency_score(centers, num_cosets)

    print(f"\n{'='*50}")
    print(f"ADJACENCY ANALYSIS RESULTS (Multiplication mod {P})")
    print(f"{'='*50}")
    print(f"Analyzing {num_cosets} cosets (k mod {num_cosets})")
    print(f"Overall adjacency score: {score:.1%}")
    print(f"(100% = all nearest cosets are k±1)")
    print(f"(Random baseline ≈ 2/{num_cosets} ≈ {2/num_cosets:.1%})")

    correct_counts = [d["correct"] for d in neighbor_details]
    print(f"\nDistribution:")
    print(f"  2/2 correct: {correct_counts.count(2)} cosets ({correct_counts.count(2)/len(correct_counts):.1%})")
    print(f"  1/2 correct: {correct_counts.count(1)} cosets ({correct_counts.count(1)/len(correct_counts):.1%})")
    print(f"  0/2 correct: {correct_counts.count(0)} cosets ({correct_counts.count(0)/len(correct_counts):.1%})")

    print("\nNeighbor details:")
    for detail in neighbor_details:
        status = "✅" if detail["correct"] == 2 else ("🤔" if detail["correct"] == 1 else "❌")
        print(f"  Coset {detail['coset']:2d}: neighbors={detail['nearest_neighbors']}, expected={detail['expected_neighbors']} {status}")

    print("\nGenerating visualization...")
    visualize_adjacency(
        embedding, labels, discrete_log, centers, neighbor_details,
        os.path.join(OUTPUT_DIR, "coset_adjacency_visualization.png"),
        num_cosets
    )

    # 保存结果
    results = {
        "step": STEP,
        "p": P,
        "g": G,
        "operation": "multiplication",
        "num_cosets": num_cosets,
        "adjacency_score": score,
        "random_baseline": 2 / num_cosets,
        "distribution": {
            "2_correct": correct_counts.count(2),
            "1_correct": correct_counts.count(1),
            "0_correct": correct_counts.count(0),
        },
        "neighbor_details": neighbor_details,
    }

    with open(os.path.join(OUTPUT_DIR, "coset_adjacency_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")

    print(f"\n{'='*50}")
    print("CONCLUSION:")
    if score > 0.6:
        print(f"✅ Strong evidence for Z_{num_cosets} structure in cosets")
        print(f"   Adjacency score ({score:.1%}) >> random baseline ({2/num_cosets:.1%})")
    elif score > 0.3:
        print(f"🤔 Moderate evidence for coset structure")
        print(f"   Adjacency score ({score:.1%}) > random baseline ({2/num_cosets:.1%})")
    else:
        print(f"❌ Weak evidence for coset structure")
        print(f"   Adjacency score ({score:.1%}) ≈ random baseline ({2/num_cosets:.1%})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
