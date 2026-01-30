"""
群结构邻接关系分析 - 模加法
验证"发现了结构"而非"分类变紧"

核心问题：
- 97 个簇在 UMAP 空间中的邻接关系是否符合群结构？
- 即：label s 的簇是否与 label s±1 的簇空间相邻？

方法：
1. 加载 Grokking 后的 UMAP embedding
2. 计算每个簇的中心
3. 构建邻接矩阵（基于簇中心距离）
4. 检查邻接关系是否符合循环群结构 Z_97

用法（在 Docker 容器内）：
    python /workspace/ai-theorys-study/arxiv/wechat67/exp_group1_addition/code/analyze_adjacency.py
"""

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from scipy.spatial.distance import cdist
import os
import json

# ============ 配置 ============
RESULTS_DIR = "/workspace/ai-theorys-study/arxiv/wechat67/exp_group1_addition/results"
ACTIVATIONS_DIR = os.path.join(RESULTS_DIR, "activations")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "adjacency_analysis")

# Grokking 后的时间点
STEP = 100000

P = 97  # 模数


def load_activations(step):
    """加载指定 step 的激活"""
    path = os.path.join(ACTIVATIONS_DIR, f"step_{step:06d}.npz")
    if not os.path.exists(path):
        # 尝试找最近的
        import glob
        files = sorted(glob.glob(os.path.join(ACTIVATIONS_DIR, "step_*.npz")))
        if not files:
            raise FileNotFoundError(f"No activation files found in {ACTIVATIONS_DIR}")
        path = files[-1]  # 用最后一个
        print(f"Using {path} instead of step {step}")
    data = np.load(path)
    return data["hidden"], data["labels"]


def compute_cluster_centers(embedding, labels, p):
    """计算每个 label 对应簇的中心"""
    centers = {}
    for label in range(p):
        mask = labels == label
        if mask.sum() > 0:
            centers[label] = embedding[mask].mean(axis=0)
    return centers


def compute_adjacency_score(centers, p):
    """
    计算邻接关系得分

    对于循环群 Z_p，label s 应该与 s-1 和 s+1 相邻
    得分 = 实际最近邻中有多少是"正确的"（即 s±1）
    """
    # 构建中心矩阵
    center_labels = sorted(centers.keys())
    center_matrix = np.array([centers[l] for l in center_labels])

    # 计算距离矩阵
    dist_matrix = cdist(center_matrix, center_matrix, metric='euclidean')

    # 对每个簇，找最近的 2 个邻居（排除自己）
    correct_neighbors = 0
    total_neighbors = 0

    neighbor_details = []

    for i, label in enumerate(center_labels):
        # 获取到其他簇的距离
        distances = dist_matrix[i].copy()
        distances[i] = np.inf  # 排除自己

        # 找最近的 2 个
        nearest_indices = np.argsort(distances)[:2]
        nearest_labels = [center_labels[idx] for idx in nearest_indices]

        # 检查是否是 s±1（mod p）
        expected_neighbors = [(label - 1) % p, (label + 1) % p]

        for neighbor in nearest_labels:
            total_neighbors += 1
            if neighbor in expected_neighbors:
                correct_neighbors += 1

        neighbor_details.append({
            "label": int(label),
            "nearest_neighbors": [int(n) for n in nearest_labels],
            "expected_neighbors": expected_neighbors,
            "correct": sum(1 for n in nearest_labels if n in expected_neighbors),
        })

    score = correct_neighbors / total_neighbors if total_neighbors > 0 else 0

    return score, neighbor_details, dist_matrix, center_labels


def visualize_adjacency(embedding, labels, centers, neighbor_details, output_path):
    """可视化邻接关系"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 左图：UMAP embedding，用箭头连接相邻簇
    ax1 = axes[0]
    scatter = ax1.scatter(
        embedding[:, 0], embedding[:, 1],
        c=labels, cmap='hsv', s=3, alpha=0.5
    )

    # 画出每个簇到其最近邻的连线
    for detail in neighbor_details:
        label = detail["label"]
        if label not in centers:
            continue
        center = centers[label]
        for neighbor in detail["nearest_neighbors"]:
            if neighbor in centers:
                neighbor_center = centers[neighbor]
                # 颜色：正确邻接=绿色，错误=红色
                expected = detail["expected_neighbors"]
                color = 'green' if neighbor in expected else 'red'
                ax1.plot(
                    [center[0], neighbor_center[0]],
                    [center[1], neighbor_center[1]],
                    color=color, alpha=0.3, linewidth=0.5
                )

    ax1.set_title(f"Adjacency Relations (green=correct, red=incorrect)")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")

    # 右图：正确邻接比例的分布
    ax2 = axes[1]
    correct_counts = [d["correct"] for d in neighbor_details]
    ax2.hist(correct_counts, bins=[0, 1, 2, 3], align='left', rwidth=0.8, edgecolor='black')
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['0/2 correct', '1/2 correct', '2/2 correct'])
    ax2.set_xlabel("Correct neighbors (out of 2)")
    ax2.set_ylabel("Number of clusters")
    ax2.set_title(f"Distribution of Correct Adjacencies")

    # 统计
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

    print("Loading activations...")
    hidden, labels = load_activations(STEP)
    print(f"Loaded {len(labels)} samples")

    print("Running UMAP...")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(hidden)

    print("Computing cluster centers...")
    centers = compute_cluster_centers(embedding, labels, P)
    print(f"Found {len(centers)} clusters (expected {P})")

    print("Analyzing adjacency relations...")
    score, neighbor_details, dist_matrix, center_labels = compute_adjacency_score(centers, P)

    print(f"\n{'='*50}")
    print(f"ADJACENCY ANALYSIS RESULTS (Addition mod {P})")
    print(f"{'='*50}")
    print(f"Overall adjacency score: {score:.1%}")
    print(f"(100% = all nearest neighbors are s±1)")
    print(f"(Random baseline ≈ 2/{P} ≈ {2/P:.1%})")

    # 详细统计
    correct_counts = [d["correct"] for d in neighbor_details]
    print(f"\nDistribution:")
    print(f"  2/2 correct: {correct_counts.count(2)} clusters ({correct_counts.count(2)/len(correct_counts):.1%})")
    print(f"  1/2 correct: {correct_counts.count(1)} clusters ({correct_counts.count(1)/len(correct_counts):.1%})")
    print(f"  0/2 correct: {correct_counts.count(0)} clusters ({correct_counts.count(0)/len(correct_counts):.1%})")

    # 可视化
    print("\nGenerating visualization...")
    visualize_adjacency(
        embedding, labels, centers, neighbor_details,
        os.path.join(OUTPUT_DIR, "adjacency_visualization.png")
    )

    # 保存结果
    results = {
        "step": STEP,
        "p": P,
        "operation": "addition",
        "num_clusters": len(centers),
        "adjacency_score": score,
        "random_baseline": 2 / P,
        "distribution": {
            "2_correct": correct_counts.count(2),
            "1_correct": correct_counts.count(1),
            "0_correct": correct_counts.count(0),
        },
        "neighbor_details": neighbor_details,
    }

    with open(os.path.join(OUTPUT_DIR, "adjacency_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")

    # 结论
    print(f"\n{'='*50}")
    print("CONCLUSION:")
    if score > 0.5:
        print(f"✅ Strong evidence for group structure discovery")
        print(f"   Adjacency score ({score:.1%}) >> random baseline ({2/P:.1%})")
    elif score > 0.2:
        print(f"🤔 Moderate evidence for group structure")
        print(f"   Adjacency score ({score:.1%}) > random baseline ({2/P:.1%})")
    else:
        print(f"❌ Weak evidence for group structure")
        print(f"   Adjacency score ({score:.1%}) ≈ random baseline ({2/P:.1%})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
