"""
实验 5：流形可视化
用 UMAP 把表示降到 2D，直观看到 Grokking 前后的结构变化

预期：
- Before Grokking：一团乱点（记忆态，每个样本独立编码）
- After Grokking：环/簇结构（泛化态，发现了模运算的循环群结构）

用法（在 Docker 容器内）：
    python /workspace/ai-theorys-study/arxiv/wechat67/code/visualize_manifold.py
"""

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import os
from glob import glob

# ============ 配置 ============
RESULTS_DIR = "/workspace/ai-theorys-study/arxiv/wechat67/results"
ACTIVATIONS_DIR = os.path.join(RESULTS_DIR, "activations")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "manifold_viz")

# 选择要可视化的时间点（根据之前实验，Grokking 大约在 step 40000-60000）
# 挑 3 个典型时间点：记忆期、过渡期、Grok 后
STEPS_TO_VISUALIZE = [5000, 30000, 100000]

# UMAP 参数
UMAP_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "metric": "cosine",  # 用余弦距离，和语义空间一致
    "random_state": 42,
}


def load_activations(step):
    """加载指定 step 的激活"""
    path = os.path.join(ACTIVATIONS_DIR, f"step_{step:06d}.npz")
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return None, None
    data = np.load(path)
    return data["hidden"], data["labels"]


def visualize_single_step(hidden, labels, step, output_path):
    """对单个时间点做 UMAP 可视化"""
    print(f"  Running UMAP for step {step}...")

    reducer = UMAP(**UMAP_PARAMS)
    embedding = reducer.fit_transform(hidden)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 10))

    # 按 label 着色（label 是 (a+b) mod 97 的结果，0-96）
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="hsv",  # 用 HSV 色环，体现循环群结构
        s=5,
        alpha=0.7
    )

    ax.set_title(f"Step {step:,}", fontsize=16)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # 添加 colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("(a+b) mod 97")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {output_path}")

    return embedding


def visualize_comparison(embeddings_dict, output_path):
    """并排对比多个时间点"""
    steps = sorted(embeddings_dict.keys())
    n = len(steps)

    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1:
        axes = [axes]

    for ax, step in zip(axes, steps):
        emb, labels = embeddings_dict[step]
        scatter = ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=labels,
            cmap="hsv",
            s=3,
            alpha=0.6
        )
        ax.set_title(f"Step {step:,}", fontsize=14)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    plt.suptitle("Manifold Structure: Before → During → After Grokking", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison saved to {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 检查可用的激活文件
    available_files = sorted(glob(os.path.join(ACTIVATIONS_DIR, "step_*.npz")))
    available_steps = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in available_files]
    print(f"Available steps: {available_steps[:5]}...{available_steps[-5:]} (total: {len(available_steps)})")

    # 调整要可视化的 steps（如果指定的不存在，找最近的）
    steps_to_use = []
    for target in STEPS_TO_VISUALIZE:
        closest = min(available_steps, key=lambda x: abs(x - target))
        steps_to_use.append(closest)
        if closest != target:
            print(f"  Adjusted {target} -> {closest}")

    # 可视化每个时间点
    embeddings_dict = {}
    for step in steps_to_use:
        print(f"\nProcessing step {step}...")
        hidden, labels = load_activations(step)
        if hidden is None:
            continue

        output_path = os.path.join(OUTPUT_DIR, f"manifold_step_{step:06d}.png")
        embedding = visualize_single_step(hidden, labels, step, output_path)
        embeddings_dict[step] = (embedding, labels)

    # 生成对比图
    if len(embeddings_dict) > 1:
        comparison_path = os.path.join(OUTPUT_DIR, "manifold_comparison.png")
        visualize_comparison(embeddings_dict, comparison_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
