"""
Grokking 实验一：画图
生成 Grokking 曲线 + 内在维度变化图

用法（在 Docker 容器内）：
    python /workspace/ai-theorys-study/arxiv/wechat67/plot_results.py
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_grokking_and_dimension(results_dir):
    """画 Grokking 曲线和内在维度变化"""

    # 加载数据
    log_path = os.path.join(results_dir, "train_log.json")
    dim_path = os.path.join(results_dir, "dimension_analysis.json")

    with open(log_path) as f:
        train_log = json.load(f)

    with open(dim_path) as f:
        dim_results = json.load(f)

    # 创建图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ===== 图 1: Grokking 曲线 =====
    ax1 = axes[0]

    steps = train_log["steps"]
    train_acc = train_log["train_acc"]
    test_acc = train_log["test_acc"]

    ax1.plot(steps, train_acc, label="Train Accuracy", color="blue", alpha=0.7)
    ax1.plot(steps, test_acc, label="Test Accuracy", color="red", linewidth=2)

    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right")
    ax1.set_title("Grokking: Delayed Generalization in Modular Addition", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 标记 Grokking 点
    for i, acc in enumerate(test_acc):
        if acc > 0.9:
            grok_step = steps[i]
            ax1.axvline(x=grok_step, color="green", linestyle="--", alpha=0.7, label=f"Grokking @ step {grok_step}")
            ax1.legend(loc="lower right")
            break

    # ===== 图 2: 内在维度 =====
    ax2 = axes[1]

    dim_steps = dim_results["steps"]
    pca_95 = dim_results["pca_dim_95"]
    pca_99 = dim_results["pca_dim_99"]
    twonn = dim_results["twonn_dim"]

    ax2.plot(dim_steps, pca_95, label="PCA (95% var)", color="purple", marker="o", markersize=3)
    ax2.plot(dim_steps, pca_99, label="PCA (99% var)", color="orange", marker="s", markersize=3)

    # TwoNN 可能有 None 值
    if any(x is not None for x in twonn):
        twonn_valid = [(s, d) for s, d in zip(dim_steps, twonn) if d is not None]
        if twonn_valid:
            ax2.plot([x[0] for x in twonn_valid], [x[1] for x in twonn_valid],
                     label="TwoNN", color="green", marker="^", markersize=3)

    ax2.set_xlabel("Training Steps", fontsize=12)
    ax2.set_ylabel("Intrinsic Dimension", fontsize=12)
    ax2.legend(loc="upper right")
    ax2.set_title("Intrinsic Dimension of Hidden Representations", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 标记 Grokking 点
    if "grok_step" in dir():
        ax2.axvline(x=grok_step, color="green", linestyle="--", alpha=0.7)

    plt.tight_layout()

    # 保存
    output_path = os.path.join(results_dir, "grokking_dimension.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")

    plt.show()


def plot_variance_explained(results_dir):
    """画 PCA 方差解释曲线（对比 Grokking 前后）"""

    activations_dir = os.path.join(results_dir, "activations")

    # 选择 Grokking 前后的两个 checkpoint
    # 需要根据实际结果调整
    from sklearn.decomposition import PCA

    fig, ax = plt.subplots(figsize=(10, 6))

    # 加载维度分析结果找 Grokking 点
    dim_path = os.path.join(results_dir, "dimension_analysis.json")
    with open(dim_path) as f:
        dim_results = json.load(f)

    # 找 Grokking 前后的步数
    test_acc = dim_results["test_acc"]
    steps = dim_results["steps"]

    grok_idx = None
    for i, acc in enumerate(test_acc):
        if acc is not None and acc > 0.9:
            grok_idx = i
            break

    if grok_idx is None or grok_idx < 2:
        print("Cannot find suitable before/after Grokking checkpoints")
        return

    before_step = steps[max(0, grok_idx - 2)]
    after_step = steps[min(len(steps) - 1, grok_idx + 2)]

    for step, label, color in [(before_step, "Before Grokking", "blue"),
                                (after_step, "After Grokking", "red")]:
        fpath = os.path.join(activations_dir, f"step_{step:06d}.npz")
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}")
            continue

        data = np.load(fpath)
        hidden = data["hidden"]

        pca = PCA()
        pca.fit(hidden)

        cumsum = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(range(1, len(cumsum) + 1), cumsum, label=f"{label} (step {step})", color=color, linewidth=2)

    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% threshold")
    ax.axhline(y=0.99, color="gray", linestyle=":", alpha=0.5, label="99% threshold")

    ax.set_xlabel("Number of Components", fontsize=12)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax.set_title("PCA Explained Variance: Before vs After Grokking", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(results_dir, "pca_variance.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    results_dir = "/workspace/ai-theorys-study/arxiv/wechat67/results"

    print("Plotting Grokking curve and dimension analysis...")
    plot_grokking_and_dimension(results_dir)

    print("\nPlotting PCA variance comparison...")
    plot_variance_explained(results_dir)
