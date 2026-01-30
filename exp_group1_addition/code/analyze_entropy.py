"""
Grokking 实验三：激活动态分析
分析训练过程中激活模式的变化（稀疏性、标准差、范数）

用法（在 Docker 容器内）：
    python /workspace/ai-theorys-study/arxiv/wechat67/code/analyze_entropy.py
"""

import numpy as np
import os
import json
import matplotlib.pyplot as plt
from glob import glob


def analyze_activation_dynamics(results_dir):
    """分析训练过程中激活模式的变化"""

    # 加载训练日志
    log_path = os.path.join(results_dir, "train_log.json")
    with open(log_path) as f:
        train_log = json.load(f)

    activations_dir = os.path.join(results_dir, "activations")
    files = sorted(glob(os.path.join(activations_dir, "step_*.npz")))

    if not files:
        print("No activation files found")
        return None

    print(f"Found {len(files)} activation files")

    results = {
        "steps": [],
        "activation_sparsity": [],      # 激活的稀疏性（接近0的比例）
        "activation_std": [],           # 激活的标准差
        "activation_max": [],           # 激活的最大值
        "activation_l2_norm": [],       # 激活的 L2 范数
    }

    for f in files:
        step = int(os.path.basename(f).split("_")[1].split(".")[0])
        data = np.load(f)
        hidden = data["hidden"]  # (n_samples, embed_dim)

        # 计算各种统计量
        sparsity = np.mean(np.abs(hidden) < 0.1)
        std = np.std(hidden)
        max_val = np.max(np.abs(hidden))
        l2_norm = np.mean(np.linalg.norm(hidden, axis=1))

        results["steps"].append(step)
        results["activation_sparsity"].append(float(sparsity))
        results["activation_std"].append(float(std))
        results["activation_max"].append(float(max_val))
        results["activation_l2_norm"].append(float(l2_norm))

    # 画图
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    steps = results["steps"]

    # 找 Grokking 点
    grok_step = None
    for i, acc in enumerate(train_log["test_acc"]):
        if acc > 0.9:
            grok_step = train_log["steps"][i]
            break

    # 图1: 测试准确率
    ax1 = axes[0]
    ax1.plot(train_log["steps"], train_log["test_acc"], color="red", linewidth=2)
    ax1.set_ylabel("Test Accuracy", fontsize=12)
    ax1.set_title("Grokking Dynamics: Accuracy and Activation Patterns", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    if grok_step:
        ax1.axvline(x=grok_step, color="green", linestyle="--", alpha=0.7, label=f"Grokking @ {grok_step}")
        ax1.legend()

    # 图2: 激活稀疏性
    ax2 = axes[1]
    ax2.plot(steps, results["activation_sparsity"], color="blue", marker=".", markersize=2)
    ax2.set_ylabel("Sparsity (|x|<0.1)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    if grok_step:
        ax2.axvline(x=grok_step, color="green", linestyle="--", alpha=0.7)

    # 图3: 激活标准差
    ax3 = axes[2]
    ax3.plot(steps, results["activation_std"], color="purple", marker=".", markersize=2)
    ax3.set_ylabel("Activation Std", fontsize=12)
    ax3.grid(True, alpha=0.3)
    if grok_step:
        ax3.axvline(x=grok_step, color="green", linestyle="--", alpha=0.7)

    # 图4: L2 范数
    ax4 = axes[3]
    ax4.plot(steps, results["activation_l2_norm"], color="orange", marker=".", markersize=2)
    ax4.set_xlabel("Training Steps", fontsize=12)
    ax4.set_ylabel("Mean L2 Norm", fontsize=12)
    ax4.grid(True, alpha=0.3)
    if grok_step:
        ax4.axvline(x=grok_step, color="green", linestyle="--", alpha=0.7)

    plt.tight_layout()

    output_path = os.path.join(results_dir, "activation_dynamics.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")

    plt.close()

    # 保存结果
    results_path = os.path.join(results_dir, "activation_analysis.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")

    # 分析
    print("\n" + "="*60)
    print("激活动态分析结果")
    print("="*60)

    if grok_step:
        grok_idx = None
        for i, s in enumerate(results["steps"]):
            if s >= grok_step:
                grok_idx = i
                break

        if grok_idx and grok_idx > 0:
            before_idx = grok_idx - 1
            after_idx = min(grok_idx + 5, len(results["steps"]) - 1)

            print(f"\nGrokking 前后对比 (step {results['steps'][before_idx]} vs {results['steps'][after_idx]}):")
            print(f"  Sparsity: {results['activation_sparsity'][before_idx]:.3f} → {results['activation_sparsity'][after_idx]:.3f}")
            print(f"  Std: {results['activation_std'][before_idx]:.3f} → {results['activation_std'][after_idx]:.3f}")
            print(f"  L2 Norm: {results['activation_l2_norm'][before_idx]:.3f} → {results['activation_l2_norm'][after_idx]:.3f}")

    return results


if __name__ == "__main__":
    results_dir = "/workspace/ai-theorys-study/arxiv/wechat67/results"
    results = analyze_activation_dynamics(results_dir)
