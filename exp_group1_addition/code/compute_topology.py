"""
Grokking 实验二：表示的拓扑结构（持续同调）
验证 Grokking 后表示是否形成环结构（对应循环群 Z_p）

用法（在 Docker 容器内）：
    pip install ripser persim -i https://pypi.tuna.tsinghua.edu.cn/simple
    python /workspace/ai-theorys-study/arxiv/wechat67/compute_topology.py
"""

import numpy as np
import os
import json
import matplotlib.pyplot as plt

# 尝试导入 ripser
try:
    from ripser import ripser
    from persim import plot_diagrams
    HAS_RIPSER = True
except ImportError:
    print("Warning: ripser not installed")
    print("Install with: pip install ripser persim")
    HAS_RIPSER = False


def compute_persistence(X, max_dim=1):
    """计算持续同调，返回 persistence diagrams"""
    if not HAS_RIPSER:
        return None

    # ripser 直接接受点云
    result = ripser(X, maxdim=max_dim)
    return result['dgms']  # 返回 [H_0 diagram, H_1 diagram, ...]


def extract_betti_numbers(diagrams, threshold=0.1):
    """从 persistence diagrams 提取 Betti 数"""
    if diagrams is None:
        return None

    betti = {}

    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            betti[f"b{dim}"] = 0
            betti[f"b{dim}_max_persistence"] = 0
            continue

        # 计算持久性 (death - birth)
        persistence = dgm[:, 1] - dgm[:, 0]

        # 处理无穷大
        finite_mask = np.isfinite(persistence)

        if dim == 0:
            # H_0: 统计有限持久性超过阈值的 + 无穷远的（全局连通）
            significant = int(np.sum(persistence[finite_mask] > threshold))
            betti[f"b{dim}"] = significant + int(np.sum(~finite_mask))
        else:
            # H_1: 只统计有限的
            betti[f"b{dim}"] = int(np.sum(persistence[finite_mask] > threshold))

        # 记录最大持久性
        finite_pers = persistence[finite_mask]
        betti[f"b{dim}_max_persistence"] = float(np.max(finite_pers)) if len(finite_pers) > 0 else 0.0

    return betti


def analyze_topology(results_dir):
    """分析 Grokking 前后的拓扑结构"""
    if not HAS_RIPSER:
        print("ripser not available, skipping topology analysis")
        return None

    activations_dir = os.path.join(results_dir, "activations")

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
        return None

    # 选择 Grokking 前后的 checkpoint
    before_step = steps[max(0, grok_idx - 2)]
    after_step = steps[min(len(steps) - 1, grok_idx + 5)]

    print(f"Grokking at step {steps[grok_idx]}")
    print(f"Analyzing: before={before_step}, after={after_step}")

    results = {
        "grokking_step": steps[grok_idx],
        "before_step": before_step,
        "after_step": after_step,
        "before": {},
        "after": {}
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (step, label) in enumerate([(before_step, "before"), (after_step, "after")]):
        fpath = os.path.join(activations_dir, f"step_{step:06d}.npz")
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}")
            continue

        print(f"\nProcessing {label} Grokking (step {step})...")

        # 加载激活
        data = np.load(fpath)
        hidden = data["hidden"]  # (n_samples, embed_dim)

        print(f"  Shape: {hidden.shape}")

        # 为了计算效率，降采样
        max_points = 500  # ripser 对大数据集很慢
        if hidden.shape[0] > max_points:
            print(f"  Subsampling to {max_points} points...")
            indices = np.random.choice(hidden.shape[0], max_points, replace=False)
            hidden = hidden[indices]

        # 计算持续同调
        print(f"  Computing persistence (this may take a moment)...")
        diagrams = compute_persistence(hidden, max_dim=1)

        if diagrams is not None:
            # 提取 Betti 数
            betti = extract_betti_numbers(diagrams, threshold=0.1)
            results[label] = betti
            print(f"  Betti numbers: {betti}")

            # 画 persistence diagram
            ax = axes[idx]

            h0, h1 = diagrams[0], diagrams[1]

            # 找最大值用于画对角线
            all_finite = []
            for dgm in diagrams:
                finite_vals = dgm[np.isfinite(dgm)]
                if len(finite_vals) > 0:
                    all_finite.extend(finite_vals)
            max_val = max(all_finite) if all_finite else 1

            # 画对角线
            ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3)

            # 画 H_0 (蓝色)
            h0_finite = h0[np.isfinite(h0[:, 1])]
            if len(h0_finite) > 0:
                ax.scatter(h0_finite[:, 0], h0_finite[:, 1], c='blue', alpha=0.5, s=30, label='H₀ (components)')

            # 画 H_1 (红色) - 这是我们关心的环结构
            h1_finite = h1[np.isfinite(h1[:, 1])]
            if len(h1_finite) > 0:
                ax.scatter(h1_finite[:, 0], h1_finite[:, 1], c='red', alpha=0.8, s=60, label='H₁ (loops)')

            ax.set_xlabel("Birth", fontsize=12)
            ax.set_ylabel("Death", fontsize=12)
            ax.set_title(f"{'Before' if label == 'before' else 'After'} Grokking (step {step})\n"
                        f"β₀={betti['b0']}, β₁={betti['b1']}", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, max_val * 1.1)
            ax.set_ylim(-0.05, max_val * 1.1)

    plt.tight_layout()

    # 保存图
    output_path = os.path.join(results_dir, "topology_persistence.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")

    plt.close()

    # 保存结果
    results_path = os.path.join(results_dir, "topology_analysis.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")

    return results


def interpret_results(results):
    """解读拓扑分析结果"""
    if results is None:
        return

    print("\n" + "="*60)
    print("拓扑分析结果解读")
    print("="*60)

    before = results.get("before", {})
    after = results.get("after", {})

    print(f"\n预期（如果流形发现假说成立）：")
    print(f"  - 模运算 (a+b) mod p 的真实结构是循环群 Z_p ≈ 圆周 S¹")
    print(f"  - 圆周的 Betti 数：β₀=1（一个连通分量），β₁=1（一个洞/环）")
    print(f"  - Grokking 后应该看到 β₁ 的持久性增强（更稳定的环结构）")

    print(f"\n实际观测：")
    print(f"  Before Grokking: β₀={before.get('b0', '?')}, β₁={before.get('b1', '?')}, "
          f"H₁ max persistence={before.get('b1_max_persistence', '?'):.3f}")
    print(f"  After Grokking:  β₀={after.get('b0', '?')}, β₁={after.get('b1', '?')}, "
          f"H₁ max persistence={after.get('b1_max_persistence', '?'):.3f}")

    b1_pers_before = before.get('b1_max_persistence', 0)
    b1_pers_after = after.get('b1_max_persistence', 0)

    if b1_pers_after > b1_pers_before * 1.2:  # 20% 提升算显著
        print(f"\n>>> H₁ 持久性增强！支持流形发现假说 ✅")
        print(f"    Grokking 后环结构更稳定")
    elif b1_pers_after > b1_pers_before:
        print(f"\n>>> H₁ 持久性略有增加，需要更多实验确认")
    else:
        print(f"\n>>> H₁ 持久性无明显变化或下降，需要进一步分析")


if __name__ == "__main__":
    np.random.seed(42)  # 可重复性
    results_dir = "/workspace/ai-theorys-study/arxiv/wechat67/results"

    results = analyze_topology(results_dir)
    interpret_results(results)
