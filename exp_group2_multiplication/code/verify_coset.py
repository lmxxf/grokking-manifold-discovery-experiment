"""
验证实验：12 个簇是否对应 k mod 12（陪集结构）

假说：模型学到了离散对数的商群坐标 k mod 12，而非完整坐标 k

验证方法：
1. 找 97 的原根 g
2. 对每个 label y，计算离散对数 k 使得 g^k ≡ y (mod 97)
3. 用 UMAP 降维，按 k mod 12 着色
4. 如果簇按 k mod 12 分组，假说成立

用法：
    python verify_coset.py
"""

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import os

# ============ 配置 ============
RESULTS_DIR = "/workspace/ai-theorys-study/arxiv/wechat67/exp_group2_multiplication/results"
ACTIVATIONS_DIR = os.path.join(RESULTS_DIR, "activations")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "coset_analysis")

P = 97  # 质数
STEP = 100000  # 用 Grok 后的激活

# ============ 数论工具 ============
def find_primitive_root(p):
    """找质数 p 的原根（生成元）"""
    # 对于质数 p，原根 g 满足 g 的阶是 p-1
    # 即 g^1, g^2, ..., g^(p-1) 遍历 1 到 p-1
    phi = p - 1

    # 分解 phi 的质因数
    factors = []
    n = phi
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)

    # 检验每个候选
    for g in range(2, p):
        is_root = True
        for f in factors:
            if pow(g, phi // f, p) == 1:
                is_root = False
                break
        if is_root:
            return g
    return None


def discrete_log(g, y, p):
    """计算离散对数：找 k 使得 g^k ≡ y (mod p)"""
    # Baby-step giant-step 或暴力搜索（p=97 很小，暴力即可）
    for k in range(p - 1):
        if pow(g, k, p) == y:
            return k
    return None


def build_dlog_table(g, p):
    """构建离散对数表：y -> k where g^k ≡ y (mod p)"""
    table = {}
    for k in range(p - 1):
        y = pow(g, k, p)
        table[y] = k
    return table


# ============ 主程序 ============
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 找原根
    g = find_primitive_root(P)
    print(f"原根 g = {g}")
    print(f"验证：g^96 mod 97 = {pow(g, 96, P)} (应该是 1)")

    # 2. 构建离散对数表
    dlog_table = build_dlog_table(g, P)
    print(f"离散对数表大小：{len(dlog_table)}")

    # 3. 加载激活
    path = os.path.join(ACTIVATIONS_DIR, f"step_{STEP:06d}.npz")
    data = np.load(path)
    hidden = data["hidden"]
    labels = data["labels"]

    print(f"激活形状：{hidden.shape}")
    print(f"Label 范围：{labels.min()} - {labels.max()}")

    # 4. 计算每个样本的 k mod 12
    k_mod_12 = []
    valid_mask = []
    for y in labels:
        if y in dlog_table:
            k = dlog_table[y]
            k_mod_12.append(k % 12)
            valid_mask.append(True)
        else:
            # y=0 不在乘法群中（但乘法实验排除了 0）
            k_mod_12.append(-1)
            valid_mask.append(False)

    k_mod_12 = np.array(k_mod_12)
    valid_mask = np.array(valid_mask)

    print(f"有效样本：{valid_mask.sum()} / {len(valid_mask)}")
    print(f"k mod 12 分布：{np.bincount(k_mod_12[valid_mask])}")

    # 5. UMAP 降维
    print("Running UMAP...")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    embedding = reducer.fit_transform(hidden)

    # 6. 可视化：按 label 着色 vs 按 k mod 12 着色
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：按 label 着色（原来的方式）
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        embedding[:, 0], embedding[:, 1],
        c=labels, cmap="hsv", s=3, alpha=0.6
    )
    ax1.set_title(f"Colored by Label (y)", fontsize=14)
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    plt.colorbar(scatter1, ax=ax1, label="y = (a*b) mod 97")

    # 右图：按 k mod 12 着色
    ax2 = axes[1]
    scatter2 = ax2.scatter(
        embedding[valid_mask, 0], embedding[valid_mask, 1],
        c=k_mod_12[valid_mask], cmap="tab20", s=3, alpha=0.6
    )
    ax2.set_title(f"Colored by k mod 12 (Coset ID)", fontsize=14)
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    plt.colorbar(scatter2, ax=ax2, label="k mod 12")

    plt.suptitle(f"Step {STEP:,}: Label vs Coset Structure", fontsize=16, y=1.02)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "label_vs_coset.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")

    # 7. 分析：每个簇的 k mod 12 分布
    # 用 KMeans 找 12 个簇，看每个簇的 k mod 12 是否一致
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embedding)

    print("\n" + "="*60)
    print("簇分析：每个 KMeans 簇的 k mod 12 分布")
    print("="*60)

    for c in range(12):
        mask = (cluster_ids == c) & valid_mask
        if mask.sum() == 0:
            continue
        k_values = k_mod_12[mask]
        counts = np.bincount(k_values, minlength=12)
        dominant = np.argmax(counts)
        purity = counts[dominant] / counts.sum()
        print(f"簇 {c:2d}: 主要 k mod 12 = {dominant:2d}, 纯度 = {purity:.2%}, 分布 = {counts.tolist()}")

    # 8. 保存分析结果
    analysis = {
        "primitive_root": g,
        "step": STEP,
        "n_samples": len(labels),
        "n_valid": int(valid_mask.sum()),
    }

    import json
    with open(os.path.join(OUTPUT_DIR, "coset_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nAnalysis saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
