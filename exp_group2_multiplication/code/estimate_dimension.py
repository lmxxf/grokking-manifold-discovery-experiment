"""
Grokking 实验一：内在维度估计
对每个 checkpoint 的激活计算内在维度，验证"Grokking = 维度突变"假说

用法（在 Docker 容器内）：
    pip install scikit-dimension  # 首次运行需要安装
    python /workspace/ai-theorys-study/arxiv/wechat67/estimate_dimension.py
"""

import numpy as np
import os
import json
from glob import glob
from sklearn.decomposition import PCA

# 尝试导入 TwoNN，如果没有就只用 PCA
try:
    from skdim.id import TwoNN
    HAS_TWONN = True
except ImportError:
    print("Warning: scikit-dimension not installed, using PCA only")
    print("Install with: pip install scikit-dimension")
    HAS_TWONN = False


def estimate_pca_dimension(X, threshold=0.95):
    """用 PCA 估计内在维度（保留 threshold 方差所需的维度数）"""
    pca = PCA()
    pca.fit(X)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= threshold) + 1

    return n_components, pca.explained_variance_ratio_


def estimate_twonn_dimension(X):
    """用 TwoNN 估计内在维度（Facco et al. 2017）"""
    if not HAS_TWONN:
        return None

    try:
        twonn = TwoNN()
        dim = twonn.fit_transform(X)
        return dim
    except Exception as e:
        print(f"TwoNN failed: {e}")
        return None


def analyze_activations(results_dir):
    """分析所有 checkpoint 的激活"""
    activations_dir = os.path.join(results_dir, "activations")

    # 找到所有激活文件
    files = sorted(glob(os.path.join(activations_dir, "step_*.npz")))

    if not files:
        print(f"No activation files found in {activations_dir}")
        return

    print(f"Found {len(files)} activation files")

    # 加载训练日志
    log_path = os.path.join(results_dir, "train_log.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            train_log = json.load(f)
    else:
        train_log = None

    # 分析每个 checkpoint
    results = {
        "steps": [],
        "pca_dim_95": [],      # 保留 95% 方差的维度
        "pca_dim_99": [],      # 保留 99% 方差的维度
        "twonn_dim": [],       # TwoNN 估计
        "test_acc": [],        # 对应的测试准确率
    }

    for f in files:
        # 解析步数
        step = int(os.path.basename(f).split("_")[1].split(".")[0])

        # 加载激活
        data = np.load(f)
        hidden = data["hidden"]  # (n_samples, embed_dim)

        print(f"\nStep {step}: shape = {hidden.shape}")

        # PCA 维度估计
        pca_95, var_ratio = estimate_pca_dimension(hidden, threshold=0.95)
        pca_99, _ = estimate_pca_dimension(hidden, threshold=0.99)

        print(f"  PCA dim (95% var): {pca_95}")
        print(f"  PCA dim (99% var): {pca_99}")

        # TwoNN 维度估计
        twonn = estimate_twonn_dimension(hidden)
        if twonn is not None:
            print(f"  TwoNN dim: {twonn:.2f}")

        # 获取对应的测试准确率
        if train_log and step in train_log["steps"]:
            idx = train_log["steps"].index(step)
            test_acc = train_log["test_acc"][idx]
        else:
            test_acc = None

        # 记录
        results["steps"].append(step)
        results["pca_dim_95"].append(int(pca_95))
        results["pca_dim_99"].append(int(pca_99))
        results["twonn_dim"].append(float(twonn) if twonn is not None else None)
        results["test_acc"].append(test_acc)

    # 保存结果
    output_path = os.path.join(results_dir, "dimension_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


def find_grokking_point(results):
    """找到 Grokking 发生的步数（测试准确率首次超过 90%）"""
    for i, acc in enumerate(results["test_acc"]):
        if acc is not None and acc > 0.9:
            return results["steps"][i]
    return None


if __name__ == "__main__":
    results_dir = "/workspace/ai-theorys-study/arxiv/wechat67/exp_group2_multiplication/results"

    results = analyze_activations(results_dir)

    if results:
        grok_step = find_grokking_point(results)
        if grok_step:
            print(f"\nGrokking detected at step {grok_step}")

            # 找到 Grokking 前后的维度
            idx = results["steps"].index(grok_step)
            if idx > 0:
                before_dim = results["pca_dim_95"][idx - 1]
                after_dim = results["pca_dim_95"][idx]
                print(f"PCA dim before: {before_dim}, after: {after_dim}")

                if before_dim > after_dim:
                    print(">>> 维度下降！支持流形发现假说")
                else:
                    print(">>> 维度未下降，需要进一步分析")
        else:
            print("\nGrokking not detected (test acc never exceeded 90%)")
