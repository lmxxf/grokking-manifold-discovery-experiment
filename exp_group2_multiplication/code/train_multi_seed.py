"""
多 seed 稳定性实验 - 模乘法
跑 3 个额外的随机种子，验证核心发现的可复现性

用法（在 Docker 容器内）：
    python /workspace/ai-theorys-study/arxiv/wechat67/exp_group2_multiplication/code/train_multi_seed.py

输出目录结构：
    results_multi_seed/
        seed_1001/
        seed_1002/
        seed_1003/
        summary.json  # 汇总统计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
from tqdm import tqdm

# ============ 配置 ============
BASE_CONFIG = {
    "p": 97,                    # 质数，定义模运算 (a * b) mod p
    "train_ratio": 0.3,         # 训练集比例
    "embed_dim": 128,           # embedding 维度
    "num_heads": 4,             # attention heads
    "num_layers": 2,            # transformer 层数
    "lr": 1e-3,                 # 学习率
    "weight_decay": 1.0,        # 权重衰减
    "batch_size": 512,          # batch size
    "total_steps": 150000,      # 总训练步数
    "eval_every": 1000,         # 每隔多少步评估一次
    "save_activations_every": 5000,  # 每隔多少步保存激活（比原来稀疏，节省空间）
    "operation": "multiplication",
}

# 要跑的随机种子
SEEDS = [1001, 1002, 1003]

OUTPUT_BASE = "/workspace/ai-theorys-study/arxiv/wechat67/exp_group2_multiplication/results_multi_seed"


# ============ 数据生成 ============
def generate_modular_multiplication_data(p, train_ratio, seed=42):
    """
    生成模乘法数据集：(a, b) -> (a * b) mod p
    乘法群 Z_p^* 不包含 0，所以 a, b ∈ {1, 2, ..., p-1}
    """
    np.random.seed(seed)

    all_pairs = [(a, b) for a in range(1, p) for b in range(1, p)]
    np.random.shuffle(all_pairs)

    n_train = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:n_train]
    test_pairs = all_pairs[n_train:]

    def pairs_to_tensors(pairs):
        a = torch.tensor([x[0] for x in pairs], dtype=torch.long)
        b = torch.tensor([x[1] for x in pairs], dtype=torch.long)
        y = torch.tensor([(x[0] * x[1]) % p for x in pairs], dtype=torch.long)
        return a, b, y

    train_a, train_b, train_y = pairs_to_tensors(train_pairs)
    test_a, test_b, test_y = pairs_to_tensors(test_pairs)

    return (train_a, train_b, train_y), (test_a, test_b, test_y)


# ============ 模型定义 ============
class GrokkingTransformer(nn.Module):
    """简化的 Transformer 用于模运算任务"""

    def __init__(self, p, embed_dim, num_heads, num_layers):
        super().__init__()
        self.p = p
        self.embed_dim = embed_dim

        self.embed_a = nn.Embedding(p, embed_dim)
        self.embed_b = nn.Embedding(p, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(2, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(embed_dim, p)
        self.last_hidden = None

    def forward(self, a, b, save_hidden=False):
        emb_a = self.embed_a(a)
        emb_b = self.embed_b(b)
        x = torch.stack([emb_a, emb_b], dim=1)
        x = x + self.pos_embed.unsqueeze(0)
        x = self.transformer(x)

        if save_hidden:
            self.last_hidden = x[:, 0, :].detach().cpu()

        logits = self.output(x[:, 0, :])
        return logits


# ============ 单个 seed 训练 ============
def train_single_seed(seed, output_dir):
    """训练单个 seed"""
    print(f"\n{'='*50}")
    print(f"Training MULTIPLICATION with seed {seed}")
    print(f"{'='*50}")

    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "activations"), exist_ok=True)

    # 生成数据
    (train_a, train_b, train_y), (test_a, test_b, test_y) = generate_modular_multiplication_data(
        BASE_CONFIG["p"], BASE_CONFIG["train_ratio"], seed
    )

    train_dataset = TensorDataset(train_a, train_b, train_y)
    test_dataset = TensorDataset(test_a, test_b, test_y)

    train_loader = DataLoader(train_dataset, batch_size=BASE_CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BASE_CONFIG["batch_size"], shuffle=False)

    # 模型
    model = GrokkingTransformer(
        p=BASE_CONFIG["p"],
        embed_dim=BASE_CONFIG["embed_dim"],
        num_heads=BASE_CONFIG["num_heads"],
        num_layers=BASE_CONFIG["num_layers"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_CONFIG["lr"],
        weight_decay=BASE_CONFIG["weight_decay"]
    )

    # 训练日志
    log = {
        "seed": seed,
        "operation": "multiplication",
        "steps": [],
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    # 关键指标追踪
    first_grok_step = None  # 首次达到 90% 测试准确率的 step
    oscillation_count = 0   # 准确率震荡次数（从 >90% 跌到 <50%）
    last_above_90 = False

    # 训练
    step = 0
    pbar = tqdm(total=BASE_CONFIG["total_steps"], desc=f"Seed {seed}")

    while step < BASE_CONFIG["total_steps"]:
        for batch_a, batch_b, batch_y in train_loader:
            if step >= BASE_CONFIG["total_steps"]:
                break

            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_a, batch_b)
            loss = F.cross_entropy(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            pbar.update(1)

            # 评估
            if step % BASE_CONFIG["eval_every"] == 0:
                model.eval()

                # 训练集准确率
                train_correct = 0
                train_total = 0
                for ba, bb, by in train_loader:
                    ba, bb, by = ba.to(device), bb.to(device), by.to(device)
                    with torch.no_grad():
                        pred = model(ba, bb).argmax(dim=1)
                    train_correct += (pred == by).sum().item()
                    train_total += len(by)
                train_acc = train_correct / train_total

                # 测试集准确率
                test_correct = 0
                test_total = 0
                for ba, bb, by in test_loader:
                    ba, bb, by = ba.to(device), bb.to(device), by.to(device)
                    with torch.no_grad():
                        pred = model(ba, bb).argmax(dim=1)
                    test_correct += (pred == by).sum().item()
                    test_total += len(by)
                test_acc = test_correct / test_total

                # 记录
                log["steps"].append(step)
                log["train_loss"].append(loss.item())
                log["train_acc"].append(train_acc)
                log["test_acc"].append(test_acc)

                # 追踪关键指标
                if first_grok_step is None and test_acc >= 0.9:
                    first_grok_step = step

                current_above_90 = test_acc >= 0.9
                if last_above_90 and test_acc < 0.5:
                    oscillation_count += 1
                last_above_90 = current_above_90

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "train": f"{train_acc:.3f}",
                    "test": f"{test_acc:.3f}"
                })

                model.train()

            # 保存激活（稀疏保存）
            if step % BASE_CONFIG["save_activations_every"] == 0:
                model.eval()

                all_hidden = []
                all_labels = []
                for ba, bb, by in test_loader:
                    ba, bb = ba.to(device), bb.to(device)
                    with torch.no_grad():
                        _ = model(ba, bb, save_hidden=True)
                    all_hidden.append(model.last_hidden)
                    all_labels.append(by)

                all_hidden = torch.cat(all_hidden, dim=0).numpy()
                all_labels = torch.cat(all_labels, dim=0).numpy()

                np.savez(
                    os.path.join(output_dir, "activations", f"step_{step:06d}.npz"),
                    hidden=all_hidden,
                    labels=all_labels
                )

                model.train()

    pbar.close()

    # 保存日志
    log["first_grok_step"] = first_grok_step
    log["oscillation_count"] = oscillation_count
    log["final_test_acc"] = log["test_acc"][-1] if log["test_acc"] else 0

    with open(os.path.join(output_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    # 保存配置
    config = BASE_CONFIG.copy()
    config["seed"] = seed
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Seed {seed}: first_grok={first_grok_step}, oscillations={oscillation_count}, final_acc={log['final_test_acc']:.4f}")

    return log


# ============ 主函数 ============
def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    results = []

    for seed in SEEDS:
        output_dir = os.path.join(OUTPUT_BASE, f"seed_{seed}")
        log = train_single_seed(seed, output_dir)
        results.append({
            "seed": seed,
            "first_grok_step": log["first_grok_step"],
            "oscillation_count": log["oscillation_count"],
            "final_test_acc": log["final_test_acc"],
        })

    # 汇总统计
    summary = {
        "operation": "multiplication",
        "seeds": SEEDS,
        "results": results,
        "statistics": {
            "first_grok_step": {
                "values": [r["first_grok_step"] for r in results if r["first_grok_step"]],
                "mean": np.mean([r["first_grok_step"] for r in results if r["first_grok_step"]]) if any(r["first_grok_step"] for r in results) else None,
                "std": np.std([r["first_grok_step"] for r in results if r["first_grok_step"]]) if any(r["first_grok_step"] for r in results) else None,
            },
            "oscillation_count": {
                "values": [r["oscillation_count"] for r in results],
                "mean": np.mean([r["oscillation_count"] for r in results]),
                "std": np.std([r["oscillation_count"] for r in results]),
            },
            "final_test_acc": {
                "values": [r["final_test_acc"] for r in results],
                "mean": np.mean([r["final_test_acc"] for r in results]),
                "std": np.std([r["final_test_acc"] for r in results]),
            },
        }
    }

    with open(os.path.join(OUTPUT_BASE, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*50)
    print("Multi-seed MULTIPLICATION experiment complete!")
    print("="*50)
    if summary['statistics']['first_grok_step']['mean']:
        print(f"First Grok Step: {summary['statistics']['first_grok_step']['mean']:.0f} ± {summary['statistics']['first_grok_step']['std']:.0f}")
    else:
        print("First Grok Step: N/A (some seeds did not grok)")
    print(f"Oscillation Count: {summary['statistics']['oscillation_count']['mean']:.1f} ± {summary['statistics']['oscillation_count']['std']:.1f}")
    print(f"Final Test Acc: {summary['statistics']['final_test_acc']['mean']:.4f} ± {summary['statistics']['final_test_acc']['std']:.4f}")
    print(f"Results saved to: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
