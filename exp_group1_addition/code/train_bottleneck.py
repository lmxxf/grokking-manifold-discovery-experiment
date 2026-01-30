"""
Grokking 实验四：秩约束对 Grokking 的影响
在中间层添加低秩瓶颈，测试不同瓶颈维度对 Grokking 时间的影响

论文预测：
- bottleneck_dim ≈ 任务自由度（模加法 = 1 维循环群）→ 加速 Grokking
- bottleneck_dim < 任务自由度 → 阻止 Grokking
- bottleneck_dim > 任务自由度 → 不影响或轻微减慢

用法（在 Docker 容器内）：
    python /workspace/ai-theorys-study/arxiv/wechat67/code/train_bottleneck.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
from tqdm import tqdm
import argparse

# ============ 配置 ============
BASE_CONFIG = {
    "p": 97,                    # 质数，定义模运算 (a + b) mod p
    "train_ratio": 0.3,         # 训练集比例
    "embed_dim": 128,           # embedding 维度
    "num_heads": 4,             # attention heads
    "num_layers": 2,            # transformer 层数
    "lr": 1e-3,                 # 学习率
    "weight_decay": 1.0,        # 权重衰减
    "batch_size": 512,          # batch size
    "total_steps": 50000,       # 总训练步数（比完整实验短，只为检测 Grokking 时间）
    "eval_every": 500,          # 每隔多少步评估一次
    "seed": 42,
}


# ============ 数据生成 ============
def generate_modular_addition_data(p, train_ratio, seed=42):
    """生成模加法数据集：(a, b) -> (a + b) mod p"""
    np.random.seed(seed)

    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    np.random.shuffle(all_pairs)

    n_train = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:n_train]
    test_pairs = all_pairs[n_train:]

    def pairs_to_tensors(pairs):
        a = torch.tensor([x[0] for x in pairs], dtype=torch.long)
        b = torch.tensor([x[1] for x in pairs], dtype=torch.long)
        y = torch.tensor([(x[0] + x[1]) % p for x in pairs], dtype=torch.long)
        return a, b, y

    train_a, train_b, train_y = pairs_to_tensors(train_pairs)
    test_a, test_b, test_y = pairs_to_tensors(test_pairs)

    return (train_a, train_b, train_y), (test_a, test_b, test_y)


# ============ 带瓶颈的模型 ============
class BottleneckTransformer(nn.Module):
    """带低秩瓶颈的 Transformer"""

    def __init__(self, p, embed_dim, num_heads, num_layers, bottleneck_dim=None):
        super().__init__()
        self.p = p
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim

        # Embedding
        self.embed_a = nn.Embedding(p, embed_dim)
        self.embed_b = nn.Embedding(p, embed_dim)

        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(2, embed_dim) * 0.02)

        # Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 低秩瓶颈（核心修改）
        if bottleneck_dim is not None and bottleneck_dim < embed_dim:
            # 两层线性：embed_dim -> bottleneck_dim -> embed_dim
            self.bottleneck = nn.Sequential(
                nn.Linear(embed_dim, bottleneck_dim),
                nn.ReLU(),  # 非线性保证不会退化成单个矩阵
                nn.Linear(bottleneck_dim, embed_dim)
            )
        else:
            self.bottleneck = None

        # 输出层
        self.output = nn.Linear(embed_dim, p)

        # 用于保存中间激活
        self.last_hidden = None
        self.bottleneck_hidden = None

    def forward(self, a, b, save_hidden=False):
        # Embed
        emb_a = self.embed_a(a)
        emb_b = self.embed_b(b)

        # Stack 成序列
        x = torch.stack([emb_a, emb_b], dim=1)
        x = x + self.pos_embed.unsqueeze(0)

        # Transformer
        x = self.transformer(x)

        # 取第一个位置的输出
        h = x[:, 0, :]  # (batch, embed_dim)

        if save_hidden:
            self.last_hidden = h.detach().cpu()

        # 通过瓶颈（如果有）
        if self.bottleneck is not None:
            h = self.bottleneck(h)
            if save_hidden:
                self.bottleneck_hidden = h.detach().cpu()

        # 输出
        logits = self.output(h)

        return logits


# ============ 单次训练 ============
def train_single(bottleneck_dim, config, output_dir, device):
    """训练单个配置，返回 Grokking 时间（首次达到 90% 测试准确率的步数）"""

    # 生成数据
    (train_a, train_b, train_y), (test_a, test_b, test_y) = generate_modular_addition_data(
        config["p"], config["train_ratio"], config["seed"]
    )

    train_dataset = TensorDataset(train_a, train_b, train_y)
    test_dataset = TensorDataset(test_a, test_b, test_y)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # 模型
    model = BottleneckTransformer(
        p=config["p"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        bottleneck_dim=bottleneck_dim
    ).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    # 训练日志
    log = {
        "bottleneck_dim": bottleneck_dim,
        "steps": [],
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    # Grokking 时间（首次达到阈值的步数）
    grokking_time_90 = None  # 90% 测试准确率
    grokking_time_95 = None  # 95% 测试准确率
    grokking_time_99 = None  # 99% 测试准确率

    # 训练
    step = 0
    pbar = tqdm(total=config["total_steps"], desc=f"bottleneck={bottleneck_dim}")

    while step < config["total_steps"]:
        for batch_a, batch_b, batch_y in train_loader:
            if step >= config["total_steps"]:
                break

            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            batch_y = batch_y.to(device)

            # Forward
            logits = model(batch_a, batch_b)
            loss = F.cross_entropy(logits, batch_y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            pbar.update(1)

            # 评估
            if step % config["eval_every"] == 0:
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

                # 检测 Grokking
                if grokking_time_90 is None and test_acc >= 0.90:
                    grokking_time_90 = step
                if grokking_time_95 is None and test_acc >= 0.95:
                    grokking_time_95 = step
                if grokking_time_99 is None and test_acc >= 0.99:
                    grokking_time_99 = step

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "train": f"{train_acc:.3f}",
                    "test": f"{test_acc:.3f}"
                })

                model.train()

    pbar.close()

    # 保存结果
    result = {
        "bottleneck_dim": bottleneck_dim,
        "grokking_time_90": grokking_time_90,
        "grokking_time_95": grokking_time_95,
        "grokking_time_99": grokking_time_99,
        "final_train_acc": log["train_acc"][-1] if log["train_acc"] else None,
        "final_test_acc": log["test_acc"][-1] if log["test_acc"] else None,
        "log": log
    }

    # 保存到文件
    result_file = os.path.join(output_dir, f"bottleneck_{bottleneck_dim}.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ============ 主函数 ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bottleneck_dims", type=str, default="1,2,4,8,16,32,64,128",
                        help="逗号分隔的瓶颈维度列表")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/ai-theorys-study/arxiv/wechat67/results/exp4_bottleneck",
                        help="输出目录")
    args = parser.parse_args()

    # 解析瓶颈维度
    bottleneck_dims = [int(x) for x in args.bottleneck_dims.split(",")]
    # 加上 None（无瓶颈的 baseline）
    bottleneck_dims = [None] + bottleneck_dims

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    config = BASE_CONFIG.copy()
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 运行所有配置
    all_results = []
    for bottleneck_dim in bottleneck_dims:
        print(f"\n{'='*50}")
        print(f"Training with bottleneck_dim = {bottleneck_dim}")
        print(f"{'='*50}")

        result = train_single(bottleneck_dim, config, args.output_dir, device)
        all_results.append(result)

        print(f"Grokking time (90%): {result['grokking_time_90']}")
        print(f"Grokking time (95%): {result['grokking_time_95']}")
        print(f"Grokking time (99%): {result['grokking_time_99']}")
        print(f"Final test acc: {result['final_test_acc']:.4f}" if result['final_test_acc'] else "N/A")

    # 汇总结果
    summary = []
    for r in all_results:
        summary.append({
            "bottleneck_dim": r["bottleneck_dim"],
            "grokking_time_90": r["grokking_time_90"],
            "grokking_time_95": r["grokking_time_95"],
            "grokking_time_99": r["grokking_time_99"],
            "final_test_acc": r["final_test_acc"]
        })

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 打印汇总表格
    print("\n" + "="*70)
    print("SUMMARY: Bottleneck Dimension vs Grokking Time")
    print("="*70)
    print(f"{'Bottleneck':<12} {'Grok@90%':<12} {'Grok@95%':<12} {'Grok@99%':<12} {'Final Acc':<12}")
    print("-"*70)
    for r in summary:
        bn = r["bottleneck_dim"] if r["bottleneck_dim"] is not None else "None"
        g90 = str(r["grokking_time_90"]) if r["grokking_time_90"] else "N/A"
        g95 = str(r["grokking_time_95"]) if r["grokking_time_95"] else "N/A"
        g99 = str(r["grokking_time_99"]) if r["grokking_time_99"] else "N/A"
        acc = f"{r['final_test_acc']:.4f}" if r["final_test_acc"] else "N/A"
        print(f"{bn:<12} {g90:<12} {g95:<12} {g99:<12} {acc:<12}")
    print("="*70)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
