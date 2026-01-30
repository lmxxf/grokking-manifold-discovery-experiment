"""
Grokking 实验：模乘法
验证流形发现假说在不同运算上的泛化性

任务：(a * b) mod p
注意：乘法群 Z_p^* 只有 p-1 个元素（排除 0），结构不同于加法群

用法（在 Docker 容器内）：
    python /workspace/ai-theorys-study/arxiv/wechat67/exp_group2_multiplication/code/train_grokking.py
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
CONFIG = {
    "p": 97,                    # 质数，定义模运算 (a * b) mod p
    "train_ratio": 0.3,         # 训练集比例
    "embed_dim": 128,           # embedding 维度
    "num_heads": 4,             # attention heads
    "num_layers": 2,            # transformer 层数
    "lr": 1e-3,                 # 学习率
    "weight_decay": 1.0,        # 权重衰减（关键！）
    "batch_size": 512,          # batch size
    "total_steps": 150000,      # 总训练步数
    "eval_every": 1000,         # 每隔多少步评估一次
    "save_activations_every": 1000,  # 每隔多少步保存激活
    "seed": 42,
    "output_dir": "/workspace/ai-theorys-study/arxiv/wechat67/exp_group2_multiplication/results",
    "operation": "multiplication",  # 标记运算类型
}

# ============ 数据生成 ============
def generate_modular_multiplication_data(p, train_ratio, seed=42):
    """
    生成模乘法数据集：(a, b) -> (a * b) mod p

    注意：乘法群 Z_p^* 不包含 0，所以 a, b ∈ {1, 2, ..., p-1}
    输出 y ∈ {0, 1, ..., p-1}（因为 a*b mod p 可以是 0 到 p-1）
    但实际上 a,b 都非零时，y 也非零（费马小定理）
    """
    np.random.seed(seed)

    # 乘法群：a, b ∈ {1, 2, ..., p-1}（排除 0）
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

        # Embedding: 两个输入各自 embed
        # 注意：虽然乘法群只用 1~p-1，但 embedding 还是 p 个（索引方便）
        self.embed_a = nn.Embedding(p, embed_dim)
        self.embed_b = nn.Embedding(p, embed_dim)

        # 位置编码（简单学习的）
        self.pos_embed = nn.Parameter(torch.randn(2, embed_dim) * 0.02)

        # Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,  # Grokking 实验通常不用 dropout
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.output = nn.Linear(embed_dim, p)

        # 用于保存中间激活
        self.last_hidden = None

    def forward(self, a, b, save_hidden=False):
        # Embed
        emb_a = self.embed_a(a)  # (batch, embed_dim)
        emb_b = self.embed_b(b)  # (batch, embed_dim)

        # Stack 成序列 (batch, seq_len=2, embed_dim)
        x = torch.stack([emb_a, emb_b], dim=1)

        # 加位置编码
        x = x + self.pos_embed.unsqueeze(0)

        # Transformer
        x = self.transformer(x)  # (batch, 2, embed_dim)

        # 保存中间层激活（用第一个位置的输出，即 [CLS] 风格）
        if save_hidden:
            self.last_hidden = x[:, 0, :].detach().cpu()

        # 用第一个位置的输出做分类
        logits = self.output(x[:, 0, :])  # (batch, p)

        return logits


# ============ 训练循环 ============
def train():
    # 设置随机种子
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "activations"), exist_ok=True)

    # 生成数据（模乘法）
    (train_a, train_b, train_y), (test_a, test_b, test_y) = generate_modular_multiplication_data(
        CONFIG["p"], CONFIG["train_ratio"], CONFIG["seed"]
    )

    train_dataset = TensorDataset(train_a, train_b, train_y)
    test_dataset = TensorDataset(test_a, test_b, test_y)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # 乘法群大小是 (p-1)^2
    print(f"Operation: modular multiplication (a * b) mod {CONFIG['p']}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # 模型
    model = GrokkingTransformer(
        p=CONFIG["p"],
        embed_dim=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"]
    ).to(device)

    # 优化器（AdamW with weight decay）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    # 训练日志
    log = {
        "steps": [],
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    # 训练
    step = 0
    pbar = tqdm(total=CONFIG["total_steps"], desc="Training")

    while step < CONFIG["total_steps"]:
        for batch_a, batch_b, batch_y in train_loader:
            if step >= CONFIG["total_steps"]:
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
            if step % CONFIG["eval_every"] == 0:
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

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "train_acc": f"{train_acc:.3f}",
                    "test_acc": f"{test_acc:.3f}"
                })

                model.train()

            # 保存激活
            if step % CONFIG["save_activations_every"] == 0:
                model.eval()

                # 收集所有测试样本的激活
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

                # 保存
                np.savez(
                    os.path.join(CONFIG["output_dir"], "activations", f"step_{step:06d}.npz"),
                    hidden=all_hidden,
                    labels=all_labels
                )

                model.train()

    pbar.close()

    # 保存训练日志
    with open(os.path.join(CONFIG["output_dir"], "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    # 保存配置
    with open(os.path.join(CONFIG["output_dir"], "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(CONFIG["output_dir"], "model_final.pt"))

    print(f"\nTraining complete!")
    print(f"Final train acc: {log['train_acc'][-1]:.4f}")
    print(f"Final test acc: {log['test_acc'][-1]:.4f}")
    print(f"Results saved to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    train()
