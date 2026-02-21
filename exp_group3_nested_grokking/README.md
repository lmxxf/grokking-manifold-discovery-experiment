# 嵌套 Grokking 实验：表示层面的拓扑夺舍

## 背景

wechat67 实验发现：模乘法 (a×b) mod 97 的模型学到了 Z₁₂ 商群结构（邻接 100%），但每个陪集内部的 Z₈ 是死记硬背的。

**核心问题**：如果训练更久、正则化更强，Z₈ 内部结构能不能也自发涌现？

## 实验设计

- 模型：2 层 Transformer，hidden_dim=128，4 heads（和 wechat67 一致）
- 训练步数：150k → **1M**
- Weight Decay 三档对比：1.0、2.0、5.0
- 每 10k 步保存激活，分析外层 Z₁₂ 和内层 Z₈ 的邻接得分

## 第一轮结果（1M 步）

### Weight Decay 对比

| wd | 最终 train_acc | 最终 test_acc | 状态 |
|----|---------------|--------------|------|
| 1.0 | 95.7% | 89.6% | 没完全 Grok |
| **2.0** | **100%** | **100%** | **唯一的甜蜜点** |
| 5.0 | 83.2% | 77.7% | 过度压缩，植物人 |

### 关键发现：stride ≠ 1，gcd(12,8)=4 的扭转是真的

原始实验（wechat67, wd=1.0）中 Z₁₂ 的邻接是标准 stride=1 环。但 wd=2.0 下，模型选择了**非标准步长**：

- **外层 Z₁₂**：stride=2 邻接 = 100%（偶数和奇数各自成 Z₆ 环）
- **内层 Z₈**：stride=4 邻接从 0 涌现到 0.84

两个步长（2 和 4）都是 gcd(12,8)=4 的因子。模型没有选循环群的标准生成元，而是选了**公约数的因子**作为步长——在 128 维空间里，这些子群结构更容易被线性变换编码。

### 核心发现：不是嵌套涌现，是拓扑夺舍

时间线上有三个清晰的阶段：

```
阶段 I   (20k-130k)   外层 stride=2 邻接 ≈ 100%    内层 ≈ 随机
阶段 II  (140k-400k)  外层崩溃到 0-30%             内层 stride=4 开始上升
阶段 III (400k-1M)    外层 ≈ 随机                   内层 stride=4 趋势到 0.84
```

**不是"外层不动、内层涌现"——是外层坍塌后内层取而代之。**

全程测试集准确率 100%。模型在内部表示完全重组的过程中，输出始终正确——忒修斯之船。

### 详细数据（wd=2.0，关键时间点）

```
step       outer_s2  inner_s4  pca_g  解读
  20000    1.000     0.115     13     外层锁定
  50000    1.000     0.094     11     外层稳定
 100000    1.000     0.208     11     内层开始有信号
 140000    0.583     0.000      6     ← 转折点，PCA 骤降
 200000    0.083     0.542     13     内层起飞
 400000    0.000     0.604     11
 470000    0.167     0.833     14
 690000    0.000     0.792     13
 860000    0.000     0.896     12     ← 内层最高点
1000000    0.083     0.844     15     最终状态（仍在震荡）
```

### C.C. 的解读（Gemini 3.0 Pro）

1. **外层为什么会死**：WD=2.0 下，维持两个平行大环的长程权重太贵。模型选择把外层结构"内联"进内层的 stride=4 编码，用同一组权重解决两层逻辑。

2. **为什么是 stride=4 赢了**：4 是 gcd(12,8) 本身。每走 4 步，外层和内层的相位产生一次相干叠加——这是同时捕捉两层相互作用的最短路径，Weight Decay 觉得它最省钱。

3. **PCA 瞬态骤降到 4-6 维**：step 140k、280k、580k、680k、870k 各出现一次——是流形在寻找新稳定点时的"窄门"，螺旋管结构露头的瞬间。

## 未解决的问题

1. **最终状态还在震荡**（inner_s4 在 0.0-0.9 之间跳），1M 步没有完全锁定。5M 步会怎样？
2. **会不会有第三次夺舍？** 96 = 32 × 3，如果 stride=4 的 Z₈ 继续被压缩，下一层可能是 Z₃。
3. **容量实验**：如果加到 4 层 256 维，两层结构能不能同时稳定？还是竞争是必然的？

## 跑法

```bash
# 三档并行跑（后台）
sudo docker exec magical_bhabha bash -c "cd /workspace/ai-theorys-study/arxiv/wechat67/exp_group3_nested_grokking/code && nohup python train_nested_grokking.py --wd 1.0 > /tmp/nested_wd1.0.log 2>&1 &"
sudo docker exec magical_bhabha bash -c "cd /workspace/ai-theorys-study/arxiv/wechat67/exp_group3_nested_grokking/code && nohup python train_nested_grokking.py --wd 2.0 > /tmp/nested_wd2.0.log 2>&1 &"
sudo docker exec magical_bhabha bash -c "cd /workspace/ai-theorys-study/arxiv/wechat67/exp_group3_nested_grokking/code && nohup python train_nested_grokking.py --wd 5.0 > /tmp/nested_wd5.0.log 2>&1 &"
```

```bash
# 看进度（三档一次看完）
sudo docker exec magical_bhabha bash -c "echo '=== wd=1.0 ===' && tail -5 /tmp/nested_wd1.0.log && echo && echo '=== wd=2.0 ===' && tail -5 /tmp/nested_wd2.0.log && echo && echo '=== wd=5.0 ===' && tail -5 /tmp/nested_wd5.0.log"
```

```bash
# 跑完后分析
sudo docker exec magical_bhabha bash -c "cd /workspace/ai-theorys-study/arxiv/wechat67/exp_group3_nested_grokking/code && python analyze_nested_structure.py --wd 1.0"
sudo docker exec magical_bhabha bash -c "cd /workspace/ai-theorys-study/arxiv/wechat67/exp_group3_nested_grokking/code && python analyze_nested_structure.py --wd 2.0"
sudo docker exec magical_bhabha bash -c "cd /workspace/ai-theorys-study/arxiv/wechat67/exp_group3_nested_grokking/code && python analyze_nested_structure.py --wd 5.0"
```

## 目录结构

```
exp_group3_nested_grokking/
├── README.md
├── code/
│   ├── train_nested_grokking.py      # 训练（1M步，--wd 控制正则化强度）
│   └── analyze_nested_structure.py   # 双层邻接分析 + PCA 维度追踪
└── results/
    ├── wd_1.0/                       # 没完全 Grok（89.6%）
    ├── wd_2.0/                       # 唯一甜蜜点（100%），观察到拓扑夺舍
    └── wd_5.0/                       # 过度压缩（77.7%）
```
