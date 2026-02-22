# 嵌套 Grokking 实验：表示层面的拓扑夺舍

## 一句话

**小模型只能同时记住一层规律：学会粗的就忘掉细的。加大模型可能解决这个问题（验证中）。**

## 背景

### 之前的发现（wechat67）

模乘法 (a×b) mod 97 的乘法群 Z₉₇* 有 96 个元素，可以分成两层结构：
- **外层 Z₁₂**：96 个元素按"除以 8 的余数"分成 12 组（陪集）
- **内层 Z₈**：每组内部有 8 个元素

原始实验发现：模型完美学到了外层 Z₁₂ 的分组（邻接得分 100%），但每组内部的 Z₈ 是**死记硬背**的——没有发现内部规律。

测试集准确率 100%，输出全对，但"脑子里的知识"只有一半是结构化的。

### 本实验的问题

如果训练更久、正则化更强，模型能不能也发现 Z₈ 内部的规律？

## 术语解释

### 什么是 stride（步长）？

**stride = 模型认为谁是"最近的邻居"。**

把 0-11 排成一个时钟：

```
        0
    11      1
  10          2
   9          3
    8       4
      7   5
        6
```

- **stride=1**：每个数和它旁边的数最亲近。0 的邻居是 1 和 11。这是标准的时钟。
- **stride=2**：每个数和隔一个的数最亲近。0 的邻居是 2 和 10。时钟被拆成两个小圈：
  - 偶数圈：0→2→4→6→8→10→0
  - 奇数圈：1→3→5→7→9→11→1

stride≠1 说明模型学到了一种**压缩版**的结构——发现了规律，但用了一种更省钱的编码方式。

### 什么是邻接得分？

检查模型内部表示中，每个元素的最近邻是不是"应该的邻居"。
- **100%** = 完美的环结构
- **~17%（外层）/ ~25%（内层）** = 随机，没有结构

### 什么是 Weight Decay（WD）？

训练时给模型权重加的"税"——权重越大，罚得越多。逼迫模型用更少的资源编码信息。
- WD 太小：模型懒得找规律，直接硬背
- WD 适中：逼出规律（Grokking）
- WD 太大：资源太少，连硬背都做不到

## 实验设计

- 基础模型：2 层 Transformer，hidden_dim=128，4 heads
- 大模型：4 层 Transformer，hidden_dim=256，8 heads
- Weight Decay：1.0 / 1.5 / 2.0 / 5.0
- 训练步数：1M / 5M
- 分析工具：`scan_strides.py`（检测 stride=1/2/4 的邻接得分）

## 核心发现

### 1. 拓扑夺舍：外层和内层不能共存

不管 WD 多少（只要能 Grok），外层结构最终都会**坍塌**，被内层结构取代：

| wd | 外层初始拓扑 | 外层存活时间 | 内层 stride=4 趋势 |
|----|-------------|-------------|-------------------|
| 1.5 | stride=1（标准环，100%） | ~350k 步 | 震荡上升到 ~0.7-0.9 |
| 2.0 | stride=2（双环，100%） | ~140k 步 | 震荡上升到 ~0.84 |

**WD 越强，外层死得越快，但最终结局相同——只有内层 stride=4 活下来。**

全程测试集准确率 100%。模型在内部表示完全重组的过程中，输出始终正确——换了脑子但没换行为。

时间线（wd=2.0）：

```
阶段 I   (20k-130k)   外层 stride=2 ≈ 100%    内层 ≈ 随机     外层统治
阶段 II  (140k-400k)  外层崩溃到 0-30%        内层开始上升     夺舍发生
阶段 III (400k-1M)    外层 ≈ 随机              内层 ≈ 0.84     内层统治
```

### 2. WD 决定模型选哪种拓扑

| wd | 外层选择 | 解读 |
|----|---------|------|
| 1.5 | stride=1（标准 12 步大环） | WD 压力适中，模型负担得起"精确"的结构 |
| 2.0 | stride=2（两个 6 步小环） | WD 压力大，模型选择更省钱的"有损压缩" |

stride=2 把 12 步大环拆成了两个 6 步小环——权重之间的耦合度降低，维护成本更低。这是模型在 WD 压力下的自发对称性破缺。

### 3. 内层都走 stride=4——因为 gcd(12,8)=4

不管外层选 stride=1 还是 2，内层都选了 stride=4。

4 是 gcd(12,8) 本身——外层和内层的"最大公约数"。每走 4 步，外层和内层的相位刚好对齐一次。模型发现这是同时编码两层信息的最短路径。

### 4. 超长训练是毒药

5M 步实验（wd=2.0）：

```
1M 步:   test_acc = 100%    ← 甜蜜点
3.56M 步: test_acc = 3.4%    ← 崩溃
5M 步:   test_acc = 73.4%   ← 永久退化
```

**Weight Decay 是非单调的**：先逼出结构（Grokking），再摧毁结构（过度压缩）。存在一个最优训练长度，过了就是毒药。

类比：WD 像引力——适度的引力让星球成形，过度的引力压成黑洞。

### 5. 根本原因：容量不足

2 层 128 维的微型模型（~10 万参数）只够编码一层拓扑结构。当 WD 逼它同时编码两层时，它只能二选一——所以出现"夺舍"。

这就像一个 128 平米的房子只能放客厅或卧室二选一，但一个 12000 平米的庄园可以两个都有。

## Weight Decay 相图

```
wd=1.0  →  没 Grok（89.6%）—— 税太轻，不找规律
wd=1.5  →  Grok，外层 stride=1 先涌现(20k)，350k 崩溃，内层 stride=4 接管
wd=2.0  →  Grok，外层 stride=2 先涌现(20k)，140k 崩溃，内层 stride=4 接管
wd=5.0  →  过度压缩（77.7%）—— 税太重，连硬背都做不到
```

## 详细数据

### wd=2.0，1M 步

```
step       outer_s2  inner_s4  pca_g  解读
  20000    1.000     0.115     13     外层锁定
  50000    1.000     0.094     11     外层稳定
 100000    1.000     0.208     11     内层开始有信号
 140000    0.583     0.000      6     ← 转折点，PCA 骤降
 200000    0.083     0.542     13     内层起飞
 400000    0.000     0.604     11
 470000    0.167     0.833     14
 860000    0.000     0.896     12     ← 内层最高点
1000000    0.083     0.844     15     最终（仍在震荡）
```

### wd=1.5，1M 步

```
step       outer_s1  inner_s4  解读
  20000    0.958     0.812     两层同时有信号（罕见）
  30000    1.000     0.406     外层 stride=1 锁定
 100000    0.083     0.688     外层已衰退
 270000    0.583     0.906     ← inner_s4 最高点之一
 350000    0.042     0.302     ← 外层彻底崩溃
 550000    0.000     0.948     ← inner_s4 另一个高峰
1000000    0.000     0.562     最终（震荡剧烈）
```

### wd=2.0，5M 步

最终 train_acc=78.2%，test_acc=73.4%——**彻底退化**。

WD 在超长训练下是非单调的：先逼出结构（100k 步 Grok），再摧毁结构（3.56M 步崩溃后无法恢复）。

**结论：weight decay 存在最优训练长度，过了就是毒药。**

## C.C. 的解读（Gemini 3.0 Pro）

1. **外层为什么会死**：WD 下维持大环的长程权重太贵。模型选择把外层结构"内联"进内层的 stride=4 编码，用同一组权重解决两层逻辑。

2. **为什么 wd=1.5 选 stride=1 而 wd=2.0 选 stride=2**：WD 是"复杂性税收"。wd=2.0 的税太重，模型在起跑线上就选择了"有损压缩"——放弃全局一致性，躲进两个更省钱的 Z₆ 子环里。

3. **WD 的非单调性 = 引力 vs 黑洞**：WD 像引力，适度时让高维噪声凝聚成流形（Grokking），过度时把流形压向奇点（崩溃）。需要"退火"策略——先高温逼出结构，再降温让它稳定。

4. **PCA 瞬态骤降到 4-6 维**：是流形在寻找新稳定点时的"窄门"，螺旋管结构露头的瞬间。

## 第三轮实验：容量是瓶颈吗？（运行中）

如果小模型夺舍是因为容量不足，那加大模型应该能让两层拓扑共存。

| | 小模型 | 大模型 |
|---|---|---|
| layers | 2 | 4 |
| hidden_dim | 128 | 256 |
| heads | 4 | 8 |
| 参数量 | ~10 万 | ~80 万 |

**判定标准**：outer（stride=1 或 2）和 inner_s4 同时维持 >0.5 = 容量假说成立。

C.C. 的预测：
- 大模型可能发生"分层脱落"——Layer 1-2 锁 Z₁₂，Layer 3-4 锁 Z₈
- 但也可能"大模型惰性"——256 维太宽，直接硬背不需要泛化

```bash
# 看进度
sudo docker exec magical_bhabha bash -c "tail -5 /tmp/nested_big.log"

# 跑完后分析
sudo docker exec magical_bhabha bash -c "cd /workspace/ai-theorys-study/arxiv/wechat67/exp_group3_nested_grokking/code && python scan_strides.py --tag big_model"
```

## 未解决的问题

1. **任务选择**：换一个 gcd=1 的分解（如 mod 91 = 7×13），两层结构是否天然正交、更容易被发现？
2. **WD 调度**：先用高 WD 逼出结构，再降低 WD 让它稳定，能否避免永久震荡？
3. **大模型涌现的本质**：大模型的"涌现能力突变"，有没有可能就是容量跨过"夺舍→共存"的临界点？

## 跑法

```bash
# 基本用法
sudo docker exec magical_bhabha bash -c "cd /workspace/ai-theorys-study/arxiv/wechat67/exp_group3_nested_grokking/code && python train_nested_grokking.py --wd 2.0 --steps 1000000"

# 自定义输出目录
sudo docker exec magical_bhabha bash -c "cd /workspace/ai-theorys-study/arxiv/wechat67/exp_group3_nested_grokking/code && python train_nested_grokking.py --wd 2.0 --steps 5000000 --tag wd2.0_5M"

# 改模型大小
sudo docker exec magical_bhabha bash -c "cd /workspace/ai-theorys-study/arxiv/wechat67/exp_group3_nested_grokking/code && python train_nested_grokking.py --wd 2.0 --dim 256 --layers 4 --heads 8 --tag big_model"

# 多步长邻接扫描（核心分析工具）
sudo docker exec magical_bhabha bash -c "cd /workspace/ai-theorys-study/arxiv/wechat67/exp_group3_nested_grokking/code && python scan_strides.py --wd 1.5"
sudo docker exec magical_bhabha bash -c "cd /workspace/ai-theorys-study/arxiv/wechat67/exp_group3_nested_grokking/code && python scan_strides.py --tag big_model"
```

## 目录结构

```
exp_group3_nested_grokking/
├── README.md
├── code/
│   ├── train_nested_grokking.py      # 训练（--wd --steps --dim --layers --heads --tag）
│   ├── analyze_nested_structure.py   # 双层邻接分析（stride=1 only，已过时）
│   └── scan_strides.py              # 多步长邻接扫描（stride=1,2,4，推荐使用）
└── results/
    ├── wd_1.0/                       # 没完全 Grok（89.6%）
    ├── wd_1.5/                       # 外层 stride=1 先涌现后崩溃，内层 stride=4 接管
    ├── wd_2.0/                       # 外层 stride=2 先涌现后崩溃，内层 stride=4 接管
    ├── wd2.0_5M/                     # 5M 步：彻底退化（73.4%）
    ├── wd_5.0/                       # 过度压缩（77.7%）
    └── big_model/                    # 第三轮：4层256维，容量验证（运行中）
```
