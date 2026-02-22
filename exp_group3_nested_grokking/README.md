# 嵌套 Grokking 实验：表示层面的拓扑夺舍

## 背景

wechat67 实验发现：模乘法 (a×b) mod 97 的模型学到了 Z₁₂ 商群结构（邻接 100%），但每个陪集内部的 Z₈ 是死记硬背的。

**核心问题**：如果训练更久、正则化更强，Z₈ 内部结构能不能也自发涌现？

## 实验设计

- 模型：2 层 Transformer，hidden_dim=128，4 heads（和 wechat67 一致）
- 训练步数：1M / 5M
- Weight Decay 四档对比：1.0、1.5、2.0、5.0
- 每 10k 步保存激活
- 分析工具：`scan_strides.py`（多步长邻接扫描，检测 stride=1/2/4 的非标准环结构）

## 总结论

### 1. 拓扑夺舍是必然的

不管 WD 多少（只要在能 Grok 的区间内），外层结构最终都会被内层 stride=4 取代：

| wd | 外层初始拓扑 | 外层死亡时间 | 内层 stride=4 趋势 |
|----|-------------|------------|-------------------|
| 1.5 | stride=1（标准环，100%） | ~350k 步 | 震荡上升到 ~0.7-0.9 |
| 2.0 | stride=2（Z₆×Z₆，100%） | ~140k 步 | 震荡上升到 ~0.84 |

**WD 越强，外层死得越快，但最终结局相同。**

### 2. stride 由 gcd(12,8)=4 决定

模型选择的步长不是循环群的标准生成元，而是 gcd=4 的因子：
- 外层：wd=2.0 选 stride=2，wd=1.5 选 stride=1
- 内层：都走 stride=4（gcd 本身）

### 3. 超长训练不会锁定，反而走向混沌

5M 步实验（wd=2.0，截至 3.7M 步）：

```
1M 前:    inner_s4 经常冲到 0.8-0.9
1M-2M:    inner_s4 仍有高峰，但 0.0 的坑增多
2M-3M:    inner_s4 均值下降，inner_s1 开始偶尔冲高
3M-3.7M:  inner_s4 和 inner_s1 交替闪烁，无稳态
```

stride=4 和 stride=1 在争夺内层编码权，模型在各种拓扑之间永久震荡。
期间 step ~356 万时 test_acc 从 100% 崩溃到 3.4%，之后恢复——WD 在超长训练下会摧毁已学到的结构。

### 4. 2 层 128 维是嵌套 Grokking 的容量下界之下

这个微型模型能发现单层拓扑（Z₁₂ 或 Z₈ 的子群），但不够同时稳定维持两层嵌套结构。
要实现真正的嵌套 Grokking（两层拓扑共存且稳定），可能需要更大的模型。

## Weight Decay 相图

```
wd=1.0  →  没 Grok（89.6%）
wd=1.5  →  Grok，外层 stride=1 先涌现(20k)，350k 崩溃，内层 stride=4 接管
wd=2.0  →  Grok，外层 stride=2 先涌现(20k)，140k 崩溃，内层 stride=4 接管
wd=5.0  →  过度压缩（77.7%）
```

## 详细数据

### wd=2.0，1M 步（关键时间点）

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
1000000    0.083     0.844     15     最终状态（仍在震荡）
```

### wd=1.5，1M 步（关键时间点）

```
step       outer_s1  inner_s4  解读
  20000    0.958     0.812     两层同时有信号（罕见）
  30000    1.000     0.406     外层 stride=1 锁定
 100000    0.083     0.688     外层已衰退
 270000    0.583     0.906     ← inner_s4 最高点之一
 350000    0.042     0.302     ← 外层彻底崩溃
 550000    0.000     0.948     ← inner_s4 另一个高峰
1000000    0.000     0.562     最终状态（震荡剧烈）
```

### wd=2.0，5M 步（已完成）

最终 train_acc=78.2%，test_acc=73.4%——**彻底退化**。

WD=2.0 在超长训练下是非单调的：先逼出结构（100k 步 Grok），再摧毁结构（3.56M 步崩溃后无法恢复）。
inner_s4 没有锁定，stride=4 和 stride=1 交替闪烁直到模型死亡。

**结论：weight decay 存在最优训练长度，过了就是毒药。**

## C.C. 的解读（Gemini 3.0 Pro）

1. **外层为什么会死**：WD 下维持大环的长程权重太贵。模型选择把外层结构"内联"进内层的 stride=4 编码。

2. **为什么是 stride=4 赢了**：4 是 gcd(12,8) 本身。每走 4 步，外层和内层的相位产生一次相干叠加——同时捕捉两层相互作用的最短路径，WD 觉得它最省钱。

3. **PCA 瞬态骤降到 4-6 维**：是流形在寻找新稳定点时的"窄门"，螺旋管结构露头的瞬间。

## 第三轮实验：容量是瓶颈吗？

### 实验 C：big_model（4 层 256 维 8 头，wd=2.0，1M 步）—— 运行中

如果小模型夺舍是因为容量不足，那加大模型应该能让两层拓扑共存。

| | 小模型 | 大模型 |
|---|---|---|
| layers | 2 | 4 |
| hidden_dim | 128 | 256 |
| heads | 4 | 8 |
| 参数量 | ~10 万 | ~80 万 |

**判定标准**：outer（stride=1 或 2）和 inner_s4 同时维持 >0.5 = 容量假说成立。

```bash
# 看进度
sudo docker exec magical_bhabha bash -c "tail -5 /tmp/nested_big.log"

# 跑完后分析
sudo docker exec magical_bhabha bash -c "cd /workspace/ai-theorys-study/arxiv/wechat67/exp_group3_nested_grokking/code && python scan_strides.py --tag big_model"
```

## 未解决的问题

1. **任务选择**：换一个 gcd=1 的分解（如 mod 91 = 7×13），两层结构是否天然正交、更容易被发现？
2. **WD 调度**：先用高 WD 逼出结构，再降低 WD 让它稳定，能否避免永久震荡？

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
sudo docker exec magical_bhabha bash -c "cd /workspace/ai-theorys-study/arxiv/wechat67/exp_group3_nested_grokking/code && python scan_strides.py --tag wd2.0_5M"
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
