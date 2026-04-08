# NeurIPS 2026 Abstract

## Title

Grokking as Manifold Discovery: Dimension Collapse, Quotient Group Emergence, and Topological Possession in Modular Arithmetic

## Abstract

Grokking—the phenomenon where neural networks suddenly generalize long after memorizing training data—has been explained through weight norm dynamics, softmax collapse, and lazy-to-rich transitions. These theories share a common blind spot: they rely on external measurements (weight norms, loss curves) without directly characterizing the geometry of learned representations. We propose the **Manifold Discovery Hypothesis**: memorization corresponds to high-dimensional jagged curves passing through training points, generalization corresponds to discovering the low-dimensional manifold on which data is distributed, and Grokking is the phase transition between them. We provide extensive empirical evidence across three experiment groups on modular arithmetic. (1) **Modular addition** ($\mathbb{Z}_{97}$): PCA effective dimension drops from 78 to 8, persistent homology shows topological compactification (connected components: 500→6), and a bottleneck experiment reveals a critical dimension threshold at 8–16. (2) **Modular multiplication** ($\mathbb{Z}_{97}^*$): dimension drops from 89 to 11, and the model discovers quotient group structure—12 cosets corresponding to $\mathbb{Z}_{12}$ with 99.4% cluster purity and perfect ring adjacency. This motivates a **two-stage model**: local manifold discovery followed by global topological gluing. (3) **Nested Grokking**: under increased weight decay, a capacity-limited model exhibits **topological possession**—the outer $\mathbb{Z}_{12}$ structure collapses and is replaced by an inner stride-4 encoding ($= \gcd(12,8)$), while test accuracy remains at 100% throughout. Scaling up model capacity without proportional regularization prevents Grokking entirely, demonstrating that capacity and regularization pressure must be matched. Across all experiments, we observe **critical state oscillations**: repeated jumps between generalization and memorization (12–20 cycles), with synchronized activation dynamics (L2 norm and variance collapse during each regression). These findings suggest that Grokking is not a unidirectional phase transition but a critical competition between representational attractors.

## Keywords

Grokking, manifold learning, phase transition, representation geometry, topological data analysis, modular arithmetic, quotient groups, weight decay

## Author

Jin Yanyan (64125051@kmitl.ac.th)
King Mongkut's Institute of Technology Ladkrabang
