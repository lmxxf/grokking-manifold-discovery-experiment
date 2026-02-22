# Zenodo Metadata for Grokking Paper

## Resource type
Preprint

## Title
Grokking as Manifold Discovery: A Geometric Reinterpretation of Delayed Generalization

## Publication date
2026-02-22

## Creators
- Name: Jin Yanyan
- Affiliation: Independent Researcher

## Description
Grokking—the phenomenon where neural networks suddenly generalize after prolonged overfitting—has accumulated multiple theoretical explanations since its discovery in 2022: Goldilocks Zone, Softmax Collapse, Lazy-Rich transition, etc. This paper reviews these theories and identifies their common blind spot: most focus on external measurements, lacking direct characterization of representation space geometry. We propose a unified framework—the Manifold Discovery Hypothesis: memorization is a high-dimensional jagged curve passing through all training points, generalization is discovering the low-dimensional manifold on which data is distributed, and Grokking is the transition from the former to the latter. We provide evidence supporting this hypothesis on modular addition and modular multiplication experiments, observing significant drops in effective dimensionality (78→8 / 89→11), topological collapse, and emergence of quotient group structure (Z₁₂ cosets, purity 99.4%). Furthermore, nested Grokking experiments reveal capacity-topology competition: a small model (2-layer, 128-dim) under strong regularization undergoes "topological possession"—the outer Z₁₂ structure is replaced by inner stride=4 structure while test accuracy remains 100%. The model autonomously discovers gcd(12,8)=4 as the optimal encoding path. Weight decay exhibits non-monotonicity: it first forces structure (Grokking at 1M steps), then destroys it (collapse at 5M steps). Scaling to a larger model (4-layer, 256-dim) fails to Grok (test_acc=51.75%), demonstrating that capacity and regularization pressure must be matched—bigger is not better without proportionally increased constraints.

## Keywords
Grokking, Manifold Discovery, Deep Learning, Generalization, Weight Decay, Phase Transition, Representation Learning, Topological Phase Transition, Topological Possession, Nested Grokking, Capacity-Regularization Matching

## Languages
eng

## Version
v2.0

## License
Creative Commons Attribution 4.0 International (CC BY 4.0)

## Related works (Optional)
| Identifier | Relation | Resource type |
|------------|----------|---------------|
| arXiv:2201.02177 | IsReferencedBy | Preprint |
| arXiv:2205.10343 | IsReferencedBy | Preprint |
| arXiv:2301.05217 | IsReferencedBy | Preprint |

## Changelog
- v1.0 (2026-01-27): Initial release with modular addition/multiplication experiments
- v2.0 (2026-02-22): Added nested Grokking experiments (topological possession, WD phase diagram, gcd discovery, WD non-monotonicity, scaling experiment failure)
