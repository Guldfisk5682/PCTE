# Office-Home multi-seed baseline consolidation

Requested seeds: **[41, 42, 43]**. A/B/C completed runs: **60/60**. Source Prompt Only completed runs: **12/12**.

All methods use the same frozen RN50, class order, image feature caches and full-batch-tail evaluation. Transductive A/B/C train on unlabeled target images; Source Prompt Only loads target images exclusively after source-supervised training, for evaluation. Main accuracy uses the fixed update-1000 checkpoint, with no target-accuracy selection.

Source Prompt Only independently learns the class context and source-domain context with Source CE only. It evaluates **the source text prototypes** on every target. Initialization, source permutations, batch size, AdamW/0.003, weight decay, and per-update cosine match A/B/C. Unused target domain contexts are frozen and never used as predictors. It is not the target-influenced source prompt from A/C.

Zero-shot uses `A photo of a [CLASS]` and no trained prompts. Its accuracy is deterministic for this fixed dataset/checkpoint; repeating it across seeds does not estimate dataset uncertainty.

## Source and overall macro accuracy (%)

| Source | CLIP Zero-shot | Source Prompt Only | A | B | C |
|---|---:|---:|---:|---:|---:|
| Art | 72.26 ± 0.00 | 66.65 ± 0.37 | 72.32 ± 0.06 | 72.49 ± 0.06 | 72.05 ± 0.07 |
| Clipart | 78.95 ± 0.00 | 68.63 ± 0.52 | 78.68 ± 0.22 | 78.72 ± 0.38 | 78.50 ± 0.30 |
| Product | 68.73 ± 0.00 | 62.07 ± 0.27 | 69.23 ± 0.21 | 68.97 ± 0.24 | 68.87 ± 0.35 |
| Real World | 68.61 ± 0.00 | 67.45 ± 0.31 | 70.91 ± 0.08 | 71.17 ± 0.16 | 70.96 ± 0.16 |
| All | 72.14 ± 0.00 | 66.20 ± 0.26 | 72.79 ± 0.05 | 72.84 ± 0.15 | 72.60 ± 0.10 |

## Accuracy grouped by target (%)

Each target mean averages its three different sources per seed, then averages seeds.

| Target | CLIP Zero-shot | Source Prompt Only | A | B | C |
|---|---:|---:|---:|---:|---:|
| Art | 71.78 ± 0.00 | 63.94 ± 0.41 | 72.27 ± 0.14 | 72.06 ± 0.51 | 71.87 ± 0.26 |
| Clipart | 51.71 ± 0.00 | 47.45 ± 0.22 | 52.65 ± 0.14 | 52.76 ± 0.28 | 52.48 ± 0.18 |
| Product | 82.36 ± 0.00 | 76.24 ± 0.56 | 82.67 ± 0.15 | 82.96 ± 0.15 | 82.68 ± 0.15 |
| Real World | 82.72 ± 0.00 | 77.17 ± 0.29 | 83.55 ± 0.11 | 83.57 ± 0.22 | 83.34 ± 0.24 |
| All | 72.14 ± 0.00 | 66.20 ± 0.26 | 72.79 ± 0.05 | 72.84 ± 0.15 | 72.60 ± 0.10 |

## All 12 directed tasks (%)

| Source | Target | CLIP Zero-shot | Source Prompt Only | A | B | C |
|---|---|---:|---:|---:|---:|---:|
| Art | Clipart | 51.71 ± 0.00 | 47.47 ± 0.67 | 52.11 ± 0.36 | 52.29 ± 0.45 | 51.60 ± 0.13 |
| Art | Product | 82.36 ± 0.00 | 73.49 ± 1.00 | 80.67 ± 0.06 | 81.08 ± 0.32 | 80.66 ± 0.47 |
| Art | Real World | 82.72 ± 0.00 | 78.98 ± 0.28 | 84.18 ± 0.31 | 84.09 ± 0.21 | 83.88 ± 0.17 |
| Clipart | Art | 71.78 ± 0.00 | 60.94 ± 0.54 | 71.80 ± 0.10 | 71.80 ± 0.67 | 71.80 ± 1.05 |
| Clipart | Product | 82.36 ± 0.00 | 71.53 ± 0.34 | 81.69 ± 0.44 | 81.74 ± 0.03 | 81.41 ± 0.10 |
| Clipart | Real World | 82.72 ± 0.00 | 73.41 ± 0.71 | 82.55 ± 0.35 | 82.61 ± 0.48 | 82.28 ± 0.22 |
| Product | Art | 71.78 ± 0.00 | 61.79 ± 0.56 | 71.30 ± 0.16 | 70.55 ± 0.63 | 70.22 ± 0.19 |
| Product | Clipart | 51.71 ± 0.00 | 45.30 ± 0.67 | 52.49 ± 0.51 | 52.33 ± 0.71 | 52.52 ± 0.56 |
| Product | Real World | 82.72 ± 0.00 | 79.11 ± 0.23 | 83.92 ± 0.13 | 84.03 ± 0.31 | 83.87 ± 0.55 |
| Real World | Art | 71.78 ± 0.00 | 69.08 ± 0.40 | 73.73 ± 0.19 | 73.81 ± 0.42 | 73.59 ± 0.50 |
| Real World | Clipart | 51.71 ± 0.00 | 49.57 ± 0.16 | 53.36 ± 0.52 | 53.65 ± 0.12 | 53.33 ± 0.18 |
| Real World | Product | 82.36 ± 0.00 | 83.69 ± 0.39 | 85.66 ± 0.19 | 86.06 ± 0.14 | 85.97 ± 0.17 |

Means and sample standard deviations are computed **across seed-level macro scores**, not across images or by pooling tasks. Paired deltas are computed within each seed first. Missing seed/task results remain pending; aggregate differences are not final until the requested matrix and source-only controls are complete.

## Interpretation and attribution

- source_only_minus_zero_shot: -5.94 ± 0.26 percentage points.
- A_minus_zero_shot: 0.65 ± 0.05 percentage points.
- B_minus_zero_shot: 0.70 ± 0.15 percentage points.
- C_minus_zero_shot: 0.46 ± 0.10 percentage points.
- A_minus_source_only: 6.59 ± 0.27 percentage points.
- B_minus_source_only: 6.64 ± 0.14 percentage points.
- C_minus_source_only: 6.40 ± 0.17 percentage points.
- A_minus_C: 0.19 ± 0.10 percentage points.
- B_minus_C: 0.24 ± 0.09 percentage points.
- A_minus_B: -0.05 ± 0.18 percentage points.

Source Only − Zero-shot measures the effect of source supervision together with learning the source prompt. A/B/C − Source Only measures the additional target-adaptation pipeline, which changes both the target-training signal and inference prompt. It does not isolate a universal pure-SPL effect independently of prompt design. B is the cleanest source-teacher control: its source contexts receive Source CE only, while its target contexts receive only SPL. A/B/C comparisons test their prescribed sharing choices.

Three seeds quantify initialization/sampling variability under this protocol; they do not establish statistical significance, generalization beyond Office-Home, or a causal benefit from the single-source bank (its SPL weights are identically one).

## Provenance and resources

| Method | Completed runs | Sum of training seconds | Peak allocated GiB |
|---|---:|---:|---:|
| A | 12 | 2503.4 | 2.24 |
| B | 12 | 2515.3 | 2.26 |
| C | 36 | 3909.8 | 2.24 |
| source_only | 12 | 729.2 | 2.24 |

Upstream commit: `2c032a62fc706b643bfc30f49c9a064dd24aa7ac`. Unchanged A/B/C training implementation SHA256: `4ea3c0f993cdea0b38ff165236b94044e1308d2bec7c55b178fb9a13727edc28`. Each run preserves configuration, seed, checkpoint, history, and its actual command in `environment.json`.

Data root: `/home/tangxunchi/office_home`; CLIP checkpoint SHA256: `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`.

The original seed-42 report is preserved under `reports/seed_42/`. Literature research is stored in `docs/spl_reliability_literature.md`; no geometry-based SPL method, target graph, new bank weighting, or exploratory experiment was implemented.
