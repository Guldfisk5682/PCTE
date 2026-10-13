# Multi-seed baseline consolidation protocol

## Scope and preserved experiment

The user requested two additional seeds, **41 and 43**, retaining the complete A/B/C results, plus matched CLIP Zero-shot and Source Prompt Only controls. Together with the completed seed 42, the experiment includes **60 A/B/C training runs** and **12 independent source-only runs**. This phase implements no new pseudo-label geometry, bank weighting, target neighborhood graph, or exploratory method validation.

The A/B/C training core remains byte-for-byte unchanged, with implementation SHA256 `4ea3c0f993cdea0b38ff165236b94044e1308d2bec7c55b178fb9a13727edc28`. The original seed-42 report is archived under `reports/seed_42/`. Source-only code lives in a separate entry point, `train_source_only.py`, so adding controls does not change the baseline training hash or invalidate existing checkpoints.

## Reference definitions

- **CLIP Zero-shot:** frozen official RN50 with `A photo of a [CLASS]`; no learned prompt or source supervision. Every run uses the same class order, frozen image features, scale, and complete evaluation loader. It is deterministic for the fixed dataset/checkpoint, so seed repetition is not an independent estimate of dataset uncertainty.
- **Source Prompt Only:** train only the per-class context `[65,16,512]` and the source domain context `[16,512]` using Source CE. Freeze CLIP, its scale, and unused target domain contexts. Use the learned **source** text features directly for all three target predictions. Target images are loaded only after training, for evaluation; target labels are instantiated inside the existing `mtda/evaluate.py` pipeline.
- **A/B/C:** unchanged definitions and losses from `.codex/plan.md`; main results still use each target's own current prompt at the fixed final checkpoint.

Source-only uses the same class/domain initialization, source permutations, batch size 30, 1,000 updates, AdamW LR 0.003/weight decay 0.01, and per-update cosine as A/B/C. Its optimizer contains exactly two prompt tensors. The extra target contexts exist only to reuse the established prompt layout and are frozen; the source-only evaluation adapter never calls a target prompt. Source bank statistics are maintained for audit but do not enter the loss or classifier.

The B source teacher receives Source CE gradients exclusively, so it is mathematically the same source-supervised learner as this control. A deterministic five-update test with AdamW verifies exact equality of their source class/domain tensors while B also processes unrelated target batches. A/C source prompts are target-influenced through shared class context and are not used as the clean source-only control.

## Evaluation and aggregation

The shared evaluator counts every target sample, including tail batches. It loads fixed final prompt values and recomputes text prototypes; no target-label checkpoint selection is performed. The protocol remains transductive UDA for A/B/C, while source-only is ordinary source-supervised transfer to the same evaluation images.

Each directed source→target task retains all three seed values. Source and target group means first average their three directed tasks within each seed; the overall macro first averages all 12 tasks within each seed. Reported standard deviations are **sample deviations across these seed-level quantities**. Paired differences are computed per seed before estimating their mean and deviation, preserving correlation between methods. Incomplete seeds cannot silently enter a complete macro comparison.

The source-only summary rejects changes in optimizer/scheduler protocol, source class order, source feature fingerprint, CLIP checkpoint, data root, or baseline implementation. Its target feature fingerprints and zero-shot results must match the primary evaluation. Source-only artifacts use `runs/source_prompt_only/`, separate from `runs/mtda_spl/`.

## Attribution limits

Source-only minus zero-shot measures source supervision **and** learning the source prompt. A/B/C minus source-only compares the additional target-adaptation pipeline, including its inference prompt. B provides a particularly clear fixed-source-teacher comparison because its source branch is gradient-isolated. These controls do not, by themselves, fully factor SPL and prompt architecture into independent causal effects; no extra exploratory ablation is launched here.

Three seeds assess initialization/sampling variability for this protocol, not statistical significance or generalization beyond Office-Home. The bank's single-source weights remain one; improvements cannot be credited to bank-driven pseudo-label reweighting.

## Commands

Use `/home/tangxunchi/miniconda3/envs/coop-da/bin/python` or activate `coop-da`.

```bash
python scripts/run_officehome_mtda.py --seeds 41 43 --devices cuda:0 cuda:1 \
  --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt
python scripts/run_source_only_matrix.py --seeds 41 42 43 --devices cuda:0 cuda:1 \
  --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt
python scripts/summarize_mtda_baselines.py --seeds 41 42 43
```

The A/B/C matrix is dispatched first. Source-only GPU runs follow after it releases the devices, preserving comparable runtime measurements. The runner reports ETA from actual updates and completion events; progress checks follow predicted completion rather than frequent log polling.

Validation includes the original suite, source-only gradient/inference tests, paired-statistics and missing-control tests, plus a three-update real Office-Home CPU smoke test. Literature review by the requested `gpt-6-luna`/`xhigh` subagent is in `docs/spl_reliability_literature.md`; candidate methods were not implemented or evaluated.

## Completed results and artifact audit

Completed all 60 A/B/C runs and 12 Source Prompt Only runs on 2026-10-13 (UTC+8). Every run finished its fixed 1,000 updates, for 72,000 total updates. The original seed-42 runs were retained. All training histories contain finite values; source banks observe all 65 classes and count 30,000 source samples per run. Full evaluation counts are Art 2,427, Clipart 4,365, Product 4,439 and Real World 4,357.

The audit confirmed identical initialization across A/B/C, identical cumulative source banks for matched source/seed runs, and the unchanged training-core hash. For every one of the 12 source-only controls, its final class context, source-domain context and bank tensors equal the corresponding B source teacher **exactly**. Source-only optimizers contain two tensors and consume zero target training images. The correctness suite passes 24 tests.

The overall macro results (mean ± sample standard deviation over seeds 41/42/43) are Zero-shot **72.14%**, Source Prompt Only **66.20 ± 0.26%**, A **72.79 ± 0.05%**, B **72.84 ± 0.15%**, and C **72.60 ± 0.10%**. Source-supervised prompt transfer alone loses 5.94 percentage points versus Zero-shot. A/B/C recover 6.40–6.64 points versus source-only, but their net gains versus Zero-shot are only 0.46–0.70 points. The A–B paired difference is −0.05 ± 0.18 points, so these seeds do not support a clear superiority claim.

Clipart as the **target** remains difficult: Zero-shot 51.71%, Source Only 47.45 ± 0.22%, A 52.65 ± 0.14%, B 52.76 ± 0.28%, and C 52.48 ± 0.18%. These results motivate the separately documented SPL-reliability research; no proposed method has been tested. Complete task/seed tables and reproducible summaries are in `reports/multiseed/`, with audit evidence in `validation.json`.
