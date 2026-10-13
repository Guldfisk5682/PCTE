# Controlled B SPL reliability experiments

## Scope and preservation

Only B's independent source/target class and domain prompts are developed. CLIP stays frozen; source CE and detached target soft CE are the only losses. Final inference uses each target's own prompt. A/B/C, their entry points and all prior checkpoints remain intact. New code is isolated in `mtda/reliability/` and `train_b_reliability.py`. New results use `runs/b_reliability/`, a separate tree to preserve compatibility with existing recursive A/B/C summarizers.

No OT, contrastive objective, additional distillation loss, replay, feature alignment, entropy minimization, confidence loss weighting, agreement filtering or class-specific threshold is introduced. Entropy and target-label precision are diagnostics, not optimization criteria.

## Fixed budget and ordering

First round: seed 42, RN50 FP32, four Office-Home sources, batch 30 per domain, M1=M2=16, AdamW LR 0.003/weight decay 0.01, lambda=0.5. Every source and every target receives exactly 1,000 effective optimizer updates and 30,000 sample exposures. Target gradients retain B's `lambda/3` multiplier. Each independent branch advances its cosine schedule only on its effective update.

Joint variants have 1,000 global steps; sequential variants have 3,000 global steps and update Source at steps 1,4,7,…,2998. Each sequential stage trains only its own target for 1,000 updates, then freezes it. Both organizations execute 4,000 branch-optimizer calls; R0 executes 1,000 combined calls. B's parameters are independent, so separate AdamW optimizers are mathematically equivalent. R0M explicitly verifies this against unchanged R0. Source teacher maturity still differs across sequential stages; R1 tests this ordering/scheduling organization, not pure transfer between independent targets.

Difficulty is fixed before any target training using the **existing source-only seed-42 final model**, with Base+Source similarities fused 0.5/0.5. Reusing this model for ordering adds no new formal source updates. Rank mean entropy ascending, with acceptance rate/max confidence descending for exact ties. Store all statistics, sample/cache identities and ordering-teacher hash; use the same order across variants and additional seeds. Manual order remains configurable. No target labels enter ordering.

## Variants and prescribed comparisons

| Variant | Organization | Other-target teacher | Bank behavior | Main control |
|---|---|---|---|---|
| R0 | Original B joint | None | Original Source statistics | Preserved results |
| R0M | Matched joint, separate branch optimizers | None | No Target bank | R0 equivalence |
| R1 | E2H sequential | None | None | R0M |
| R2 | E2H sequential | Completed frozen prompt snapshots, uniform | None | R1 |
| R3 | E2H sequential | Same completed snapshots, distance weighted | Built once at stage end | R2 |
| R4 | Joint progressive | Fixed snapshots admitted at 25/50/75%, uniform | None | R0M |
| R5 | Joint progressive | Same snapshots, distance weighted | Built at admission, fixed | R4 |
| R6 | Joint progressive | Admitted snapshots, distance weighted | Snapshot and bank replaced every 200 global steps | R5 |

R6 refreshes only admitted domains at global multiples of 200; a newly admitted domain is not redundantly rebuilt at the same step. A refresh at the final step contributes diagnostics but cannot influence subsequent training. R3 has no refresh because its teachers are frozen. Uniform and distance pairs have exactly the same candidate teachers.

## Bank and teacher contract

Cache raw FP32 visual features, L2-normalized features, stable opaque sample IDs and domain identity. Keys include RN50/checkpoint, deterministic preprocessing and sample/file manifests; no target labels or image tensors are cached. Confidence threshold is 0.7 inclusive. Rebuild each bank from unique full-cache samples, with zero-initialized FP32 sums; never accumulate old statistics. Invalid classes are masked. Snapshots and banks commit as one version after inference completes and survive checkpoint restoration.

Exclude the current target's snapshot and bank. Fuse normalized prototype similarities before logit-scale/softmax, with no normalization of their sum. Uniform references have equal weights. Distance uses `softmax_domain(-10 * squared_raw_L2)` independently per class. If Source has not observed a class, use Source-only for that class; if no other centroid is valid, Source gets weight one. With no admitted teacher, call the unchanged original SPL function exactly. Source running means/counts retain their original semantics.

## Validation, diagnostics and decision rule

Phase 1 tests the inclusive threshold, unique/replacement-only rebuild, invalid mask, self exclusion, domain-axis normalization, exact no-teacher fallback, snapshot immutability, changed-student refresh, gradient isolation, exact matched B updates, budget counters and resume.

Phase 2 runs Art/seed42 for 24 updates per domain across R0–R6 and R0M; use interval5 refresh to exercise admissions/rebuilds. Check finite logs, complete evaluation, exact checkpoint predictions and a pause/resume comparison before formal training. Preserve all smoke artifacts separately.

Phase 3 runs one full seed for four sources, first R0M/R1/R2/R3, then R4/R5/R6. Predeclare a practical follow-up trigger: a method gains at least **0.5 macro percentage points** over its prescribed control and improves at least three of four source macros at seed42. Such methods and their matched controls get seeds41/43; if R6 meets this trigger, add interval100/400 at seed42. This is a compute-allocation rule, not a significance claim; negative and mixed results remain reported.

Unified logs identify source, target, variant, seed, global step, stage and teacher version. Keep per-domain losses/update/exposure counts; evaluate at fixed intervals and stage/admission/refresh boundaries without selecting checkpoints. Save aggregate teacher weights, distance inputs and saturation, plus full prediction/accepted-set snapshots at events. After final training, a separate evaluation module alone loads target ground truth for precision/confusion/hard-class/oracle-centroid analysis. Those files never feed training or hyperparameter choices. `metrics.json` always comes from the fixed final checkpoint.

```bash
python train_b_reliability.py --variant R0 --source Art --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt
python scripts/run_b_reliability.py --variants R0M R1 R2 R3 --seeds 42 --devices cuda:0 cuda:1 --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt
python scripts/run_b_reliability.py --variants R4 R5 R6 --seeds 42 --devices cuda:0 cuda:1 --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt
python scripts/summarize_b_reliability.py
```

Formal training monitoring follows printed completion ETAs rather than frequent log polling. Commands, hashes, cache fingerprints, domain orders and effective budgets are saved with each run.

## Engineering validation completed

The correctness suite passes **46 tests**, including fixed-Source-Teacher joint/sequential/reverse-order invariance. All eight Art/seed42 short-run entries complete (original R0 plus seven new variants); full-cache evaluation counts match 4,365/4,439/4,357 target images. Both 24-update and 1,000-update R0M have exactly the same prompt tensors as preserved R0. The seven final smoke artifacts pass the budget/version/self-exclusion/frozen-accuracy audit. A real R6 pause11/resume24 comparison reproduces continuous prompts, optimizers/schedulers, banks, snapshots, streams and final predictions exactly.

The formal A/B/C hash remains `4ea3c0f993cdea0b38ff165236b94044e1308d2bec7c55b178fb9a13727edc28`; the B reliability implementation was sealed before formal training as `f33008a054906a428f079d001a7eb84df87a97c8bd6789d231662b39c6ff5bb2`.

At the end of E2H, an ensemble using **all** final snapshots is a post-hoc diagnostic: teachers admitted after a target froze did not teach that target. Actual training contribution tables therefore replay the historical feature batches, Source cumulative means and saved Target Bank/Teacher versions. Weight computation is independent of text logits, so this replay needs no Target ground truth and performs no optimization. Use `python scripts/record_b_teacher_contributions.py` after GPU training completes, then regenerate the summary. This distinguishes actual knowledge received from the final hypothetical ensemble.

Scientific plots export PNG/PDF using `scripts/plot_b_reliability.py`. Matplotlib is an optional plotting dependency. In this workspace it was installed in `/tmp/b_reliability_plot_deps` and invoked with a command-local `PYTHONPATH`, leaving the active training environment unchanged.
