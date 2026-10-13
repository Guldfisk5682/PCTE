# B reliability execution log

Dates and ETAs below use UTC+8 (Asia/Hong_Kong).

## Engineering and short-run gate

On 2026-10-13, implemented an isolated B-only package and passed 45 synthetic tests. Ran Art/seed42 for 24 effective updates per domain across the unchanged R0 and new R0M/R1–R6, with refresh interval5. The final seven new artifacts passed the audit: matching budgets, finite losses, consistent bank/snapshot versions, self-teacher exclusion, full evaluation counts, and unchanged accuracy after an E2H target froze. R0M parameters and Source Bank matched R0 exactly.

An actual GPU R6 continuous24 versus pause11/resume24 test matched prompts, Source/Target banks, snapshots, optimizers, schedulers, streams and final predictions exactly. The sealed version additionally completed a CPU pause11/resume24 run, with checkpoint restoration and full evaluation. CPU training is an extra smoke check, not part of formal accuracy comparisons.

## Formal seed42 runs

First batch launched at approximately **01:03**, finished **01:42**: four sources each for R0M/R1/R2/R3 (16 new runs). All 16 audited runs preserve exact B Source parameters and cumulative banks; all four R0M results match the old B prompt tensors exactly. Source and each target receive 1,000 effective updates. Source/target exposure counts and final-checkpoint rules are recorded in every run.

First-batch macro results: R0/R0M **72.9987%**, R1 **73.2173%**, R2 **74.5042%**, R3 **74.1581%**. The E2H effect is +0.2186 points; Uniform adds +1.2868 versus R1, improving all four sources. Distance weighting loses 0.3461 versus Uniform. These are first-seed comparisons.

Second batch launched at approximately **01:43**: four sources each for R4/R5/R6. R4 completes at **74.6470%**, +1.6483 versus R0M with four positive sources. R5 completes at **74.2683%**, −0.3788 versus R4 with four negative sources. R6 runs at refresh interval200.

Second batch completed at approximately **02:07**. R6 gives **74.1124%**, −0.1558 points versus R5. Across 36 refreshes, accepted precision increases 15 times and decreases 21 times; the mean immediate change is −0.0065 percentage points. No refresh-interval sensitivity run was launched.

Target bank coverage/precision is high: across the 12 admission banks, R3 mean coverage **99.62%**, accepted pseudo-label precision **92.45%**, Oracle-centroid cosine **0.9759**; R5 gives **99.49%**, **92.75%**, **0.9759**. This quality alone does not establish that raw-distance weights choose the better teacher. In actual R3 Art→Clipart training, mean Source reference weight is **0.8200**; the first Real World stage receives Source only, despite later snapshots being available for post-hoc inference.

## Conditional multi-seed follow-up

The protocol predeclares a ≥0.5 macro-point gain versus each method's prescribed control and at least three positive sources. R2/R1 and R4/R0M satisfy this at seed42. Append seeds41/43 for those methods and controls after first-round R6 completes. No change to bank threshold, beta, difficulty order, or checkpoint selection is made from offline diagnostics. Refresh interval100/400 is conditional on the R6/R5 criterion.

The 32 conditional runs (R0M/R1/R2/R4 × four sources × seeds41/43) launched at approximately **02:07** and completed **03:16**. Final three-seed macro mean ± sample deviation: B/R0M **72.84 ± 0.15%**, R1 **73.26 ± 0.05%**, R2 **74.47 ± 0.03%**, R4 **74.62 ± 0.05%**. Paired deltas: R1−R0M **+0.42 ± 0.18**; R2−R1 **+1.21 ± 0.08**; R4−R0M **+1.79 ± 0.16** percentage points. Clipart target macros are **52.76 ± 0.28%**, **52.76 ± 0.18%**, **54.42 ± 0.24%**, and **54.43 ± 0.18%**, respectively.

All **60 new formal runs** pass the final audit, for 240,000 effective branch updates: 60,000 Source and 180,000 Target updates, 116,000 global steps, 1.8 million Source and 5.4 million Target training sample exposures. All 12 R0M prompt states match preserved B exactly, and every variant's final Source parameters/bank match its B reference. Reissuing both completed matrices skips all 60 runs. The suite now passes 46 tests, including fixed-Source-Teacher order invariance. Full validation is in `reports/b_reliability/validation.json`.

## Artifacts and monitoring

Runtime implementation SHA256 is `f33008a054906a428f079d001a7eb84df87a97c8bd6789d231662b39c6ff5bb2`; preserved baseline SHA256 remains `4ea3c0f993cdea0b38ff165236b94044e1308d2bec7c55b178fb9a13727edc28`. Actual commands and environment details are stored per run. Source-specific orders are fixed using the existing source-only seed42 model; the teacher checkpoint and feature-cache manifests are hashed.

Formal GPU progress is read around printed completion ETAs, typically every few minutes. Matrix dispatch uses subprocess output events; no frequent log-file polling is used. Teacher contribution reconstruction, offline quality summaries and standalone plots run separately after training. The plotting dependency is isolated in `/tmp/b_reliability_plot_deps`; the training environment stays unchanged.
