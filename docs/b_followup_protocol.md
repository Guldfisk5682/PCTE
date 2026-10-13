# R4 follow-up protocol

The previous R2/R4 runs already include seeds 41, 42, 43. Reuse these completed
fixed-budget results; do not restart them or choose checkpoints by target accuracy.
R4 minus B is +1.7857 ± 0.1571 percentage points across paired seeds.

## Fixed teacher temperature scan

First run seed 42 for all four Office-Home sources. `FUniform` must reproduce R4
exactly. `FBeta0`, `FBeta0p1`, `FBeta1`, `FBeta10` use the same R4 snapshots created
at steps 250/500/750 and the same admission times. Build banks from those exact
snapshots, fixed confidence ≥0.7, full deterministic cached features. Only beta
varies; all snapshots and banks remain fixed after admission. Beta zero retains
invalid-class masking, so it need not match uniform in missing classes.

Compare accuracy, pseudo-label corrections/damage, conditional weight entropy,
near-one-hot rate, and Source preference. The existing R5 beta=10 is a separate
online experiment whose later snapshots can differ. A fixed-reference beta=10
control distinguishes this feedback from weighting effects.

## Admission controls

`FStartInit` opens all initial, untrained Target Prompt snapshots at step zero,
then holds them fixed. This is an online one-pass comparison with R4, changing
both the timing and maturity of teachers.

`FStartRef` opens the exact same trained R4 snapshots as `FUniform` at step zero.
It isolates admission timing at fixed teacher quality. This is an **offline
two-stage control with future teacher snapshots**, not a causal online method.
The common R4 preparation costs an additional 1,000 Source and 1,000 updates per
Target, shared by every reference experiment. Report preparation separately from
the 1,000-update student budget; never claim equal end-to-end compute with R4.

Every new student run retains B's initialization, streams, branch cosine schedule,
batch size 30, Source CE, detached Target Soft CE scaled by 0.5/3, frozen CLIP,
and own-target exclusion. No added losses, normalization, or threshold selection.

## Analysis and follow-up gate

Target labels enter only the offline error analysis and fixed-budget evaluation.
Partition original-SPL errors corrected by the ensemble, original correct samples
damaged, shared same-class errors, and different-class errors. Define vote
unanimity explicitly; agreement alone is not a precision estimate. Analyze
admission pre/post points, each teacher's leave-one-out change, and Clipart.

If a new variant exceeds R4 by ≥0.5 macro percentage points with ≥3 positive
Source settings, add seeds 41/43 to that variant and its required controls.
Otherwise retain the seed-42 scan as exploratory evidence and state its limits.
All old runtimes/results remain sealed. Monitor subprocess completion events at
estimated completion times, without frequent status polling.
