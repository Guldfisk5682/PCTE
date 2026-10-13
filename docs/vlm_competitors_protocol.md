# VLM single-source, multiple-target comparison

## Scope and provenance

The user selected the controlled **1,000 updates per domain** experiment, not
the papers' longer epoch budgets. Each Office-Home source setting trains **one
model on its source and all three targets simultaneously**. Seed 42, frozen
RN50, 65 classes, batch 30 per domain, deterministic CLIP preprocessing, cached
FP32 visual features, and final-budget checkpoint selection match the SPL
comparison. Target labels appear only in independent evaluation.

Official clones are `/home/tangxunchi/workspace/AD-CLIP` at
`06a30ce32ab7a0dcc29a5ac72e66c4e868f70c89` and
`/home/tangxunchi/workspace/DAPrompt` at
`f24b0c16c06b585104ba97615f89cd5df4def872`. Their original trainers, CLIP,
configs and datasets remain unchanged. New `mtda_adapter/` and `train_mtda.py`
files reuse the official prompt classes. The shared runner lives in SPL's
`mtda/competitors/`; original SPL entry points and runtimes are preserved.

Primary sources: [AD-CLIP code](https://github.com/mainaksingha01/AD-CLIP),
[AD-CLIP paper](https://arxiv.org/abs/2308.05659),
[DAPrompt code](https://github.com/LeapLabTHU/DAPrompt), and
[DAPL paper](https://arxiv.org/abs/2202.06687).

## Minimal MTDA adaptations

**DAPrompt** keeps the RN50 configuration's shared, class-agnostic 16-token
context and extends its 16-token domain vectors from two to four domains.
Source CE uses the source head; each target uses its own head and hard
pseudo-labels from the detached, frozen zero-shot CLIP classifier. Average the
three native target losses, retaining U=1 and T=1. Preserve the code's accepted
sample denominator; an empty accepted set contributes zero rather than NaN.
`DAPrompt` uses the paper's Office-Home threshold 0.6; `DAPrompt-CodeTau` retains
the published RN50 configuration's 0.5. Both thresholds were fixed before
formal evaluation. Their generic context is shared across domains by design.

**AD-CLIP** keeps the official RN50 style, content and mapping projectors and
the class-agnostic, image-conditioned 12-token prompt. One shared projector
model processes the three source/target pairs, averaging native pair losses:
source CE + 0.1 mapping MSE + KL + 0.01 entropy. Preserve the unusual code
details: KL normalizes text features over classes; entropy first L2-normalizes
target probabilities across the batch; the content decoder repeats one token
four times; mapping MSE receives **source** statistics. These are not rewritten
as new algorithms or used in SPL.

The original RN50 projector first layers are stored in Python lists, so they
are absent from optimizer parameters and checkpoints. Register them to make
device placement and restoration reliable. `ADCLIP-Code` freezes these layers
to reproduce the code's unupdated random projections; `ADCLIP-Registered`
trains them as an explicitly identified registration repair. Remaining losses
are identical. This repair alone does not make the code a paper reproduction.

The original RN50 evaluation invokes a one-image API although training requires
a source/target pair. The adapter supplies the paper-described target-only
inference: mapping(target statistics) substitutes source style, followed by
target style and content. No source images or other target batches enter test
prediction. The paper trains mapping with target statistics, whereas the
official code trains it with source statistics; both adapter variants retain
the latter, and results must carry this limitation.

## Budget, numerics and evaluation

All settings make 1,000 optimizer steps, 1,000 effective source updates and
1,000 updates for **each** target, exposing 30,000 samples per domain. AD-CLIP
has three conditional source CE views per step: 90,000 CE sample views from
30,000 source draws. Report this distinction instead of claiming identical
compute. Preserve native SGD/0.003 for DAPL and Adam/0.01 for AD-CLIP, momentum
and weight decay 0.0005. Use one source epoch of linear warmup from 1e-5 followed by cosine
decay over the fixed update budget. Native optimizer choices, parameter counts
and objectives differ from B/R4. Data augmentations and AMP are replaced by
the common deterministic FP32 protocol. No accuracy-based tuning or selection.

Visual features reuse the immutable SPL caches. AD-CLIP additionally caches
per-image layer1–4 mean, native standard deviation and GAP, without labels or
image tensors. Extraction batch 128 matches the original raw-feature cache;
different convolution batch shapes can produce small FP32 discrepancies.
Cache provenance includes weights, preprocessing, sample manifest and
statistics implementation. Statistics are checked against the same raw CLIP
features before caching.

AD-CLIP's 180 image-conditioned contexts per update are evaluated with exact
full-batch loss and bounded-memory gradient recomputation. KL and entropy are
never independently averaged over microbatches. Causal padding after the last
EOT is omitted; synthetic feature/gradient equivalence and real RN50 numerical
error are checked. This is a compute optimization, not an extra objective.

Evaluate all target samples, including partial batches, using only their own
classifier. Initial evaluation is diagnostic; the final checkpoint is the
reported checkpoint. Restore saved learned and fixed projections and verify
predictions. Save configuration, upstream and implementation SHA, losses,
budgets, runtime, memory and per-target predictions for each run.

## Reproduce

```bash
python scripts/run_vlm_competitors.py \
  --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt \
  --devices cuda:0 cuda:1
python -m pytest -q
python scripts/summarize_vlm_competitors.py
```

Each cloned repository also has a `train_mtda.py` wrapper accepting the same
single-run options. Resume an interrupted matrix with `--resume-failed`;
completed runs are validated and preserved. The runner waits on process output
and completion, estimates finish times after warmup, and never polls training
logs. Results live in `runs/vlm_competitors/` and `reports/vlm_competitors/`.

The causal evidence for cross-target teachers remains the controlled **R4 vs
B/R0M** comparison. Competitor ranking tests practical performance at this
budget; it cannot alone isolate the teacher mechanism, establish longer-budget
performance, or reproduce published single-target results.
