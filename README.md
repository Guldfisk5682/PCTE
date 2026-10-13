# PCTE: Cross-target Teachers for CLIP MTDA

Single-source, multiple-target adaptation on Office-Home with frozen CLIP.
The B baseline gives each domain independent prompts; R4 progressively admits
detached target prompt snapshots as teachers for the other target domains.

See [the controlled comparison](reports/vlm_competitors/report.md) and
[the three-seed reliability results](reports/b_reliability/multiseed/report.md).
Source code, experiment protocols, and result tables are versioned here.
Datasets, pretrained weights, feature caches, and training checkpoints stay
outside Git. Original CRPL entry points and history are retained; upstream
provenance is documented in [implementation notes](docs/implementation_notes.md).

## Single-source MTDA: SPL-only A/B/C baselines

### Multi-seed consolidation and independent controls

Seeds 41 and 43 extend the original seed-42 matrix using exactly the original
training implementation. `reports/seed_42/` preserves the first report. The
independent Source Prompt Only control trains **only Source CE** and predicts
with source text prototypes; target images are loaded only after its training
is complete. It matches A/B/C initialization, source batches, optimizer and
1,000-step scheduler. The same evaluator also computes CLIP zero-shot.

```bash
python scripts/run_officehome_mtda.py --seeds 41 43 --devices cuda:0 cuda:1 \
  --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt
# Run after the A/B/C jobs release both GPUs.
python scripts/run_source_only_matrix.py --seeds 41 42 43 --devices cuda:0 cuda:1 \
  --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt
python scripts/summarize_mtda_baselines.py --seeds 41 42 43
```

Source-only checkpoints/logs are in `runs/source_prompt_only/`; combined tables
and reports are in `reports/multiseed/`. Per-task, per-source, per-target, and
overall means and sample deviations are computed across seeds. Paired method
differences are calculated within each seed before aggregation. Zero-shot is
deterministic for this fixed dataset and weight file. Source-only versus
zero-shot measures source supervision plus source-prompt learning; A/B/C versus
source-only compares the additional target adaptation and target predictor.
These controls do not by themselves fully separate SPL from prompt design.
Literature notes are in `docs/spl_reliability_literature.md`; the original memo
predates the controlled B experiments below. Its other proposed mechanisms
remain outside this phase.

The original experiments below are preserved. The new `train_mtda.py` follows
`.codex/plan.md`: A shares class prompts across one source and three targets;
B gives every domain independent class prompts and a shared detached source
teacher; C trains each source→target pair from scratch with shared class prompts
within that pair. CLIP stays frozen. Losses are source CE and target soft CE only.

Use a compatible Python/PyTorch environment and install `requirements-mtda.txt`
if dependencies are missing. The runtime verified on this machine is
`/home/tangxunchi/miniconda3/envs/coop-da/bin/python` (activate `coop-da` to use
the `python` commands below).

```bash
python -m pytest -q
python train_mtda.py --help

# One A joint run (B uses the same command with --variant B).
python train_mtda.py --variant A --source Art --seed 42 \
  --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --device cuda:0

# One independent C pair; use a separate output root for short smoke tests.
python train_mtda.py --variant C --source Art --target Clipart --seed 42 \
  --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt \
  --device cuda:0 --steps 3 --output-dir runs/smoke

# Full seed-42 matrix: 4 A + 4 B + 12 C runs, one process on each GPU.
python scripts/run_officehome_mtda.py --seeds 42 --devices cuda:0 cuda:1 \
  --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt
python scripts/summarize_mtda.py --runs-dir runs/mtda_spl
```

Defaults: 1,000 updates, batch size 30 **per domain**, M1/M2=16, AdamW LR 0.003,
lambda_T=0.5, cosine once per update, float32. A/B use all other three domains
unless `--smoke` permits a pair check. Targets receive equal loss weights and
one batch each per update. C requires exactly one `--target` per invocation.
Domain aliases such as `Real World`, `real_world`, and `Real-World` are accepted.
Class vocabulary comes from the labeled source and is checked across domains.

Frozen image outputs are cached under `runs/feature_cache/` using the original
deterministic CLIP preprocessing; prompt text features are recomputed every
step. The target training interface contains no class labels. Evaluation is
transductive, counts every image including partial batches, and uses only the
current corresponding target prompt from the fixed final checkpoint. The
single-source centroid bank is maintained and saved, but its SPL weights are
identically one.

Artifacts per run: `config.json`, `initialization.pt`, `environment.json`,
`history.csv`, `train.log`, `last.pt`, and final `metrics.json`. Reports include
12 task rows, per-source and overall macro means, seed standard deviations,
zero-shot references, training time, and peak GPU allocation. Missing results
are marked pending and are excluded from complete macro means.

Completed runs are never overwritten. Resume an incomplete single run by
adding `--resume` to the identical original configuration. `--stop-after N`
saves an intermediate checkpoint for an intentional interruption test.
The matrix skips completed runs; `--retry-failed` resumes incomplete runs.
If failure occurred before any checkpoint, it preserves the attempt under
`failed_attempts/` and restarts that configuration from its initial seed.
Keep smoke outputs separate from full runs. Do not combine different training
protocols or implementation versions in one summary.

See `docs/implementation_notes.md` for provenance, compatibility fixes,
initialization, checkpoint state, and differences from the multi-source code.

### source-combined Setting
```
CUDA_VISIBLE_DEVICES=1 python main.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/ --dataset OfficeHome  --output_dir 'outputs/OH/' --evaluation_step 200 --prompt_learning_rate 0.005 --t_weight 0.5 --enhanced_pseudo_label 1 --OT_clustering 1 --training_mode source-combined
```


### Multi-source Setting
```
CUDA_VISIBLE_DEVICES=0 python main.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/ --dataset OfficeHome  --output_dir 'outputs/OH/' --evaluation_step 200 --prompt_learning_rate 0.005 --t_weight 0.5 --enhanced_pseudo_label 1 --OT_clustering 1 --training_mode multi-source
```


### ViT
```
CUDA_VISIBLE_DEVICES=0 python main.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/ --dataset DomainNet  --backbone ViT-L/14 --output_dir 'outputs/DN/' --evaluation_step 200 --prompt_learning_rate 0.005 --t_weight 0.5 --enhanced_pseudo_label 1 --OT_clustering 1 --training_mode multi-source
```

```
CUDA_VISIBLE_DEVICES=1 python main_PGA.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/ --dataset OfficeHome  --backbone ViT-L/14
```

### Corruption

gaussian_noise, shot_noise, impulse_noise, defocus_blur,
glass_blur,  zoom_blur, snow, frost, fog,
brightness, contrast, elastic_transform, pixelate, jpeg_compression,
speckle_noise, gaussian_blur, spatter, saturate

```
CUDA_VISIBLE_DEVICES=1 python main_corruptions.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/  --dataset OfficeHome --corruption spatter --severity 1 --output_dir 'outputs_corrupted/OH/' --evaluation_step 200 --prompt_learning_rate 0.005 --t_weight 0.5 --enhanced_pseudo_label 1 --OT_clustering 1
```

```
CUDA_VISIBLE_DEVICES=0 python main_PGA_corruptions.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/ --dataset OfficeHome --corruption defocus_blur --severity 1
```
# Controlled B SPL reliability experiments

`train_b_reliability.py` adds isolated R1–R6 experiments and the matched-budget
joint no-bank control R0M. `--variant R0` resolves to the unchanged B entry/result.
See [the protocol](docs/b_reliability_protocol.md) for budgets, frozen snapshots,
label-free ordering, bank masks, and offline-only target-label diagnostics.

```bash
python scripts/run_b_reliability.py --variants R0M R1 R2 R3 --seeds 42 \
  --devices cuda:0 cuda:1 --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt
python scripts/run_b_reliability.py --variants R4 R5 R6 --seeds 42 \
  --devices cuda:0 cuda:1 --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt
python scripts/summarize_b_reliability.py
python scripts/validate_b_reliability.py
python scripts/record_b_teacher_contributions.py
python scripts/plot_b_reliability.py
```

Artifacts live separately in `runs/b_reliability/` and `reports/b_reliability/`.
Original A/B/C results and their existing recursive summarizers remain compatible.
All methods train 1,000 source updates and 1,000 updates per target; sequential
global steps are 3,000. The initial bank threshold is 0.7, distance beta is 10,
and R6 refreshes admitted snapshots/banks every 200 joint global steps.

The completed first-round report and analysis are in `reports/b_reliability/`;
three-seed R0/R0M/R1/R2/R4 comparisons are in `reports/b_reliability/multiseed/`.
See `findings.md` for the four research questions and further controlled proposals.

## R4 agreement, temperature, and admission follow-up

`train_b_followup.py` preserves the previous runtimes and reuses R4 snapshots to
isolate distance temperature. Run the protocol in `docs/b_followup_protocol.md`:

```bash
python scripts/run_b_followup.py --seeds 42 --devices cuda:0 cuda:1 \
  --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt
python scripts/analyze_r4_agreement.py --final-source-only --device cpu \
  --trajectory-dir runs/b_followup_analysis/final_source
python scripts/analyze_bank_geometry.py
python scripts/record_b_teacher_contributions.py --runs-dir runs/b_followup
python scripts/validate_b_followup.py
python scripts/write_b_followup_report.py
python scripts/plot_b_followup.py
```

The preset variants scan beta 0/0.1/1/10 at fixed R4 teacher quality, reproduce R4
with a uniform reference control, and test all-at-start admission using initial
or trained reference teachers. The latter is an offline two-stage timing control;
its shared teacher preparation is additional compute, recorded separately from
the equal student budgets. Reports are in `reports/b_followup/` and new runs in
`runs/b_followup/`. Final Source checkpoint inference adds no training updates.
Optional Source CE trajectory reconstruction is offline diagnostic computation,
with no target training or feedback from target labels.

## Official VLM DA methods under the MTDA protocol

Sibling `AD-CLIP` and `DAPrompt` clones provide new `mtda_adapter/` modules and
`train_mtda.py` wrappers without altering their upstream trainers. The shared
harness trains one model on one source and all three targets at seed 42 with
frozen RN50 and 1,000 updates per domain. See
[the audited protocol](docs/vlm_competitors_protocol.md) for native objectives,
code/paper differences, registration repairs and compute accounting.

The adapter sources are included under `external/adapters/`. Install them into
official clones at the audited commits (by default beside this checkout):

```bash
python scripts/setup_vlm_competitors.py
```

On another machine, pass both `--adclip-repo ../AD-CLIP` and
`--daprompt-repo ../DAPrompt` to the matrix or single-run CLI; the preserved
experiment defaults point to the original machine. Invoke competitors through
this repository's `train_vlm_competitors.py`. Existing experiment code and
adapter files remain unchanged so their recorded implementation hashes match.

```bash
python scripts/run_vlm_competitors.py \
  --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt \
  --devices cuda:0 cuda:1
python scripts/evaluate_competitor_zeroshot.py
python scripts/summarize_vlm_competitors.py
python scripts/plot_vlm_competitors.py
```

Outputs are isolated in `runs/vlm_competitors/` and
`reports/vlm_competitors/`. The matrix runner estimates completion times and
waits on process events, with no frequent training-log polling. These are
controlled-budget MTDA adaptations, not reproductions of published long-budget
single-target results. Original A/B/C and R0–R6 runtimes/results are preserved.
