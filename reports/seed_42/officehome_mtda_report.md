# Office-Home CRPL → MTDA SPL-only Report

Completed training runs: **20/20**, seeds: [42]. Directed task results: 36/36.

Protocol: transductive UDA; target training sees images without class labels. Accuracy uses every target image and its own current target prompt from the fixed final checkpoint. No target-accuracy checkpoint selection is performed.

Upstream: https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning, commit `2c032a62fc706b643bfc30f49c9a064dd24aa7ac`. Official remote cloning encountered TLS failures; the unmodified local upstream Git bundle was cloned instead (see implementation notes).

Configuration: 1000 updates, batch size 30 per domain, M1/M2=16/16, float32 CLIP, AdamW LR 0.003, weight decay 0.01, lambda_T=0.5, cosine once per update.

Data: `/home/tangxunchi/office_home`. Weights: `/home/tangxunchi/.cache/clip/RN50.pt` (SHA256 `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`). Class order is defined by source directories.

Environment: Python 3.10.20, PyTorch 2.6.0+cu124, CUDA 12.4, GPU NVIDIA GeForce RTX 4090. Runtime details are saved per run.

## Results (%)

| Source | Target | Zero-shot | A | B | C | A−C | B−C |
|---|---|---:|---:|---:|---:|---:|---:|
| Art | Clipart | 51.71 | 52.51 | 52.67 | 51.52 | 0.99 | 1.15 |
| Art | Product | 82.36 | 80.60 | 81.17 | 80.85 | -0.25 | 0.32 |
| Art | Real World | 82.72 | 83.84 | 83.84 | 83.75 | 0.09 | 0.09 |
| Clipart | Art | 71.78 | 71.69 | 72.23 | 70.75 | 0.95 | 1.48 |
| Clipart | Product | 82.36 | 81.48 | 81.71 | 81.46 | 0.02 | 0.25 |
| Clipart | Real World | 82.72 | 82.79 | 82.76 | 82.53 | 0.25 | 0.23 |
| Product | Art | 71.78 | 71.12 | 71.28 | 70.33 | 0.78 | 0.95 |
| Product | Clipart | 51.71 | 51.91 | 51.96 | 53.06 | -1.15 | -1.10 |
| Product | Real World | 82.72 | 83.96 | 84.32 | 84.42 | -0.46 | -0.09 |
| Real World | Art | 71.78 | 73.55 | 74.29 | 74.04 | -0.49 | 0.25 |
| Real World | Clipart | 51.71 | 53.95 | 53.56 | 53.31 | 0.64 | 0.25 |
| Real World | Product | 82.36 | 85.51 | 86.19 | 85.81 | -0.29 | 0.38 |

## Source and overall macro means

| Source | Zero-shot | A | B | C |
|---|---:|---:|---:|---:|
| Art | 72.26 | 72.32 | 72.56 | 72.04 |
| Clipart | 78.95 | 78.65 | 78.90 | 78.25 |
| Product | 68.73 | 69.00 | 69.19 | 69.27 |
| Real World | 68.61 | 71.00 | 71.35 | 71.05 |
| All | 72.14 | 72.74 | 73.00 | 72.65 |

A source mean requires all three targets; the overall macro requires all 12 directed tasks for that seed. CSV files include sample standard deviations across seeds and completeness counts.

## Implementation and resources

A shares a single class context across all four domains. B has four independent class contexts; its source teacher is detached from each target loss. C resets prompts, optimizer, scheduler, bank, batch streams, and RNG for every pair, sharing class context within the pair. All domains have independent domain contexts with reproducible initialization.

SPL fuses base and the sole learned source similarities with weights 0.5/0.5 before applying the frozen CLIP scale and softmax. The multi-source combined branch is omitted. Only source CE and mean target soft CE are optimized; no OT, hard pseudo-label CE, entropy, filtering, or cross-target objective is used.

Frozen image outputs are cached using identical deterministic CLIP preprocessing; text features are recomputed from current trainable prompts. Cosine advances once per update, correcting upstream's inconsistent cadence. AdamW/0.003 follows official source defaults rather than the paper's SGD/0.005 description.

| Variant | Completed runs | Training seconds (sum) | Peak allocated GPU GiB |
|---|---:|---:|---:|
| A | 4 | 840.6 | 2.24 |
| B | 4 | 845.7 | 2.26 |
| C | 12 | 1318.2 | 2.24 |

Training timing excludes feature extraction and final evaluation; summed device time differs from matrix wall time when runs use both GPUs. Every target receives the same number of updates. C trains the source teacher separately three times per source.

## Source bank and interpretation

The bank is a cumulative mean of unnormalized source features, saved with integer counts. Single-source softmax weights equal 1 for every class and sample, so the bank is statistical state and does not change these SPL targets. Coverage and sample counts are recorded each step.

A: observed classes 65–65/65; source samples per run 30000–30000.
B: observed classes 65–65/65; source samples per run 30000–30000.
C: observed classes 65–65/65; source samples per run 30000–30000.

Observed A−C macro difference: 0.09 percentage points; B−C: 0.35; A−B: -0.26. Against the zero-shot macro (72.14%), A/B/C change by +0.60/+0.86/+0.51 percentage points. These measured differences do not isolate source-teacher training count from class-context sharing. Single-seed results provide no estimate of variability across seeds.

B has the highest observed macro in this experiment. B and C differ by 0.35 percentage points in macro, although individual tasks can favor different variants. The direct SPL migration is operational; the measured gains and task differences are listed above.

This one-seed result does not establish a robust advantage from sharing class context across targets. Additional seeds are needed to assess the stability of the small differences between A/B/C.

C consumes 1.57× A's and 1.56× B's summed training time. This includes its repeated source-prompt optimization across independent pairs.

## Execution and remaining work

Pending task/seed rows: 0. Inspect `matrix.log`, per-job `.log` files, and each run's `train.log` for failures. Re-run the matrix with the same arguments and `--retry-failed` to skip successful runs and restore checkpoints for interrupted runs.

Actually executed commands recorded in run environments:

```text
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant A --source Art --seed 42 --device cuda:0
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant A --source Clipart --seed 42 --device cuda:1
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant A --source Product --seed 42 --device cuda:1
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant A --source 'Real World' --seed 42 --device cuda:0
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant B --source Art --seed 42 --device cuda:1
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant B --source Clipart --seed 42 --device cuda:0
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant B --source Product --seed 42 --device cuda:1
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant B --source 'Real World' --seed 42 --device cuda:0
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant C --source Art --seed 42 --target Clipart --device cuda:1
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant C --source Art --seed 42 --target Product --device cuda:0
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant C --source Art --seed 42 --target 'Real World' --device cuda:1
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant C --source Clipart --seed 42 --target Art --device cuda:0
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant C --source Clipart --seed 42 --target Product --device cuda:1
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant C --source Clipart --seed 42 --target 'Real World' --device cuda:0
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant C --source Product --seed 42 --target Art --device cuda:1
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant C --source Product --seed 42 --target Clipart --device cuda:0
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant C --source Product --seed 42 --target 'Real World' --device cuda:1
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant C --source 'Real World' --seed 42 --target Art --device cuda:0
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant C --source 'Real World' --seed 42 --target Clipart --device cuda:1
/home/tangxunchi/miniconda3/envs/coop-da/bin/python train_mtda.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --steps 1000 --batch-size 30 --workers 4 --output-dir runs/mtda_spl --cache-dir runs/feature_cache --variant C --source 'Real World' --seed 42 --target Product --device cuda:0
```

## Future Work

After the fixed baselines finish, consider additional seeds and separately designed bank or cross-target ablations. No such algorithms are implemented in this experiment.
