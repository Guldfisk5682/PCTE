# Execution log

All times refer to 2026-10-12, UTC+8. This record separates validation from the formal seed-42 experiment. The formal report is generated from real `metrics.json` files rather than projected scores.

## Setup and compatibility

The workspace initially contained only `.codex/plan.md`. Read-only inspection confirmed four Office-Home domains, the local RN50 checkpoint, and two idle RTX 4090 GPUs. `AGENTS.md` was initialized as a 356-word contributor guide.

Both requested direct upstream clone attempts failed:

```bash
git clone https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning.git /tmp/spl-crpl-upstream
git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning.git /tmp/spl-crpl-upstream
```

The first reported `GnuTLS recv error (-110)`; the second reported `gnutls_handshake() failed`. The existing official bundle provided a clean fallback:

```bash
git clone /tmp/crpl_upstream_20261008.bundle /tmp/spl-crpl-upstream
cp -a /tmp/spl-crpl-upstream/. /home/tangxunchi/workspace/spl/
git remote set-url origin https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning.git
git switch -c feat/mtda-spl-baseline
git rev-parse HEAD
```

The resulting SHA is `2c032a62fc706b643bfc30f49c9a064dd24aa7ac`. The normal command sandbox cannot launch because unprivileged user namespaces are disabled; approved commands used the execution fallback without changing host settings.

Initial real feature extraction failed on the upstream `pkg_resources` import. Replacing it with `packaging.version` resolved compatibility. An initial synthetic softmax-order counterexample was symmetric and could not distinguish the two formulas; an asymmetric fixed tensor corrected the test. Later smoke runs exposed callable memory addresses in preprocessing cache keys; these are now stripped. Earlier caches and smoke artifacts were preserved and excluded from formal results.

## Validation actually executed

Runtime used throughout:

```bash
SPL_PYTHON=/home/tangxunchi/miniconda3/envs/coop-da/bin/python
"$SPL_PYTHON" train_mtda.py --help
"$SPL_PYTHON" scripts/run_officehome_mtda.py --help
"$SPL_PYTHON" -m pytest -q
"$SPL_PYTHON" scripts/run_officehome_mtda.py --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --dry-run
"$SPL_PYTHON" train_mtda.py --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt \
  --prepare-cache-only --device cuda:0 --workers 4
```

All 18 synthetic tests passed in the final core validation. The dry run enumerated exactly 20 training runs. Label-free frozen feature extraction covered Art=2,427, Clipart=4,365, Product=4,439, Real World=4,357: 15,588 images total.

A and B joint smoke commands (separate processes, GPUs 0 and 1):

```bash
"$SPL_PYTHON" train_mtda.py --variant A --source Art --seed 42 \
  --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt \
  --device cuda:0 --steps 3 --output-dir runs/smoke
"$SPL_PYTHON" train_mtda.py --variant B --source Art --seed 42 \
  --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt \
  --device cuda:1 --steps 3 --output-dir runs/smoke
```

Both completed three updates with finite losses and source bank counts=90, observed classes=46/65. Each evaluated all three targets with actual sample counts, including tails. Their short-run scores are validation artifacts, not formal performance estimates.

C exact-resume validation:

```bash
"$SPL_PYTHON" train_mtda.py --variant C --source Art --target Clipart --seed 42 \
  --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt \
  --device cuda:0 --steps 5 --stop-after 2 --output-dir runs/smoke_resume
"$SPL_PYTHON" train_mtda.py --variant C --source Art --target Clipart --seed 42 \
  --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt \
  --device cuda:1 --steps 5 --output-dir runs/smoke_reference
"$SPL_PYTHON" train_mtda.py --variant C --source Art --target Clipart --seed 42 \
  --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt \
  --device cuda:0 --steps 5 --resume --output-dir runs/smoke_resume
```

Direct tensor comparison confirmed exact equality of all prompt tensors, bank means/counts, optimizer states, scheduler, batch permutations/cursors/generators, and final target metrics between resumed and uninterrupted runs. Both counted all 4,365 Clipart samples. The short-run 39.95% accuracy is only a resume check.

## Formal experiment

During training, two additional report checks raised the final suite to **20 passing tests**: multi-seed sample deviations and empty-matrix reporting. Single-seed deviation fields are now blank rather than numerical zero. No training implementation or hyperparameter was changed during formal runs.

The complete matrix was launched after the above validation:

```bash
"$SPL_PYTHON" scripts/run_officehome_mtda.py --seeds 42 --devices cuda:0 cuda:1 \
  --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt
```

Defaults are 1,000 updates/run, batch size 30/domain, M1=M2=16, AdamW LR 0.003, lambda_T=0.5, and per-update cosine scheduling. Two processes train concurrently. The runner reads subprocess output as events, reports an ETA after the first 20 updates, and reports completion/failure. It does not poll run logs. `runs/mtda_spl/matrix_events.jsonl` preserves ETA and completion events; run log files preserve every executed child command.

Resume/retry and independent summary commands:

```bash
"$SPL_PYTHON" scripts/run_officehome_mtda.py --seeds 42 --devices cuda:0 cuda:1 \
  --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --retry-failed
"$SPL_PYTHON" scripts/summarize_mtda.py --runs-dir runs/mtda_spl
```

Only commands actually dispatched appear in per-run environments/logs. The generated report determines completed counts and pending work from artifacts. No additional seed has been launched.

## Completion and artifact audit

The formal matrix finished at approximately **22:10 UTC+8**, matching its ETA. All **20/20** runs and **36/36** variant/task results completed without a formal run failure. The audit verified all 20,000 history rows, step-1000 checkpoints, full target sample counts, and 12 complete directed-task CSV rows. Across A/B/C, corresponding initial contexts and cumulative source banks were exactly equal for every source. Optimizer groups contained exactly 5/8/3 prompt tensors for A/B/C.

Final macro accuracies: zero-shot **72.14%**, A **72.74%**, B **73.00%**, C **72.65%**. Relative improvements are +0.60/+0.86/+0.51 percentage points; B exceeds C by 0.35 points. These are single-seed observations, with no estimated variability across seeds. Full per-target results and resource usage are in the generated report.

The same matrix command was executed again with `--retry-failed` after completion. It skipped all 20 completed runs, launched no training process, and regenerated the same complete summary. The final suite passed 20 tests; `git diff --check` passed.
