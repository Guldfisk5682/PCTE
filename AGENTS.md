# Repository Guidelines

## Project Structure & Module Organization

This repository adapts official Clustering Reinforcement Prompt Learning (CRPL) to single-source, multiple-target domain adaptation (MTDA). The implementation contract is `.codex/plan.md`. Preserve upstream entry points (`main.py`, `main_PGA.py`, and corruption variants), `model.py`, and the bundled `clip_custom/` implementation.

Put the new CLI in `train_mtda.py`, reusable migration code in `mtda/`, experiment runners and summarizers in `scripts/`, and synthetic regression tests in `tests/`. Record upstream provenance and implementation decisions in `docs/`. Store generated checkpoints and logs under `runs/mtda_spl/` and result tables under `reports/`; keep datasets and pretrained weights outside Git.

## Build, Test, and Development Commands

Use a Python environment with PyTorch, torchvision, and the dependencies listed in `requirements-mtda.txt`; this project has no compilation step.

- `python train_mtda.py --help`: inspect single-run configuration.
- `python -m pytest -q`: run synthetic correctness tests without downloading data.
- `python scripts/run_officehome_mtda.py --help`: inspect the A/B/C experiment matrix runner.
- `python scripts/summarize_mtda.py --runs-dir runs/mtda_spl`: regenerate experiment tables and the report.

Pass dataset, checkpoint, and device paths explicitly. Start real-data validation with a short run before launching the full matrix.

## Coding Style & Naming Conventions

Use four-space indentation in new Python files, `snake_case` for functions and modules, and `PascalCase` for classes. Keep tensor shapes and domain identities explicit. Prefer small modules and clear assertions over additional frameworks. No upstream formatter or linter configuration is established; avoid unrelated formatting changes.

## Testing Guidelines

Use pytest with `test_*.py` files and deterministic synthetic tensors. Verify prompt sharing and gradient isolation, detached SPL teachers, cumulative centroid statistics, checkpoint restoration, and complete evaluation batches. No numerical coverage threshold is specified. Target training datasets must never expose class labels to the trainer. Freeze CLIP and optimize only prompt parameters; preserve the SPL-only loss contract.

## Commit & Pull Request Guidelines

Upstream history uses short summaries such as `clean` and `ViT and corruptions`, without a strict convention. Write descriptive imperative subjects, for example `Add detached single-source SPL teacher`. Pull requests should explain behavior, link relevant issues or plan sections, list validation commands, and distinguish completed experiments from pending runs. Include seeds, configuration, and upstream SHA for reported results; never select checkpoints using target accuracy.
