# CRPL → MTDA implementation notes

## Provenance and environment

- Official URL: https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning
- Upstream default branch: `main`.
- Upstream commit: `2c032a62fc706b643bfc30f49c9a064dd24aa7ac` (`ViT and corruptions`).
- Development branch: `feat/mtda-spl-baseline`.
- Two direct GitHub clone attempts failed with GnuTLS handshake/connection termination errors. The checkout was created by cloning `/tmp/crpl_upstream_20261008.bundle`, an existing unmodified official Git bundle. Its origin URL was restored to the official URL. This is a documented fallback, not a successful fresh remote clone.
- Available runtime: `/home/tangxunchi/miniconda3/envs/coop-da/bin/python`, Python 3.10.20, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, pytest 9.1.1. Two RTX 4090 GPUs (24 GiB each) were idle when inspected.
- User-supplied data: `/home/tangxunchi/office_home`, folders `art`, `clipart`, `product`, `real_world`.
- User-supplied weights: `/home/tangxunchi/.cache/clip/RN50.pt`.
- The host does not permit the sandbox's user namespace. Commands use the approval-reviewed execution fallback; no system namespace setting was changed.

The sole CLIP compatibility edit replaces removed `pkg_resources.packaging` with `packaging.version`. Original MSDA/PGA/corruption entry points, losses, and prompt modules remain intact.

## Official logic → migration logic

| Official behavior | MTDA implementation |
|---|---|
| Choose one target and treat the remaining domains as labeled sources | `train_mtda.py`: explicitly select one labeled source and three unlabeled targets |
| Shared class context, multiple sources, one target, uninitialized source-combined context | `mtda/prompts.py`: A shared, B independently cloned domain class contexts, C independently initialized pairs; no combined branch |
| Source-specific/combined logits plus base logits | `mtda/spl.py`: equal-weight base and sole learned source similarities, one frozen CLIP scale, then softmax and detach |
| Source × class cumulative means and distance weighting | `mtda/source_bank.py`: one source cumulative mean of pre-normalization features; single-source weights equal one |
| OT with companion hard CE; optional regularizers/filtering | `mtda/trainer.py`: only source CE + lambda_T × mean target soft CE |
| Target labels used for pseudo-label accuracy diagnostics | `UnlabeledImages` stores only paths and returns images; target streams supply only feature tensors |
| Target evaluation shuffled, tail dropped, total incremented by fixed batch size | `mtda/evaluate.py`: independent labeled evaluation, complete traversal, actual sample count |
| Historical/cached target text features may be stale or empty | Reload fixed final prompt checkpoint and recompute each target's text features |
| Training iterations overridden by dataset defaults | Honor CLI `--steps` |
| Cosine scheduler advanced intermittently with T_max equal to all updates | Advance after every optimizer update, T_max=steps for all variants |

## Fixed protocol and numerical choices

Both learned and base text prompts convert underscores in source-defined class names into spaces. Raw names and sorted indices remain in configurations; every branch uses that same semantic order.

CLIP stays frozen in float32/evaluation mode, including logit scale. AdamW uses source-default LR 0.003, default weight decay 0.01, batch size 30 per domain, M1=M2=16, lambda_T=0.5, and 1,000 updates. These follow official code defaults; the paper's SGD/0.005 setting is not substituted. The source loss and each weighted target loss are backpropagated sequentially to release graphs; one optimizer update follows the accumulated gradients. This matches the joint mathematical loss.

Class contexts initialize from an identical CPU-generated normal template for each `(seed, source)`, with std=0.02. B/C clone its values into independent parameters. Domain contexts use SHA256-derived `(seed, domain, role)` seeds. Each domain's batch stream uses a domain-specific seeded generator, identical across variants. This avoids target loop order changing source/domain initialization or batches. Class token embeddings and tokenizer are official; context width and aligned feature width are inferred separately.

Frozen image features are cached once with official deterministic Resize/CenterCrop/RGB/Normalize preprocessing and float32 image encoding. There is no augmentation in the official path used here. Cache identities include absolute image paths, file sizes/mtimes, weight SHA256, precision, and preprocessing. Cached target features contain no class labels. No target bank, confidence filtering, or distance-derived class weighting is added.

The protocol is transductive Office-Home: target images are seen without labels during training, then the same images receive labels only during final evaluation. This dataset has domain/class folders and no separately provided train/test split. Source folders define sorted class indices; all domains must have exactly the same 65 names. Future explicit splits require a separate, uniformly applied data configuration.

Checkpoint state includes prompts, bank, optimizer, scheduler, step, full configuration, CPU/CUDA/Python/NumPy RNG states, and each domain's batch-stream permutation/cursor/generator. Completed runs cannot be overwritten. Incomplete runs require explicit resume; resume rejects changed implementation/configuration/data fingerprints. Training logs after the saved step are truncated before resume. Prompts are the only optimizer parameters. Image features and CLIP are reproducible external inputs and are not duplicated in checkpoints.

The initial smoke tests revealed that torchvision's transform representation includes a callable memory address. Cache fingerprints now strip that address while retaining the preprocessing parameters, so independent processes reuse the same frozen features. Early smoke/cache artifacts are preserved separately; formal runs use the corrected fingerprints.

## Verification and execution

The matrix retry path restores available checkpoints. A failed attempt without a checkpoint is archived under `failed_attempts/` and restarted from the original seed; its logs are preserved rather than overwritten.

The final suite has 20 passing tests, including independent B parameter storage, empty/partial reports, and multi-seed aggregation. Standard deviations use the sample definition across seeds. Single-seed deviation fields are empty; incomplete matrices cannot produce complete macro scores.

Synthetic tests cover parameter identity, every individual gradient route, same-valued initialization, detached logits fusion, cumulative means, single-source bank invariance, equivalence of sequential backward to the total loss, unlabeled targets, vocabulary checks, evaluation tail batches, exact checkpoint resume, and 20-run/36-result summaries.

For actual commands and failures see `docs/execution_log.md`, per-run `environment.json`, `train.log`, and matrix logs. Real measured results and incomplete runs are generated in `reports/officehome_mtda_report.md`. No accuracy claim follows from code or smoke-test success alone.
