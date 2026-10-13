#!/usr/bin/env python3
"""Source-CE-only prompt control; target images enter only final evaluation."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn import functional as F

from mtda.data import (BatchStream, SourceImages, UnlabeledImages, discover_domains,
                       image_feature_cache, resolve_domain, validate_classes)
from mtda.evaluate import evaluate_targets
from mtda.experiment import (PROJECT, file_sha, load_checkpoint, save_checkpoint,
                             seed_everything, write_json)
from mtda.prompts import MTDAPromptGenerator, PromptModel
from mtda.source_bank import SourceCentroidBank


def baseline_digest():
    files = [PROJECT / "train_mtda.py", *sorted((PROJECT / "mtda").glob("*.py"))]
    return hashlib.sha256("".join(file_sha(path) for path in files).encode()).hexdigest()


def source_parameters(model):
    """Activate only class context and the source domain context."""
    model.prompts.requires_grad_(False)
    source = model.prompts.source
    parameters = [model.prompts.class_context(source),
                  model.prompts.domain_contexts[model.prompts.domain_keys[source]]]
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def source_step(model, bank, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    raw, labels = (tensor.to(model.scale.device) for tensor in batch)
    bank.update(raw, labels)
    text = model.forward_source_text()
    loss = F.cross_entropy(model.scale * F.normalize(raw, dim=-1) @ text.T, labels)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite source-only loss")
    loss.backward()
    for parameter in model.prompts.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError("nonfinite source-only gradient")
    optimizer.step()
    return float(loss.detach())


class SourcePredictor:
    """Adapt inference policy while reusing the original complete evaluation."""
    def __init__(self, model):
        self.model = model
        self.prompts = model.prompts
        self.base_features = model.base_features

    @property
    def scale(self):
        return self.model.scale

    def eval(self):
        self.model.eval()
        return self

    def forward_target_text(self, domain):
        if domain not in self.prompts.targets:
            raise ValueError(f"not an evaluation target: {domain}")
        return self.model.forward_source_text()


def run(args):
    from clip_custom import clip
    import numpy as np
    import torchvision

    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    domains = discover_domains(args.data_root)
    source = resolve_domain(args.source, domains)
    targets = [d for d in domains if d != source]
    names = validate_classes(domains, source)
    weight_path = Path(args.clip_checkpoint).resolve()
    weight_sha = file_sha(weight_path)
    clip_model, preprocess = clip.load(str(weight_path), device=args.device, jit=False)
    clip_model = clip_model.float().eval().requires_grad_(False)
    # No target images or target labels are loaded before training is complete.
    source_images = UnlabeledImages(domains[source], preprocess)
    features, fingerprint = image_feature_cache(source_images, clip_model, args.cache_dir,
                                                weight_sha, args.feature_batch_size, args.workers)
    prompts = MTDAPromptGenerator(names, clip_model, clip.tokenize, source, targets,
                                 "A", args.seed, args.m1, args.m2).to(args.device)
    base_tokens = clip.tokenize([f"A photo of a {name.replace('_', ' ')}" for name in names]).to(args.device)
    model = PromptModel(clip_model, prompts, base_tokens).to(args.device)
    parameters = source_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    bank = SourceCentroidBank(len(names), features.shape[-1]).to(args.device)
    labels = SourceImages(domains[source], preprocess, names).labels
    stream_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:batch:{source}".encode()).digest()[:8], "big") % (2**63)
    streams = {source: BatchStream(features, args.batch_size, stream_seed, labels)}
    core_sha = baseline_digest()
    config = {"variant": "SourceOnly", "source": source, "targets": targets, "seed": args.seed,
              "steps": args.steps, "batch_size": args.batch_size, "m1": args.m1, "m2": args.m2,
              "learning_rate": args.learning_rate, "optimizer": "AdamW", "weight_decay": 0.01,
              "scheduler": "cosine_per_update", "objective": "source_cross_entropy_only",
              "predictor": "source_prompt_only", "target_training_images": 0,
              "trainable_parameters": ["class_context", "source_domain_context"],
              "data_root": str(Path(args.data_root).resolve()), "clip_checkpoint": str(weight_path),
              "checkpoint_sha256": weight_sha, "feature_fingerprints": {source: fingerprint},
              "classnames": names, "feature_dim": features.shape[-1], "precision": "float32",
              "context_width": clip_model.ln_final.weight.numel(),
              "initialization": "sha256(seed:identity), CPU generator, normal std=0.02",
              "initial_class_sha256": hashlib.sha256(prompts.class_context(source).detach().cpu().numpy().tobytes()).hexdigest(),
              "baseline_implementation_sha256": core_sha,
              "implementation_sha256": hashlib.sha256((file_sha(Path(__file__)) + core_sha).encode()).hexdigest(),
              "upstream_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()}
    root = Path(args.output_dir) / source.replace(" ", "_") / f"seed_{args.seed}"
    if (root / "metrics.json").exists():
        raise FileExistsError(f"completed run is preserved: {root}")
    if (root / "config.json").exists() and not args.resume:
        raise FileExistsError(f"existing run requires --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    start_step, prior_elapsed = 0, 0.0
    if args.resume:
        saved = load_checkpoint(root / "last.pt", model, bank, optimizer, scheduler, streams, config)
        start_step, prior_elapsed = saved["step"], saved["elapsed_seconds"]
        with (root / "history.csv").open() as handle:
            history = [r for r in csv.DictReader(handle) if int(r["step"]) <= start_step]
        with (root / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "source_loss", "lr", "elapsed_seconds"])
            writer.writeheader()
            writer.writerows(history)
    else:
        write_json(root / "config.json", config)
        torch.save({"class": prompts.class_context(source).detach().cpu(),
                    "source_domain": prompts.domain_contexts[prompts.domain_keys[source]].detach().cpu()},
                   root / "initialization.pt")
    environment = {"python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
                   "numpy": np.__version__, "cuda": torch.version.cuda, "device": args.device,
                   "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
                   "command": [sys.executable, *sys.argv], "eval_batch_size": args.eval_batch_size}
    previous = json.loads((root / "environment.json").read_text()) if (root / "environment.json").exists() else {}
    environment["invocations"] = [*previous.get("invocations", []), environment["command"]]
    write_json(root / "environment.json", environment)
    stop = min(args.steps, args.stop_after) if args.stop_after else args.steps
    if stop < start_step:
        raise ValueError("stop-after precedes checkpoint")
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.monotonic()
    fields = ["step", "source_loss", "lr", "elapsed_seconds"]
    exists = (root / "history.csv").exists()
    with (root / "history.csv").open("a", newline="", buffering=1) as handle, (root / "train.log").open("a", buffering=1) as log:
        log.write(json.dumps(environment) + "\n")
        log.write(json.dumps({"trainable": {n: list(p.shape) for n, p in model.named_parameters() if p.requires_grad}}) + "\n")
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for step in range(start_step + 1, stop + 1):
            loss = source_step(model, bank, optimizer, streams[source].next_batch())
            scheduler.step()
            elapsed = prior_elapsed + time.monotonic() - started
            writer.writerow({"step": step, "source_loss": loss,
                             "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": elapsed})
            if step == 1 or step % args.log_every == 0 or step == stop:
                message = f"SourceOnly {source} step {step}/{args.steps}: loss={loss:.4f}, elapsed={elapsed:.1f}s"
                print(message, flush=True)
                log.write(message + "\n")
            if step % args.checkpoint_every == 0 or step == stop:
                save_checkpoint(root / "last.pt", model, bank, optimizer, scheduler, streams, step, config, elapsed)
        elapsed = prior_elapsed + time.monotonic() - started
        if stop != args.steps:
            print(f"Paused at {stop}; continue with --resume.", flush=True)
            return root
        load_checkpoint(root / "last.pt", model, bank, optimizer, scheduler, streams, config)
        all_features = {source: features}
        evaluation_fingerprints = {}
        for domain in targets:
            dataset = UnlabeledImages(domains[domain], preprocess)
            all_features[domain], evaluation_fingerprints[domain] = image_feature_cache(
                dataset, clip_model, args.cache_dir, weight_sha, args.feature_batch_size, args.workers)
        results = evaluate_targets(SourcePredictor(model), domains, all_features, names, preprocess, args.eval_batch_size)
        metrics = {"status": "complete", "step": stop, "config": config, "targets": results,
                   "macro_accuracy": sum(v["accuracy"] for v in results.values()) / len(results),
                   "evaluation_fingerprints": evaluation_fingerprints, "training_seconds": elapsed,
                   "peak_memory_bytes": torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else 0,
                   "bank": bank.diagnostics()}
        write_json(root / "metrics.json", metrics)
        log.write(json.dumps(metrics) + "\n")
        print(json.dumps({"run": str(root), "targets": results}), flush=True)
    return root


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", required=True)
    cli.add_argument("--data-root", required=True)
    cli.add_argument("--clip-checkpoint", required=True)
    cli.add_argument("--device", default="cuda:0")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--steps", type=int, default=1000)
    cli.add_argument("--batch-size", type=int, default=30)
    cli.add_argument("--m1", type=int, default=16)
    cli.add_argument("--m2", type=int, default=16)
    cli.add_argument("--learning-rate", type=float, default=0.003)
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--cpu-threads", type=int, default=4)
    cli.add_argument("--feature-batch-size", type=int, default=128)
    cli.add_argument("--eval-batch-size", type=int, default=128)
    cli.add_argument("--cache-dir", default="runs/feature_cache")
    cli.add_argument("--output-dir", default="runs/source_prompt_only")
    cli.add_argument("--checkpoint-every", type=int, default=100)
    cli.add_argument("--log-every", type=int, default=20)
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--stop-after", type=int)
    return cli


def main():
    cli = parser()
    args = cli.parse_args()
    if any(getattr(args, key) < 1 for key in ("steps", "batch_size", "m1", "m2", "cpu_threads",
                                             "feature_batch_size", "eval_batch_size", "checkpoint_every", "log_every")):
        cli.error("steps, sizes, context lengths, and logging intervals must be positive")
    if args.workers < 0 or args.learning_rate <= 0 or (args.stop_after is not None and args.stop_after < 1):
        cli.error("invalid workers, learning rate, or stop-after")
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    main()
