"""Configuration, provenance, exact batch-stream resume, and run artifacts."""

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from .data import (BatchStream, SourceImages, UnlabeledImages, discover_domains,
                   image_feature_cache, resolve_domain, validate_classes)
from .evaluate import evaluate_targets
from .prompts import MTDAPromptGenerator, PromptModel, context_template
from .source_bank import SourceCentroidBank
from .trainer import train_step

PROJECT = Path(__file__).resolve().parents[1]
HISTORY_FIELDS = ["step", "source_loss", "target_loss", "loss", "confidence",
                  "observed_classes", "classes", "samples", "lr", "elapsed_seconds", "target_losses"]


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(path, model, bank, optimizer, scheduler, streams, step, config, elapsed):
    state = {"prompts": model.prompts.state_dict(), "bank": bank.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "streams": {name: stream.state_dict() for name, stream in streams.items()},
             "rng": rng_state(), "step": step, "config": config, "elapsed_seconds": elapsed}
    temporary = Path(path).with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, bank, optimizer, scheduler, streams, config):
    # Only locally generated experiment checkpoints are accepted here.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["config"] != config:
        raise ValueError("resume configuration differs from saved configuration")
    model.prompts.load_state_dict(state["prompts"])
    bank.load_state_dict(state["bank"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    for name, stream in streams.items():
        stream.load_state_dict(state["streams"][name])
    restore_rng(state["rng"])
    return state


def run_experiment(args):
    from clip_custom import clip

    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable")
    domains = discover_domains(args.data_root)
    source = resolve_domain(args.source or "Art", domains)
    targets = [resolve_domain(name, domains) for name in args.target] if args.target else [
        name for name in domains if name != source]
    if source in targets or len(set(targets)) != len(targets):
        raise ValueError("targets must be unique and exclude the source")
    if args.variant == "C" and len(targets) != 1 and not args.prepare_cache_only:
        raise ValueError("C requires one --target; the matrix runner launches independent pairs")
    if args.variant in ("A", "B") and len(targets) != 3 and not args.smoke:
        raise ValueError("Office-Home A/B require all three targets (use --smoke for a pair check)")
    names = validate_classes(domains, source)
    checkpoint = Path(args.clip_checkpoint).resolve()
    if not checkpoint.is_file():
        raise ValueError(f"CLIP checkpoint not found: {checkpoint}")
    weight_sha = file_sha(checkpoint)
    clip_model, preprocess = clip.load(str(checkpoint), device=args.device, jit=False)
    clip_model = clip_model.float().eval().requires_grad_(False)
    selected = list(domains) if args.prepare_cache_only else [source, *targets]
    features, fingerprints = {}, {}
    for domain in selected:
        print(f"Loading frozen image features: {domain}", flush=True)
        dataset = UnlabeledImages(domains[domain], preprocess)
        features[domain], fingerprints[domain] = image_feature_cache(
            dataset, clip_model, args.cache_dir, weight_sha, args.feature_batch_size, args.workers)
    if args.prepare_cache_only:
        print("All four label-free image feature caches ready.", flush=True)
        return None
    prompts = MTDAPromptGenerator(names, clip_model, clip.tokenize, source, targets,
                                 args.variant, args.seed, args.m1, args.m2).to(args.device)
    base_tokens = clip.tokenize([f"A photo of a {name.replace('_', ' ')}" for name in names]).to(args.device)
    model = PromptModel(clip_model, prompts, base_tokens).to(args.device)
    bank = SourceCentroidBank(len(names), features[source].shape[-1]).to(args.device)
    optimizer = torch.optim.AdamW(prompts.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    source_labels = SourceImages(domains[source], preprocess, names).labels
    streams = {}
    for domain in [source, *targets]:
        stream_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:batch:{domain}".encode()).digest()[:8], "big") % (2**63)
        streams[domain] = BatchStream(features[domain], args.batch_size, stream_seed,
                                      source_labels if domain == source else None)
    code_files = [PROJECT / "train_mtda.py", *sorted((PROJECT / "mtda").glob("*.py"))]
    code_digest = hashlib.sha256("".join(file_sha(p) for p in code_files).encode()).hexdigest()
    config = {"variant": args.variant, "source": source, "targets": targets, "seed": args.seed,
              "steps": args.steps, "batch_size": args.batch_size, "m1": args.m1, "m2": args.m2,
              "learning_rate": args.learning_rate, "lambda_target": args.lambda_target,
              "optimizer": "AdamW", "weight_decay": 0.01, "scheduler": "cosine_per_update",
              "data_root": str(Path(args.data_root).resolve()), "clip_checkpoint": str(checkpoint),
              "checkpoint_sha256": weight_sha, "feature_fingerprints": fingerprints,
              "classnames": names, "feature_dim": features[source].shape[-1],
              "context_width": clip_model.ln_final.weight.numel(), "precision": "float32",
              "protocol": "transductive Office-Home, final-step target-prompt inference",
              "initialization": "sha256(seed:identity), CPU generator, normal std=0.02",
              "initial_class_sha256": hashlib.sha256(context_template(
                  (len(names), args.m1, clip_model.ln_final.weight.numel()), args.seed,
                  f"class:{source}").numpy().tobytes()).hexdigest(),
              "upstream_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
              "implementation_sha256": code_digest}
    root = Path(args.output_dir) / args.variant / source.replace(" ", "_")
    if args.variant == "C":
        root /= targets[0].replace(" ", "_")
    root /= f"seed_{args.seed}"
    if (root / "metrics.json").exists():
        raise FileExistsError(f"completed run is preserved: {root}")
    if (root / "config.json").exists() and not args.resume:
        raise FileExistsError(f"existing run requires --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    start_step, prior_elapsed = 0, 0.0
    if args.resume:
        if not (root / "last.pt").exists():
            raise FileNotFoundError(f"no checkpoint to resume: {root / 'last.pt'}")
        state = load_checkpoint(root / "last.pt", model, bank, optimizer, scheduler, streams, config)
        start_step, prior_elapsed = state["step"], state["elapsed_seconds"]
        if (root / "history.csv").exists():
            with (root / "history.csv").open() as handle:
                rows = [row for row in csv.DictReader(handle) if int(row["step"]) <= start_step]
            with (root / "history.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
    else:
        write_json(root / "config.json", config)
        torch.save({"class": {k: v.detach().cpu() for k, v in prompts.class_contexts.items()},
                    "domain": {d: prompts.domain_contexts[prompts.domain_keys[d]].detach().cpu()
                               for d in prompts.domains}}, root / "initialization.pt")
    import torchvision
    environment = {"python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
                   "numpy": np.__version__, "cuda": torch.version.cuda,
                   "device": args.device, "gpu": torch.cuda.get_device_name(args.device)
                   if args.device.startswith("cuda") else None,
                   "command": [sys.executable, *sys.argv], "cpu_threads": args.cpu_threads,
                   "workers": args.workers, "feature_batch_size": args.feature_batch_size,
                   "eval_batch_size": args.eval_batch_size, "preprocess": str(preprocess)}
    prior_commands = []
    if (root / "environment.json").exists():
        previous_environment = json.loads((root / "environment.json").read_text())
        prior_commands = previous_environment.get("invocations", [previous_environment["command"]])
    environment["invocations"] = [*prior_commands, environment["command"]]
    write_json(root / "environment.json", environment)
    log = (root / "train.log").open("a", buffering=1)
    log.write(json.dumps({"event": "resume" if args.resume else "start", **environment}) + "\n")
    log.write(json.dumps({"trainable": {n: list(p.shape) for n, p in prompts.named_parameters()},
                          "frozen_clip_parameters": sum(p.numel() for p in clip_model.parameters())}) + "\n")
    if any(p.requires_grad for p in clip_model.parameters()):
        raise AssertionError("CLIP must be frozen")
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
    started = time.monotonic()
    stop_step = min(args.steps, args.stop_after) if args.stop_after else args.steps
    if stop_step < start_step:
        raise ValueError("stop-after precedes saved checkpoint")
    history_exists = (root / "history.csv").exists()
    with (root / "history.csv").open("a", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if not history_exists:
            writer.writeheader()
        for step in range(start_step + 1, stop_step + 1):
            metrics = train_step(model, bank, optimizer, streams[source].next_batch(),
                                 {d: streams[d].next_batch() for d in targets}, args.lambda_target)
            scheduler.step()
            elapsed = prior_elapsed + time.monotonic() - started
            row = {"step": step, **metrics, "lr": optimizer.param_groups[0]["lr"],
                   "elapsed_seconds": elapsed}
            row["target_losses"] = json.dumps(row["target_losses"], sort_keys=True)
            writer.writerow(row)
            if step == 1 or step % args.log_every == 0 or step == stop_step:
                message = f"{args.variant} {source} step {step}/{args.steps}: loss={metrics['loss']:.4f}, bank={metrics['observed_classes']}/{len(names)}, elapsed={elapsed:.1f}s"
                print(message, flush=True)
                log.write(message + "\n")
            if step % args.checkpoint_every == 0 or step == stop_step:
                save_checkpoint(root / "last.pt", model, bank, optimizer, scheduler,
                                streams, step, config, elapsed)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    elapsed = prior_elapsed + time.monotonic() - started
    if stop_step == args.steps:
        # Reload the actual final checkpoint, then recompute all target prototypes.
        load_checkpoint(root / "last.pt", model, bank, optimizer, scheduler, streams, config)
        results = evaluate_targets(model, domains, features, names, preprocess, args.eval_batch_size)
        metrics = {"status": "complete", "step": stop_step, "config": config, "targets": results,
                   "macro_accuracy": sum(v["accuracy"] for v in results.values()) / len(results),
                   "training_seconds": elapsed, "peak_memory_bytes": torch.cuda.max_memory_allocated(args.device)
                   if args.device.startswith("cuda") else 0, "bank": bank.diagnostics()}
        write_json(root / "metrics.json", metrics)
        log.write(json.dumps(metrics) + "\n")
        print(json.dumps({"run": str(root), "targets": results}), flush=True)
    else:
        print(f"Paused at step {stop_step}; continue with --resume.", flush=True)
    log.close()
    return root
