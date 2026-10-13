"""Isolated B reliability experiment lifecycle, stage events, and exact resume."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from mtda.data import BatchStream, SourceImages, UnlabeledImages, discover_domains, resolve_domain, validate_classes
from mtda.experiment import PROJECT, file_sha, write_json, seed_everything, rng_state, restore_rng
from mtda.prompts import MTDAPromptGenerator, PromptModel
from mtda.source_bank import SourceCentroidBank
from .bank import TeacherRegistry, TeacherSnapshot, rebuild_target_bank, predict_cache
from .cache import target_cache
from .diagnostics import full_inference, bank_diagnostics
from .offline import evaluate_point, analyze_offline
from .schedule import BudgetSchedule, difficulty_order, MODES
from .training import BranchOptimizers, controlled_step, freeze_target

SEALED_BASELINE = "4ea3c0f993cdea0b38ff165236b94044e1308d2bec7c55b178fb9a13727edc28"
HISTORY_FIELDS = ("source", "target", "variant", "seed", "step", "stage", "teacher_version",
                  "source_ce", "target_spl", "total_loss", "lr", "source_lr", "source_update_count",
                  "target_update_count", "optimizer_step_calls", "source_samples", "target_samples", "elapsed_seconds")


def implementation_hashes():
    files = [PROJECT / "train_mtda.py", *sorted((PROJECT / "mtda").glob("*.py"))]
    baseline = hashlib.sha256("".join(file_sha(p) for p in files).encode()).hexdigest()
    if baseline != SEALED_BASELINE:
        raise ValueError("preserved A/B/C implementation changed")
    new = [PROJECT / "train_b_reliability.py", *sorted(Path(__file__).parent.glob("*.py"))]
    digest = hashlib.sha256((baseline + "".join(file_sha(p) for p in new)).encode()).hexdigest()
    return baseline, digest


def append_event(path, record):
    with Path(path).open("a", buffering=1) as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_state(path, model, source_bank, registry, optimizers, streams, step, config, elapsed, frozen):
    state = {"prompts": model.prompts.state_dict(), "source_bank": source_bank.state_dict(),
             "registry": registry.state_dict(), "branches": optimizers.state_dict(),
             "streams": {d: s.state_dict() for d, s in streams.items()}, "step": step,
             "config": config, "elapsed_seconds": elapsed, "frozen_targets": list(frozen), "rng": rng_state()}
    temporary = Path(path).with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_state(path, model, source_bank, registry, optimizers, streams, config):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["config"] != config:
        raise ValueError("resume configuration differs from checkpoint")
    model.prompts.load_state_dict(state["prompts"])
    source_bank.load_state_dict(state["source_bank"])
    registry.load_state_dict(state["registry"], model.scale.device)
    optimizers.load_state_dict(state["branches"])
    for domain, stream in streams.items():
        stream.load_state_dict(state["streams"][domain])
    for domain in state["frozen_targets"]:
        freeze_target(model, domain)
    restore_rng(state["rng"])
    return state


def obtain_order(args, model, caches, source, checkpoint_sha):
    path = Path(args.order_file) if args.order_file else Path(args.output_dir) / "orders" / f"{source.replace(' ', '_')}.json"
    teacher_path = Path(args.order_teacher_root) / source.replace(" ", "_") / f"seed_{args.order_teacher_seed}" / "last.pt"
    if not teacher_path.is_file():
        raise FileNotFoundError(f"trained source-only ordering teacher missing: {teacher_path}")
    teacher_hash = file_sha(teacher_path)
    manual = [resolve_domain(d, {c: None for c in [source, *caches]}) for d in args.target_order] if args.target_order else None
    expected = {"source": source, "ordering_teacher_sha256": teacher_hash,
                "checkpoint_sha256": checkpoint_sha, "manual_order": manual,
                "cache_fingerprints": {d: c.fingerprint for d, c in caches.items()}}
    if path.exists():
        record = json.loads(path.read_text())
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"fixed domain order provenance mismatch: {path}")
        return record
    state = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if state["config"]["source"] != source or state["config"]["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("ordering source-only teacher protocol mismatch")
    if state["config"]["classnames"] != list(model.prompts.classnames):
        raise ValueError("ordering source-only teacher class order mismatch")
    current_class = model.prompts.class_context(source).detach().clone()
    key = model.prompts.domain_keys[source]
    current_domain = model.prompts.domain_contexts[key].detach().clone()
    with torch.no_grad():
        model.prompts.class_context(source).copy_(state["prompts"]["class_contexts.shared"])
        model.prompts.domain_contexts[key].copy_(state["prompts"]["domain_contexts.d0"])
        source_text = model.forward_source_text()
        record = difficulty_order(caches, model.base_features, source_text, model.scale, args.threshold, manual)
        model.prompts.class_context(source).copy_(current_class)
        model.prompts.domain_contexts[key].copy_(current_domain)
    record.update({**expected, "ordering_teacher_checkpoint": str(teacher_path.resolve()),
                   "ordering_teacher_seed": args.order_teacher_seed,
                   "ordering_teacher_source_updates": state["step"],
                   "ordering_only_reuses_existing_source_model": True})
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, record)
    return record


def trim_resume_artifacts(root, step):
    if (root / "history.csv").exists():
        with (root / "history.csv").open() as handle:
            rows = [r for r in csv.DictReader(handle) if int(r["step"]) <= step]
        with (root / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    for name in ("diagnostics.jsonl", "evaluations.jsonl", "events.jsonl"):
        path = root / name
        if path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r["step"] <= step))
    for directory in ("inference", "bank_snapshots", "teacher_snapshots"):
        for path in (root / directory).glob("*.pt"):
            record = torch.load(path, map_location="cpu", weights_only=True)
            if record["step"] > step:
                path.unlink()


def run_experiment(args):
    from clip_custom import clip
    import numpy as np
    import torchvision

    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    baseline_hash, code_hash = implementation_hashes()
    domains = discover_domains(args.data_root)
    source = resolve_domain(args.source, domains)
    targets = [d for d in domains if d != source]
    names = validate_classes(domains, source)
    checkpoint = Path(args.clip_checkpoint).resolve()
    weight_sha = file_sha(checkpoint)
    clip_model, preprocess = clip.load(str(checkpoint), device=args.device, jit=False)
    clip_model = clip_model.float().eval().requires_grad_(False)
    caches = {d: target_cache(UnlabeledImages(domains[d], preprocess), d, clip_model, args.cache_dir,
                              weight_sha, args.feature_batch_size, args.workers) for d in [source, *targets]}
    target_caches = {d: caches[d] for d in targets}
    prompts = MTDAPromptGenerator(names, clip_model, clip.tokenize, source, targets, "B", args.seed,
                                 args.m1, args.m2).to(args.device)
    base_tokens = clip.tokenize([f"A photo of a {name.replace('_', ' ')}" for name in names]).to(args.device)
    model = PromptModel(clip_model, prompts, base_tokens).to(args.device)
    order = obtain_order(args, model, target_caches, source, weight_sha)
    if args.prepare_order_only:
        print(json.dumps({"event": "order_ready", "source": source, **order}), flush=True)
        return None
    schedule = BudgetSchedule(args.variant, tuple(order["order"]), args.steps, tuple(args.admission_fractions), args.refresh_interval)
    # Training batch order stays canonical, not E2H order, so joint B consumes exactly the original streams.
    source_labels = SourceImages(domains[source], preprocess, names).labels
    streams = {}
    for domain, cache in caches.items():
        seed = int.from_bytes(hashlib.sha256(f"{args.seed}:batch:{domain}".encode()).digest()[:8], "big") % (2**63)
        streams[domain] = BatchStream(cache.z_pre, args.batch_size, seed, source_labels if domain == source else None)
    source_bank = SourceCentroidBank(len(names), caches[source].z_pre.shape[-1]).to(args.device)
    registry = TeacherRegistry(targets, len(names), caches[source].z_pre.shape[-1], args.device)
    branches = BranchOptimizers(model, args.steps, args.learning_rate)
    config = {"variant": args.variant, "prompt_variant": "B", "source": source, "targets": targets,
              "seed": args.seed, "steps": args.steps, "global_steps": schedule.total_steps,
              "batch_size": args.batch_size, "m1": args.m1, "m2": args.m2, "learning_rate": args.learning_rate,
              "lambda_target": args.lambda_target, "target_gradient_scale": args.lambda_target / len(targets),
              "optimizer": "AdamW per independent branch", "weight_decay": 0.01,
              "scheduler": "cosine per branch effective update", "source_budget": args.steps,
              "target_budgets": {d: args.steps for d in targets},
              "source_update_rule": "every third sequential global step starting at 1" if schedule.sequential else "every global step",
              "threshold": args.threshold, "beta": args.beta, "refresh_interval": args.refresh_interval,
              "admission_fractions": args.admission_fractions, "admission_steps": list(schedule.admission_steps),
              "domain_order": order, "teacher_mode": MODES[args.variant],
              "data_root": str(Path(args.data_root).resolve()), "clip_checkpoint": str(checkpoint),
              "checkpoint_sha256": weight_sha, "classnames": names, "precision": "float32",
              "feature_fingerprints": {d: c.metadata["image_feature_fingerprint"] for d, c in caches.items()},
              "domain_cache_fingerprints": {d: c.fingerprint for d, c in caches.items()},
              "baseline_implementation_sha256": baseline_hash, "implementation_sha256": code_hash,
              "upstream_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
              "target_training_ground_truth": False, "checkpoint_selection": "fixed final budget",
              "eval_every": args.eval_every, "log_every": args.log_every}
    root = Path(args.output_dir) / args.variant / source.replace(" ", "_") / f"seed_{args.seed}"
    if (root / "metrics.json").exists():
        raise FileExistsError(f"completed run preserved: {root}")
    if (root / "config.json").exists() and not args.resume:
        raise FileExistsError(f"existing run requires --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("inference", "bank_snapshots", "teacher_snapshots"):
        (root / name).mkdir(exist_ok=True)
    start_step, prior_elapsed, frozen = 0, 0.0, []
    if args.resume:
        state = load_state(root / "last.pt", model, source_bank, registry, branches, streams, config)
        start_step, prior_elapsed, frozen = state["step"], state["elapsed_seconds"], state["frozen_targets"]
        trim_resume_artifacts(root, start_step)
    else:
        write_json(root / "config.json", config)
        torch.save({"class": {k: v.detach().cpu().clone() for k, v in prompts.class_contexts.items()},
                    "domain": {d: prompts.domain_contexts[prompts.domain_keys[d]].detach().cpu().clone()
                               for d in prompts.domains}}, root / "initialization.pt")
    environment = {"python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
                   "numpy": np.__version__, "cuda": torch.version.cuda, "device": args.device,
                   "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
                   "command": [sys.executable, *sys.argv], "preprocess": str(preprocess)}
    previous = json.loads((root / "environment.json").read_text()) if (root / "environment.json").exists() else {}
    environment["invocations"] = [*previous.get("invocations", []), environment["command"]]
    write_json(root / "environment.json", environment)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
    started = time.monotonic()
    def elapsed():
        return prior_elapsed + time.monotonic() - started

    def envelope(step, event, target=None):
        return {"source": source, "target": target, "variant": args.variant, "seed": args.seed,
                "step": step, "stage": schedule.stage(max(1, min(step, schedule.total_steps))), "event": event,
                "teacher_version": {"Source": f"source_update_{branches.updates[source]}",
                                    **{d: s.version for d, s in registry.snapshots.items()}}}

    def diagnose(step, event, stage_override=None):
        summaries, bundle = full_inference(model, target_caches, source_bank, registry, MODES[args.variant],
                                          args.beta, args.threshold, args.eval_batch_size)
        prefix = envelope(step, event)
        if stage_override is not None:
            prefix["stage"] = stage_override
        torch.save({**prefix, "domains": bundle}, root / "inference" / f"{step:06d}_{event}.pt")
        for domain, record in summaries.items():
            append_event(root / "diagnostics.jsonl", {**prefix, "target": domain, **record,
                         "source_bank": {**source_bank.diagnostics(), "class_counts": source_bank.running_count.cpu().tolist()},
                         "effective_updates": dict(branches.updates)})
        evaluation = evaluate_point(model, domains, target_caches, names, preprocess, args.eval_batch_size)
        append_event(root / "evaluations.jsonl", {**prefix, "targets": evaluation,
                     "effective_updates": dict(branches.updates), "elapsed_seconds": elapsed()})

    def admit_or_refresh(domain, step, event):
        tic = time.monotonic()
        snapshot = TeacherSnapshot.capture(model, domain, step)
        previous_bank = registry.bank.details.get(domain)
        record = {**envelope(step, event, domain), "new_teacher_version": snapshot.version,
                  "reference_teachers_for_each_target": {d: ["Source", *registry.references(d)] for d in targets}}
        with torch.no_grad():
            basic = predict_cache(target_caches[domain], 0.5 * model.base_features + 0.5 * model.forward_source_text(),
                                  model.scale, args.eval_batch_size)
        if MODES[args.variant] == "distance":
            candidate = rebuild_target_bank(target_caches[domain], snapshot, model.scale, len(names),
                                            args.threshold, args.eval_batch_size)
            record.update(bank_diagnostics(candidate, previous_bank))
            own = {"prediction": candidate["prediction"], "confidence": candidate["confidence"]}
            torch.save({**envelope(step, event, domain), "bank": candidate, "snapshot": snapshot.state_dict()},
                       root / "bank_snapshots" / f"{step:06d}_{event}_{domain.replace(' ', '_')}.pt")
        else:
            candidate = None
            own = predict_cache(target_caches[domain], snapshot.text, model.scale, args.eval_batch_size)
        record["snapshot_base_source_prediction_agreement"] = float((own["prediction"] == basic["prediction"]).float().mean())
        torch.save({**envelope(step, event, domain), "snapshot": snapshot.state_dict(), "prediction": own},
                   root / "teacher_snapshots" / f"{step:06d}_{event}_{domain.replace(' ', '_')}.pt")
        registry.commit(snapshot, candidate)
        record["refresh_seconds"] = time.monotonic() - tic
        record["teacher_version"] = envelope(step, event)["teacher_version"]
        append_event(root / "events.jsonl", record)

    if start_step == 0:
        diagnose(0, "initial")
        append_event(root / "events.jsonl", {**envelope(0, "order"), "domain_order": order})
    stop = min(schedule.total_steps, args.stop_after) if args.stop_after else schedule.total_steps
    if stop < start_step:
        raise ValueError("stop-after precedes checkpoint")
    with (root / "history.csv").open("a", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        for step in range(start_step + 1, stop + 1):
            if schedule.sequential and (step - 1) % args.steps == 0:
                if step > 1:
                    diagnose(step - 1, f"stage_{schedule.stage(step)}_start", schedule.stage(step))
                append_event(root / "events.jsonl", {**envelope(step, "stage_start", schedule.active_targets(step)[0]),
                             "stage_start_step": step, "stage_end_step": step + args.steps - 1,
                             "reference_teachers": ["Source", *registry.references(schedule.active_targets(step)[0])]})
            teacher_versions = envelope(step, "update")["teacher_version"]
            active = schedule.active_targets(step) if schedule.sequential else tuple(targets)
            result = controlled_step(model, source_bank, registry, branches,
                                     streams[source].next_batch() if schedule.update_source(step) else None,
                                     {d: streams[d].next_batch() for d in active}, MODES[args.variant], args.beta,
                                     args.lambda_target, collect_diagnostics=False)
            for domain in active:
                writer.writerow({"source": source, "target": domain, "variant": args.variant, "seed": args.seed,
                                 "step": step, "stage": schedule.stage(step), "teacher_version": json.dumps(teacher_versions),
                                 "source_ce": result["source_ce"], "target_spl": result["target_spl"][domain],
                                 "total_loss": result["total_loss"], "lr": branches.optimizers[domain].param_groups[0]["lr"],
                                 "source_lr": branches.optimizers[source].param_groups[0]["lr"],
                                 "source_update_count": branches.updates[source], "target_update_count": branches.updates[domain],
                                 "optimizer_step_calls": branches.step_calls, "source_samples": branches.samples[source],
                                 "target_samples": branches.samples[domain], "elapsed_seconds": elapsed()})
            admissions = schedule.admissions(step)
            refreshes = tuple(d for d in targets if args.variant == "R6" and step % args.refresh_interval == 0
                              and d in registry.snapshots and d not in admissions)
            if admissions or refreshes:
                diagnose(step, "pre_teacher_change")
                for domain in admissions:
                    admit_or_refresh(domain, step, "admission")
                for domain in refreshes:
                    admit_or_refresh(domain, step, "refresh")
                diagnose(step, "post_teacher_change")
            if schedule.stage_end(step):
                domain = schedule.active_targets(step)[0]
                freeze_target(model, domain)
                frozen.append(domain)
                diagnose(step, "stage_end")
                append_event(root / "events.jsonl", {**envelope(step, "stage_end", domain),
                             "effective_updates": dict(branches.updates)})
            elif step % args.eval_every == 0 or step == schedule.total_steps:
                diagnose(step, "final" if step == schedule.total_steps else "interval")
            if step == 20 or step % args.log_every == 0 or step == stop:
                print(json.dumps({"event": "progress", "variant": args.variant, "source": source,
                                  "step": step, "total_steps": schedule.total_steps, "elapsed_seconds": elapsed(),
                                  "effective_updates": branches.updates}), flush=True)
            if step % args.checkpoint_every == 0 or schedule.stage_end(step) or admissions or refreshes or step == stop:
                save_state(root / "last.pt", model, source_bank, registry, branches, streams,
                           step, config, elapsed(), frozen)
    if stop == schedule.total_steps:
        load_state(root / "last.pt", model, source_bank, registry, branches, streams, config)
        results = evaluate_point(model, domains, target_caches, names, preprocess, args.eval_batch_size)
        if any(count != args.steps for count in branches.updates.values()):
            raise AssertionError("effective source/target budgets differ from contract")
        offline = analyze_offline(root, domains, target_caches, names, preprocess, args.threshold)
        metrics = {"status": "complete", "config": config, "step": stop, "targets": results,
                   "macro_accuracy": sum(r["accuracy"] for r in results.values()) / len(results),
                   "effective_updates": branches.updates, "sample_exposures": branches.samples,
                   "global_steps": stop, "optimizer_step_calls": branches.step_calls,
                   "training_seconds": elapsed(), "peak_memory_bytes": torch.cuda.max_memory_allocated(args.device)
                   if args.device.startswith("cuda") else 0, "source_bank": source_bank.diagnostics(), "offline_diagnostics": offline}
        write_json(root / "metrics.json", metrics)
        print(json.dumps({"event": "complete", "run": str(root), "macro_accuracy": metrics["macro_accuracy"]}), flush=True)
    else:
        print(json.dumps({"event": "paused", "step": stop, "run": str(root)}), flush=True)
    return root
