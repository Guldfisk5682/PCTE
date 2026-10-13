#!/usr/bin/env python3
"""Offline R4 error attribution with optional exact Source CE reconstruction."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from mtda.data import BatchStream, SourceImages, discover_domains
from mtda.experiment import seed_everything, write_json, file_sha
from mtda.prompts import MTDAPromptGenerator, PromptModel
from scripts.agreement_metrics import error_partitions
from scripts.summarize_b_reliability import flat_csv


def caches_for(cfg, cache_dir):
    return {d: torch.load(Path(cache_dir) / "domain_class" / f"{key}.pt", weights_only=True, map_location="cpu")
            for d, key in cfg["domain_cache_fingerprints"].items()}


def verify_samples(data, cache):
    common = Path(os.path.commonpath([str(p.parent) for p in data.paths]))
    ids = tuple(hashlib.sha256(p.relative_to(common).as_posix().encode()).hexdigest() for p in data.paths)
    if ids != tuple(cache["sample_ids"]):
        raise ValueError("offline evaluation labels do not match cache sample IDs")


def source_trajectory(root, cache_dir, output, device, final_only=False):
    """Diagnostic reconstruction only; no target data/labels/losses or new student runs."""
    from clip_custom import clip
    cfg = json.loads((root / "config.json").read_text())
    signature = {"reference_checkpoint_sha256": file_sha(root / "last.pt"), "device": device, "final_only": final_only}
    if output.exists():
        result = torch.load(output, weights_only=True, map_location="cpu")
        if result["signature"] != signature:
            raise ValueError("Source reconstruction provenance mismatch")
        return result
    seed_everything(cfg["seed"])
    clip_model, _ = clip.load(cfg["clip_checkpoint"], device=device, jit=False)
    clip_model = clip_model.float().eval().requires_grad_(False)
    prompts = MTDAPromptGenerator(cfg["classnames"], clip_model, clip.tokenize, cfg["source"], cfg["targets"],
                                 "B", cfg["seed"], cfg["m1"], cfg["m2"]).to(device)
    tokens = clip.tokenize([f"A photo of a {n.replace('_', ' ')}" for n in cfg["classnames"]]).to(device)
    model = PromptModel(clip_model, prompts, tokens).to(device)
    if final_only:
        saved = torch.load(root / "last.pt", weights_only=False, map_location="cpu")
        prompts.load_state_dict(saved["prompts"])
        with torch.no_grad():
            text = model.forward_source_text().cpu()
        result = {"signature": signature, "texts": {cfg["steps"]: text}, "base": model.base_features.cpu(),
                  "scale": model.scale.cpu(), "source_final_parameters_exact": True,
                  "diagnostic_source_updates": 0, "target_training_updates": 0,
                  "scope": "original final checkpoint loaded exactly; CPU inference only"}
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output)
        return result
    source = cfg["source"]
    cache = caches_for(cfg, cache_dir)[source]
    labels = SourceImages(discover_domains(cfg["data_root"])[source], None, cfg["classnames"]).labels
    seed = int.from_bytes(hashlib.sha256(f"{cfg['seed']}:batch:{source}".encode()).digest()[:8], "big") % (2**63)
    stream = BatchStream(cache["z_pre"], cfg["batch_size"], seed, labels)
    parameters = [prompts.class_context(source), prompts.domain_contexts["d0"]]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["steps"])
    points = {int(p.name.split('_')[0]) for p in (root / "inference").glob("*.pt")}
    texts = {}
    for step in range(cfg["steps"] + 1):
        if step:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            raw, truth = (v.to(device) for v in stream.next_batch())
            loss = F.cross_entropy(model.scale * F.normalize(raw, dim=-1) @ model.forward_source_text().T, truth)
            loss.backward()
            optimizer.step()
            scheduler.step()
        if step in points:
            with torch.no_grad():
                texts[step] = model.forward_source_text().cpu()
    saved = torch.load(root / "last.pt", weights_only=False, map_location="cpu")
    for key in ("class_contexts.d0", "domain_contexts.d0"):
        if not torch.equal(prompts.state_dict()[key].cpu(), saved["prompts"][key]):
            raise AssertionError("Source CE reconstruction differs from recorded R4")
    result = {"signature": signature, "texts": texts, "base": model.base_features.cpu(),
              "scale": model.scale.cpu(), "source_final_parameters_exact": True,
              "diagnostic_source_updates": cfg["steps"], "target_training_updates": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    return result


def analyze(root, cache_dir, trajectory=None):
    cfg = json.loads((root / "config.json").read_text())
    caches, domains = caches_for(cfg, cache_dir), discover_domains(cfg["data_root"])
    truths = {}
    for target in cfg["targets"]:
        data = SourceImages(domains[target], None, cfg["classnames"])
        verify_samples(data, caches[target])
        truths[target] = data.labels
    snapshots = {}
    for path in (root / "teacher_snapshots").glob("*.pt"):
        r = torch.load(path, weights_only=True, map_location="cpu")
        snapshots[r["snapshot"]["version"]] = r["snapshot"]["text"]
    rows, class_rows, contributions = [], [], []
    for path in sorted((root / "inference").glob("*.pt")):
        state = torch.load(path, weights_only=True, map_location="cpu")
        for target, result in state["domains"].items():
            if tuple(result["sample_ids"]) != tuple(caches[target]["sample_ids"]):
                raise ValueError("inference sample order differs from offline cache")
            z = caches[target]["z"]
            versions = {d: v for d, v in state["teacher_version"].items() if d not in ("Source", target)}
            votes = {d: (z @ snapshots[v].T).argmax(-1) for d, v in versions.items()}
            truth = truths[target]
            prefix = {"source": cfg["source"], "target": target, "variant": cfg["variant"], "seed": cfg["seed"],
                      "step": state["step"], "event": state["event"], "teacher_version": versions,
                      "other_teacher_count": len(versions), "ground_truth_scope": "offline analysis only"}
            metrics = error_partitions(result["base_prediction"], result["prediction"], truth, result["confidence"], votes)
            metrics["reference_half_uniform_weight"] = 1 / (1 + len(versions))
            metrics["full_similarity_weight_per_reference"] = .5 / (1 + len(versions))
            if trajectory is not None and state["step"] in trajectory["texts"] and cfg["teacher_mode"] == "uniform":
                source_text, base = trajectory["texts"][state["step"]], trajectory["base"]
                source_logits = z @ source_text.T
                source_prediction = source_logits.argmax(-1)
                prompt_votes = torch.stack([source_prediction,*votes.values()])
                prompt_unanimous = prompt_votes.eq(prompt_votes[0]).all(0)
                prompt_wrong = prompt_unanimous & source_prediction.ne(truth)
                metrics.update({"actual_source_prompt_accuracy": float(source_prediction.eq(truth).float().mean()),
                                "actual_prompt_teacher_agreement_rate": float(prompt_unanimous.float().mean()),
                                "actual_prompt_teacher_unanimous_wrong_rate": float(prompt_wrong.float().mean()),
                                "actual_prompt_teacher_unanimous_wrong_cross_rate": float((prompt_wrong & result["prediction"].eq(source_prediction)).float().mean()),
                                "actual_prompt_teacher_unanimous_precision": float(source_prediction[prompt_unanimous].eq(truth[prompt_unanimous]).float().mean()) if prompt_unanimous.any() else None,
                                "actual_prompt_vote_scope": "Source Prompt plus eligible other Target Prompts; excludes fixed Base CLIP"})
                reference_logits = source_logits
                for d, v in versions.items():
                    reference_logits = reference_logits + z @ snapshots[v].T
                base_logits = .5 * z @ base.T
                fused = base_logits + .5 * reference_logits / (1 + len(versions))
                full = fused.argmax(-1)
                # CPU vs GPU matrix arithmetic can differ on exact ties. Count and report, never silently replace saved outputs.
                metrics["cpu_recomputed_prediction_disagreements"] = int(full.ne(result["prediction"]).sum())
                for d, v in versions.items():
                    without = (base_logits + .5 * (reference_logits - z @ snapshots[v].T) / len(versions)).argmax(-1)
                    correct = full.eq(truth)
                    loo_correct = without.eq(truth)
                    contributions.append({**prefix, "removed_teacher": d,
                                          "full_accuracy": float(correct.float().mean()),
                                          "leave_one_out_accuracy": float(loo_correct.float().mean()),
                                          "teacher_marginal_gain": float((correct.float() - loo_correct.float()).mean()),
                                          "helped_count": int((correct & ~loo_correct).sum()),
                                          "harmed_count": int((~correct & loo_correct).sum()),
                                          "note": "fixed other snapshots; teacher prediction effect, not retrained student accuracy"})
            rows.append({**prefix, **metrics})
            if state["event"] == "final":
                for k, name in enumerate(cfg["classnames"]):
                    mask = truth.eq(k)
                    class_rows.append({**prefix, "class": k, "class_name": name,
                                       **error_partitions(result["base_prediction"][mask], result["prediction"][mask], truth[mask],
                                                          result["confidence"][mask], {d: v[mask] for d, v in votes.items()})})
    return rows, class_rows, contributions


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs-dir", default="runs/b_reliability/R4")
    cli.add_argument("--cache-dir", default="runs/feature_cache")
    cli.add_argument("--output-dir", default="reports/b_followup/agreement")
    cli.add_argument("--trajectory-dir", default="runs/b_followup_analysis/source_trajectory")
    cli.add_argument("--reconstruct-source", action="store_true")
    cli.add_argument("--final-source-only", action="store_true", help="load final Source checkpoint for leave-one-out; no training reconstruction")
    cli.add_argument("--source-reference-root", help="reuse the original R4 Source-only diagnostic trajectory")
    cli.add_argument("--device", default="cuda:0")
    cli.add_argument("--sources", nargs="+")
    cli.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    args = cli.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    all_rows, classes, loo = [], [], []
    for path in sorted(Path(args.runs_dir).rglob("metrics.json")):
        cfg = json.loads(path.read_text())["config"]
        if cfg["seed"] not in args.seeds or args.sources and cfg["source"] not in args.sources:
            continue
        trajectory = None
        if args.reconstruct_source or args.final_source_only:
            reference_root = path.parent
            if args.source_reference_root:
                reference_root = Path(args.source_reference_root) / cfg["source"].replace(" ", "_") / f"seed_{cfg['seed']}"
                reference_cfg = json.loads((reference_root / "config.json").read_text())
                for key in ("source", "seed", "steps", "batch_size", "m1", "m2", "learning_rate", "classnames", "checkpoint_sha256", "domain_cache_fingerprints"):
                    if cfg[key] != reference_cfg[key]:
                        raise ValueError("Source diagnostic trajectory configuration mismatch")
            trajectory = source_trajectory(reference_root, args.cache_dir,
                Path(args.trajectory_dir) / cfg["source"].replace(" ", "_") / f"seed_{cfg['seed']}.pt", args.device, args.final_source_only)
        rows, class_rows, contributions = analyze(path.parent, args.cache_dir, trajectory)
        all_rows.extend(rows)
        classes.extend(class_rows)
        loo.extend(contributions)
        print(f"ANALYZED {cfg['source']} seed={cfg['seed']}: {len(rows)} points; {len(contributions)} leave-one-out records", flush=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "agreement.json", all_rows)
    write_json(output / "hard_classes.json", classes)
    write_json(output / "teacher_leave_one_out.json", loo)
    flat_csv(output / "agreement.csv", all_rows)
    flat_csv(output / "hard_classes.csv", classes)
    flat_csv(output / "teacher_leave_one_out.csv", loo)
    write_json(output / "protocol.json", {"training_feedback": False, "uses_target_labels": True,
                                           "oracle_teachers": False, "source_trajectory_diagnostic_only": args.reconstruct_source,
                                           "source_final_checkpoint_inference_only": args.final_source_only,
                                           "source_reconstruction_final_parameters_exact": args.reconstruct_source or args.final_source_only})


if __name__ == "__main__":
    main()
