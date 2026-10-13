"""Reuse immutable R4 snapshots without inspecting target labels or accuracy."""

import json
from pathlib import Path

import torch

from mtda.experiment import file_sha
from mtda.reliability.bank import TeacherSnapshot

METHODS = {
    "FUniform": {"mode": "uniform", "admission": "progressive", "snapshot": "reference"},
    **{f"FBeta{label}": {"mode": "distance", "beta": beta, "admission": "progressive", "snapshot": "reference"}
       for label, beta in (("0", 0.), ("0p1", .1), ("1", 1.), ("10", 10.))},
    "FStartRef": {"mode": "uniform", "admission": "start", "snapshot": "reference"},
    "FStartInit": {"mode": "uniform", "admission": "start", "snapshot": "online"},
}


def validate_reference(config, args, source, model, caches, order, checkpoint_sha):
    expected = {"variant": "R4", "source": source, "seed": args.seed, "steps": args.steps,
                "batch_size": args.batch_size, "m1": args.m1, "m2": args.m2,
                "learning_rate": args.learning_rate, "lambda_target": args.lambda_target,
                "threshold": .7, "checkpoint_sha256": checkpoint_sha, "domain_order": order,
                "admission_fractions": args.admission_fractions,
                "classnames": list(model.prompts.classnames), "target_training_ground_truth": False,
                "checkpoint_selection": "fixed final budget",
                "domain_cache_fingerprints": {d: c.fingerprint for d, c in caches.items()}}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("R4 reference teacher configuration differs from follow-up")


def reference_snapshots(args, source, model, caches, order, checkpoint_sha):
    if METHODS[args.variant]["snapshot"] == "online":
        return {}, {"shared_teacher_preparation": False, "teacher_preparation_budget": 0}
    root = Path(args.reference_root) / "R4" / source.replace(" ", "_") / f"seed_{args.seed}"
    config = json.loads((root / "config.json").read_text())
    if not (root / "metrics.json").is_file():
        raise ValueError("reference R4 run must have completed its fixed budget")
    validate_reference(config, args, source, model, caches, order, checkpoint_sha)
    snapshots, hashes = {}, {}
    for path in sorted((root / "teacher_snapshots").glob("*_admission_*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True)
        domain = record["target"]
        if domain in snapshots or domain not in model.prompts.targets:
            raise ValueError("duplicate or unexpected reference domain")
        snapshot = TeacherSnapshot.from_state(record["snapshot"], model.scale.device)
        rank = order["order"].index(domain)
        if snapshot.step != config["admission_steps"][rank] or snapshot.domain != domain:
            raise ValueError("reference snapshot generation milestone mismatch")
        snapshots[domain] = snapshot
        hashes[domain] = {"path": str(path.resolve()), "sha256": file_sha(path), "version": snapshot.version,
                          "generation_step": snapshot.step}
    if set(snapshots) != set(model.prompts.targets):
        raise ValueError("one immutable reference snapshot is required per target")
    return snapshots, {"root": str(root.resolve()), "config_sha256": file_sha(root / "config.json"),
                       "snapshots": hashes, "shared_teacher_preparation": True,
                       "teacher_preparation_budget": args.steps,
                       "teacher_selection": "fixed R4 admission milestones; no accuracy selection",
                       "lookahead_control": METHODS[args.variant]["admission"] == "start"}
