"""Ground-truth evaluation and post-training diagnostics; never returns labels to training."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from mtda.data import SourceImages
from mtda.evaluate import evaluate_targets
from mtda.experiment import write_json


def evaluate_point(model, domains, caches, names, preprocess, batch_size):
    return evaluate_targets(model, domains, {d: c.z_pre for d, c in caches.items()}, names, preprocess, batch_size)


def quality(prediction, confidence, truth, classes, threshold=0.7):
    accepted = confidence.ge(threshold)
    confusion = torch.bincount(truth * classes + prediction, minlength=classes * classes).reshape(classes, classes)
    per_class = []
    for k in range(classes):
        mask = accepted & prediction.eq(k)
        true_class = truth.eq(k)
        per_class.append({"class": k, "accepted_predictions": int(mask.sum()),
                          "precision": float(prediction[mask].eq(truth[mask]).float().mean()) if mask.any() else None,
                          "pseudo_label_accuracy": float(prediction[true_class].eq(truth[true_class]).float().mean()) if true_class.any() else None})
    hard = sorted((r for r in per_class if r["pseudo_label_accuracy"] is not None),
                  key=lambda r: (r["pseudo_label_accuracy"], r["class"]))[:10]
    return {"all_sample_pseudo_label_accuracy": float(prediction.eq(truth).float().mean()),
            "accepted_pseudo_label_precision": float(prediction[accepted].eq(truth[accepted]).float().mean()) if accepted.any() else None,
            "accepted_samples": int(accepted.sum()), "per_class": per_class,
            "confusion_matrix_true_rows": confusion.tolist(), "hard_classes": [r["class"] for r in hard]}


@torch.no_grad()
def analyze_offline(root, domains, caches, names, preprocess, threshold=0.7):
    """Called only after final checkpoint evaluation, with no subsequent training selection."""
    root = Path(root)
    truths, oracles = {}, {}
    for domain, cache in caches.items():
        evaluation_data = SourceImages(domains[domain], preprocess, names)
        if len(evaluation_data) != len(cache.z_pre):
            raise ValueError("offline cache/evaluation ordering differs")
        truth = evaluation_data.labels
        truths[domain] = truth
        counts = torch.bincount(truth, minlength=len(names))
        sums = torch.zeros(len(names), cache.z_pre.shape[-1])
        sums.index_add_(0, truth, cache.z_pre)
        oracles[domain] = sums / counts.clamp_min(1).unsqueeze(-1)
    points = []
    for path in sorted((root / "inference").glob("*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True)
        for domain, record in state["domains"].items():
            points.append({"source": state["source"], "target": domain, "variant": state["variant"],
                           "seed": state["seed"], "step": state["step"], "stage": state["stage"],
                           "event": state["event"], "teacher_version": state["teacher_version"],
                           **quality(record["prediction"], record["confidence"], truths[domain], len(names), threshold)})
    banks, previous_banks = [], {}
    for path in sorted((root / "bank_snapshots").glob("*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True)
        result, domain = state["bank"], state["target"]
        cosine = F.cosine_similarity(result["centroids"], oracles[domain], dim=-1)
        valid = result["valid_mask"]
        record = {"source": state["source"], "target": domain, "variant": state["variant"],
                      "seed": state["seed"], "step": result["last_refresh_step"], "event": state["event"],
                      "teacher_version": result["teacher_version"], "coverage": float(valid.float().mean()),
                      "oracle_centroid_cosine_mean": float(cosine[valid].mean()) if valid.any() else None,
                      "oracle_centroid_cosine_by_class": [float(v) if m else None for v, m in zip(cosine, valid)],
                      **quality(result["prediction"], result["confidence"], truths[domain], len(names), threshold)}
        if domain in previous_banks:
            old = previous_banks[domain]
            record["previous_accepted_pseudo_label_precision"] = old["accepted_pseudo_label_precision"]
            new_precision, old_precision = record["accepted_pseudo_label_precision"], old["accepted_pseudo_label_precision"]
            record["accepted_precision_change"] = new_precision - old_precision if new_precision is not None and old_precision is not None else None
        previous_banks[domain] = record
        banks.append(record)
    snapshots = []
    for path in sorted((root / "teacher_snapshots").glob("*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True)
        domain, record = state["target"], state["prediction"]
        snapshots.append({"source": state["source"], "target": domain, "variant": state["variant"],
                          "seed": state["seed"], "step": state["step"], "event": state["event"],
                          "teacher_version": state["snapshot"]["version"],
                          **quality(record["prediction"], record["confidence"], truths[domain], len(names), threshold)})
    directory = root / "offline_analysis"
    directory.mkdir(exist_ok=True)
    write_json(directory / "pseudo_label_quality.json", points)
    write_json(directory / "bank_quality.json", banks)
    write_json(directory / "admitted_teacher_quality.json", snapshots)
    write_json(directory / "protocol.json", {"uses_target_labels": True, "training_feedback": False,
                                             "oracle_bank_used_for_training": False,
                                             "classnames": names, "threshold": threshold})
    return {"pseudo_label_records": len(points), "bank_records": len(banks), "teacher_records": len(snapshots)}
