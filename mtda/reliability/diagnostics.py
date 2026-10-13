"""Unlabeled inference summaries and bank replacement diagnostics."""

import torch
from torch.nn import functional as F

from .teacher import ensemble_spl, weight_diagnostics


def prediction_statistics(probabilities, threshold=0.7):
    confidence, prediction = probabilities.max(-1)
    accepted = confidence.ge(threshold)
    classes = probabilities.shape[-1]
    counts = torch.bincount(prediction[accepted], minlength=classes)
    return {"mean_max_confidence": float(confidence.mean()),
            "mean_prediction_entropy": float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean()),
            "acceptance_rate": float(accepted.float().mean()), "accepted_samples": int(accepted.sum()),
            "accepted_classes": int((counts > 0).sum()), "classes": classes,
            "accepted_class_counts": counts.tolist(),
            "pseudo_label_class_counts": torch.bincount(prediction, minlength=classes).tolist()}


@torch.no_grad()
def full_inference(model, caches, source_bank, registry, mode, beta, threshold=0.7, batch_size=512):
    source_text = model.forward_source_text()
    summaries, bundle = {}, {}
    for domain, cache in caches.items():
        text = model.forward_target_text(domain)
        probabilities, students, basics, weights, distances = [], [], [], [], []
        last = None
        for raw in cache.z_pre.split(batch_size):
            raw = raw.to(model.scale.device)
            p, info = ensemble_spl(raw, model.base_features, source_text, model.scale, source_bank,
                                   registry, domain, mode, beta)
            basic, _ = ensemble_spl(raw, model.base_features, source_text, model.scale, source_bank,
                                    registry, domain, "base", beta)
            probabilities.append(p.cpu())
            students.append((model.scale * F.normalize(raw, dim=-1) @ text.T).argmax(-1).cpu())
            basics.append(basic.argmax(-1).cpu())
            weights.append(info["weights"].cpu())
            if info["distances"] is not None:
                distances.append(info["distances"].cpu())
            last = info
        p = torch.cat(probabilities)
        confidence, prediction = p.max(-1)
        student, basic = torch.cat(students), torch.cat(basics)
        combined = {**last, "weights": torch.cat(weights),
                    "distances": torch.cat(distances) if distances else None,
                    "valid": last["valid"].cpu() if last["valid"] is not None else None}
        summaries[domain] = {**prediction_statistics(p, threshold),
                             "teacher_student_agreement": float((prediction == student).float().mean()),
                             "base_cross_teacher_agreement": float((prediction == basic).float().mean()),
                             "contribution": weight_diagnostics(combined)}
        bundle[domain] = {"prediction": prediction, "confidence": confidence,
                          "entropy": -(p * p.clamp_min(1e-12).log()).sum(-1),
                          "student_prediction": student, "base_prediction": basic,
                          "sample_ids": cache.sample_ids, "cache_fingerprint": cache.fingerprint}
    return summaries, bundle


def bank_diagnostics(result, previous=None):
    counts, valid = result["counts"], result["valid_mask"]
    selected = counts[valid].float()
    diagnostic = {"bank_identity": "Target", "teacher_version": result["teacher_version"],
                  "step": result["last_refresh_step"], "coverage": float(valid.float().mean()),
                  "valid_classes": int(valid.sum()), "classes": len(counts), "class_counts": counts.tolist(),
                  "accepted_samples": int(counts.sum()), "unique_samples": result["unique_samples"],
                  "duplicates_skipped": result["duplicates_skipped"],
                  "count_min_all_classes": int(counts.min()), "count_max_all_classes": int(counts.max()),
                  "count_mean_all_classes": float(counts.float().mean()),
                  "count_median_all_classes": float(counts.float().median()),
                  "count_min_valid_classes": int(selected.min()) if len(selected) else None,
                  "bank_mean_confidence": float(result["confidence"][result["accepted"]].mean()) if result["accepted"].any() else None}
    if previous is not None:
        if previous["sample_ids"] != result["sample_ids"]:
            raise ValueError("refresh sample manifest must remain fixed")
        overlap = previous["valid_mask"] & valid
        cosine = F.cosine_similarity(previous["centroids"], result["centroids"], dim=-1)
        shifts = (previous["centroids"] - result["centroids"]).norm(dim=-1)
        union = previous["accepted"] | result["accepted"]
        intersection = previous["accepted"] & result["accepted"]
        diagnostic.update({"previous_teacher_version": previous["teacher_version"],
                           "previous_coverage": float(previous["valid_mask"].float().mean()),
                           "centroid_cosine_by_class": [float(v) if m else None for v, m in zip(cosine, overlap)],
                           "centroid_cosine_mean": float(cosine[overlap].mean()) if overlap.any() else None,
                           "centroid_shift_by_class": [float(v) if m else None for v, m in zip(shifts, overlap)],
                           "centroid_shift_mean": float(shifts[overlap].mean()) if overlap.any() else None,
                           "accepted_set_jaccard": float(intersection.sum() / union.sum()) if union.any() else 1.0,
                           "accepted_added": int((result["accepted"] & ~previous["accepted"]).sum()),
                           "accepted_removed": int((previous["accepted"] & ~result["accepted"]).sum()),
                           "pseudo_labels_changed": int((previous["prediction"] != result["prediction"]).sum()),
                           "refresh_teacher_prediction_agreement": float((previous["prediction"] == result["prediction"]).float().mean())})
    return diagnostic
