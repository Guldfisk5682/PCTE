"""Detached prompt snapshots and replacement-only pseudo-labeled target banks."""

from dataclasses import dataclass
import hashlib

import torch
from torch.nn import functional as F


@dataclass
class TeacherSnapshot:
    domain: str
    step: int
    version: str
    text: torch.Tensor
    class_context: torch.Tensor
    domain_context: torch.Tensor

    @classmethod
    @torch.no_grad()
    def capture(cls, model, domain, step):
        text = model.forward_target_text(domain).detach().clone()
        class_context = model.prompts.class_context(domain).detach().cpu().clone()
        domain_context = model.prompts.domain_contexts[model.prompts.domain_keys[domain]].detach().cpu().clone()
        digest = hashlib.sha256(class_context.numpy().tobytes() + domain_context.numpy().tobytes()).hexdigest()
        return cls(domain, step, f"{domain}:{step}:{digest}", text, class_context, domain_context)

    def state_dict(self):
        return {"domain": self.domain, "step": self.step, "version": self.version,
                "text": self.text.detach().cpu(), "class_context": self.class_context.clone(),
                "domain_context": self.domain_context.clone()}

    @classmethod
    def from_state(cls, state, device):
        return cls(state["domain"], state["step"], state["version"], state["text"].to(device),
                   state["class_context"].cpu(), state["domain_context"].cpu())


@torch.no_grad()
def predict_cache(cache, text, scale, batch_size=512):
    predictions, confidences, entropies = [], [], []
    for features in cache.z.split(batch_size):
        probabilities = (scale * features.to(text.device) @ text.T).softmax(-1)
        confidence, prediction = probabilities.max(-1)
        predictions.append(prediction.cpu())
        confidences.append(confidence.cpu())
        entropies.append(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).cpu())
    return {"prediction": torch.cat(predictions), "confidence": torch.cat(confidences),
            "entropy": torch.cat(entropies)}


@torch.no_grad()
def build_from_predictions(cache, prediction, confidence, classes, version, step, threshold=0.7):
    """One pass over unique sample IDs; every rebuild starts from zero FP32 sums."""
    if len(prediction) != len(cache.sample_ids) or len(confidence) != len(prediction):
        raise ValueError("bank prediction and cache lengths differ")
    if not torch.isfinite(confidence).all() or (prediction < 0).any() or (prediction >= classes).any():
        raise ValueError("invalid bank predictions")
    first, unique = [], set()
    for index, sample_id in enumerate(cache.sample_ids):
        if sample_id not in unique:
            first.append(index)
            unique.add(sample_id)
    unique_mask = torch.zeros(len(prediction), dtype=torch.bool)
    unique_mask[first] = True
    accepted = confidence.cpu().ge(threshold) & unique_mask
    counts = torch.bincount(prediction.cpu()[accepted], minlength=classes)
    sums = torch.zeros(classes, cache.z_pre.shape[1], dtype=torch.float32)
    sums.index_add_(0, prediction.cpu()[accepted], cache.z_pre.cpu()[accepted].float())
    valid = counts > 0
    centroids = sums / counts.clamp_min(1).unsqueeze(-1)
    return {"centroids": centroids, "counts": counts, "valid_mask": valid,
            "last_refresh_step": step, "teacher_version": version,
            "prediction": prediction.cpu().clone(), "confidence": confidence.cpu().clone(),
            "accepted": accepted, "sample_ids": cache.sample_ids,
            "cache_fingerprint": cache.fingerprint, "unique_samples": len(unique),
            "duplicates_skipped": len(prediction) - len(unique)}


@torch.no_grad()
def rebuild_target_bank(cache, snapshot, scale, classes, threshold=0.7, batch_size=512):
    prediction = predict_cache(cache, snapshot.text, scale, batch_size)
    return build_from_predictions(cache, prediction["prediction"], prediction["confidence"],
                                  classes, snapshot.version, snapshot.step, threshold)


class TargetDomainClassBank:
    def __init__(self, targets, classes, dimensions, device="cpu"):
        self.targets = tuple(targets)
        self.centroids = torch.zeros(len(targets), classes, dimensions, device=device, dtype=torch.float32)
        self.counts = torch.zeros(len(targets), classes, device=device, dtype=torch.long)
        self.valid_mask = torch.zeros(len(targets), classes, device=device, dtype=torch.bool)
        self.last_refresh_step = torch.full((len(targets),), -1, device=device, dtype=torch.long)
        self.teacher_version = {d: None for d in targets}
        self.details = {}

    def replace(self, domain, result):
        index = self.targets.index(domain)
        if result["centroids"].shape != self.centroids[index].shape:
            raise ValueError("target centroid shape mismatch")
        self.centroids[index].copy_(result["centroids"])
        self.counts[index].copy_(result["counts"])
        self.valid_mask[index].copy_(result["valid_mask"])
        self.last_refresh_step[index] = result["last_refresh_step"]
        self.teacher_version[domain] = result["teacher_version"]
        self.details[domain] = result

    def state_dict(self):
        return {"centroids": self.centroids.cpu(), "counts": self.counts.cpu(),
                "valid_mask": self.valid_mask.cpu(), "last_refresh_step": self.last_refresh_step.cpu(),
                "teacher_version": dict(self.teacher_version), "details": self.details}

    def load_state_dict(self, state):
        for key in ("centroids", "counts", "valid_mask", "last_refresh_step"):
            getattr(self, key).copy_(state[key])
        self.teacher_version = dict(state["teacher_version"])
        self.details = state["details"]


class TeacherRegistry:
    def __init__(self, targets, classes, dimensions, device="cpu"):
        self.snapshots = {}
        self.bank = TargetDomainClassBank(targets, classes, dimensions, device)

    def commit(self, snapshot, bank_result=None):
        # Build/inference finishes before this commit. No training or checkpoint runs during replacement.
        if bank_result is not None:
            if bank_result["teacher_version"] != snapshot.version:
                raise ValueError("bank and teacher snapshot must have the same version")
            self.bank.replace(snapshot.domain, bank_result)
        self.snapshots[snapshot.domain] = snapshot

    def references(self, current):
        return [d for d in self.bank.targets if d != current and d in self.snapshots]

    def state_dict(self):
        return {"snapshots": {d: s.state_dict() for d, s in self.snapshots.items()},
                "bank": self.bank.state_dict()}

    def load_state_dict(self, state, device):
        self.bank.load_state_dict(state["bank"])
        self.snapshots = {d: TeacherSnapshot.from_state(s, device) for d, s in state["snapshots"].items()}
        for domain, version in self.bank.teacher_version.items():
            if version is not None and self.snapshots[domain].version != version:
                raise ValueError("restored bank/snapshot versions differ")
