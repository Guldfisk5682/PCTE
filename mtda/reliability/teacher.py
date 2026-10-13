"""Similarity-level SPL fusion, with weights normalized only across domains."""

import torch
from torch.nn import functional as F

from mtda.spl import compute_spl_soft_targets


@torch.no_grad()
def ensemble_spl(raw, base, source_text, scale, source_bank, registry, current,
                 mode="base", beta=10.0):
    if mode not in ("base", "uniform", "distance") or beta < 0:
        raise ValueError("invalid ensemble mode or distance scale")
    other_domains = registry.references(current) if mode != "base" else []
    domains = ["Source", *other_domains]
    z = F.normalize(raw.float(), dim=-1)
    if not other_domains:
        # Preserve B's matrix-multiplication path bit for bit before any teacher is admitted.
        probabilities = compute_spl_soft_targets(z, base, source_text, scale)
        weights = torch.ones(len(raw), 1, len(base), device=raw.device)
        return probabilities, {"domains": domains, "weights": weights, "distances": None,
                               "valid": None, "beta": beta}
    texts = torch.stack([source_text.detach(), *[registry.snapshots[d].text for d in other_domains]])
    similarities = torch.einsum("bd,nkd->bnk", z, texts)
    distances, valid = None, None
    if mode != "distance" or not other_domains:
        weights = torch.full_like(similarities, 1 / len(domains))
    else:
        indices = [registry.bank.targets.index(d) for d in other_domains]
        for domain in other_domains:
            if registry.bank.teacher_version[domain] != registry.snapshots[domain].version:
                raise ValueError("distance teacher bank is missing or stale")
        centroids = torch.cat((source_bank.running_means.unsqueeze(0), registry.bank.centroids[indices]))
        valid = torch.cat(((source_bank.running_count > 0).unsqueeze(0), registry.bank.valid_mask[indices]))
        # Never compute a distance to an invalid zero-initialized centroid.
        selected = centroids.flatten(0, 1)[valid.flatten()]
        computed = (raw.float().square().sum(-1)[:, None] + selected.square().sum(-1)[None]
                    - 2 * raw.float() @ selected.T).clamp_min(0)
        distances = torch.full_like(similarities, torch.inf)
        distances.flatten(1)[:, valid.flatten()] = computed
        inputs = (-beta * distances).masked_fill(~valid[None], -torch.inf)
        # An unseen Source class always falls back to Source-only, even if a target predicts that class.
        fallback = ~valid[0]
        inputs[:, :, fallback] = -torch.inf
        inputs[:, 0, fallback] = 0
        weights = inputs.softmax(dim=1)
    fused = 0.5 * (z @ base.detach().T) + 0.5 * (weights * similarities).sum(dim=1)
    probabilities = (scale.detach() * fused).softmax(dim=-1).detach()
    return probabilities, {"domains": domains, "weights": weights.detach(),
                           "distances": distances, "valid": valid, "beta": beta}


def weight_diagnostics(info):
    weights, domains = info["weights"], info["domains"]
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(1)
    result = {"teachers": domains, "teacher_count": len(domains),
              "mean_weights": dict(zip(domains, weights.mean((0, 2)).cpu().tolist())),
              "min_weights": dict(zip(domains, weights.amin((0, 2)).cpu().tolist())),
              "max_weights": dict(zip(domains, weights.amax((0, 2)).cpu().tolist())),
              "weight_entropy": float(entropy.mean()),
              "near_one_hot_sample_fraction": float((weights.amax(1).gt(0.99).float().mean(1).gt(0.99)).float().mean()),
              "near_one_hot_class_sample_fraction": float(weights.amax(1).gt(0.99).float().mean()),
              "dominant_teacher_by_class": [domains[i] for i in weights.mean(0).argmax(0).cpu().tolist()]}
    if info["valid"] is None:
        effective_count = torch.full((weights.shape[-1],), len(domains), device=weights.device)
    else:
        effective_count = info["valid"].sum(0).clone()
        effective_count[~info["valid"][0]] = 1
    multiple = effective_count > 1
    result["effective_teacher_count_by_class"] = effective_count.cpu().tolist()
    result["multi_valid_class_fraction"] = float(multiple.float().mean())
    result["one_hot_multi_valid_class_sample_fraction"] = float(weights.amax(1)[:, multiple].gt(.99).float().mean()) if multiple.any() else None
    result["weight_entropy_multi_valid"] = float(entropy[:, multiple].mean()) if multiple.any() else None
    if info["distances"] is not None:
        valid = info["valid"]
        result["source_only_class_fraction"] = float((~valid[1:].any(0)).float().mean())
        result["source_unseen_fallback_class_fraction"] = float((~valid[0]).float().mean())
        result["invalid_domain_classes"] = int((~valid).sum())
        result["squared_distance"] = {}
        result["softmax_distance_input"] = {}
        for index, domain in enumerate(domains):
            values = info["distances"][:, index, valid[index]].flatten()
            result["squared_distance"][domain] = distribution(values)
            result["softmax_distance_input"][domain] = distribution(-info["beta"] * values)
    return result


def distribution(values):
    values = values.detach().float().cpu()
    if not len(values):
        return None
    q = torch.quantile(values, torch.tensor([0., 0.05, 0.5, 0.95, 1.]))
    return {"min": float(q[0]), "p05": float(q[1]), "median": float(q[2]),
            "p95": float(q[3]), "max": float(q[4]), "mean": float(values.mean())}
