"""Single-source source-enhanced pseudo-labels; no auxiliary objectives."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def compute_spl_soft_targets(image_features, base_features, source_features, scale):
    similarities = 0.5 * (image_features @ base_features.T) + 0.5 * (image_features @ source_features.T)
    return (scale * similarities).softmax(dim=-1).detach()


def soft_cross_entropy(logits, probabilities):
    return -(probabilities.detach() * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
