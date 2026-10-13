"""One optimizer update per source batch and equal-weight target batches."""

import torch
from torch.nn import functional as F

from .spl import compute_spl_soft_targets, soft_cross_entropy


def train_step(model, bank, optimizer, source_batch, target_batches, lambda_target=0.5):
    if set(target_batches) != set(model.prompts.targets):
        raise ValueError("every configured target must supply exactly one batch")
    if lambda_target < 0:
        raise ValueError("lambda_target must be nonnegative")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = model.scale.device
    source_pre, source_labels = (tensor.to(device) for tensor in source_batch)
    source_features = F.normalize(source_pre, dim=-1)
    bank.update(source_pre, source_labels)
    source_text = model.forward_source_text()
    teacher_source = source_text.detach()
    source_loss = F.cross_entropy(model.scale * source_features @ source_text.T, source_labels)
    if not torch.isfinite(source_loss):
        raise ValueError("nonfinite source loss")
    source_loss.backward()
    losses, confidences = {}, []
    for domain, raw_features in target_batches.items():
        # Raw target features are the only target data accepted by this function.
        features = F.normalize(raw_features.to(device), dim=-1)
        probabilities = compute_spl_soft_targets(features, model.base_features, teacher_source, model.scale)
        text = model.forward_target_text(domain)
        loss = soft_cross_entropy(model.scale * features @ text.T, probabilities)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite target loss: {domain}")
        (lambda_target * loss / len(target_batches)).backward()
        losses[domain] = float(loss.detach())
        confidences.append(float(probabilities.max(-1).values.mean()))
    for parameter in model.prompts.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError("nonfinite prompt gradient")
    optimizer.step()
    mean_target = sum(losses.values()) / len(losses)
    return {"source_loss": float(source_loss.detach()), "target_loss": mean_target,
            "loss": float(source_loss.detach()) + lambda_target * mean_target,
            "confidence": sum(confidences) / len(confidences), "target_losses": losses,
            **bank.diagnostics()}
