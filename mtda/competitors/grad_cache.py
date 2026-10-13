"""Exact gradient recomputation for image-conditioned text prompts, without extra losses."""

import torch


def text_gradient_cache(contexts, encode, loss_function, microbatch):
    """Compute the full-batch loss, then recompute encoder gradients in bounded chunks.

    The frozen encoder has no dropout. Full-batch KL and batch-normalized entropy
    remain intact; splitting their losses into microbatches would change AD-CLIP.
    Caller applies returned context gradients to the small projector graph.
    """
    if microbatch < 1:
        raise ValueError("positive text microbatch required")
    with torch.no_grad():
        features = torch.cat([encode(c) for c in contexts.detach().split(microbatch)])
    leaf = features.detach().requires_grad_(True)
    loss, diagnostics = loss_function(leaf)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite full-batch AD-CLIP loss")
    feature_gradient, = torch.autograd.grad(loss, leaf)
    context_gradients = []
    for context, gradient in zip(contexts.detach().split(microbatch), feature_gradient.split(microbatch)):
        local = context.detach().requires_grad_(True)
        output = encode(local)
        grad, = torch.autograd.grad(output, local, gradient)
        context_gradients.append(grad.detach())
    return torch.cat(context_gradients), float(loss.detach()), diagnostics
