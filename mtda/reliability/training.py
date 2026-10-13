"""Independent B prompt optimizers with matched effective update schedules."""

import torch
from torch.nn import functional as F

from mtda.spl import soft_cross_entropy
from .teacher import ensemble_spl, weight_diagnostics
from .diagnostics import prediction_statistics


class BranchOptimizers:
    def __init__(self, model, updates, learning_rate=0.003, weight_decay=0.01):
        if model.prompts.variant != "B":
            raise ValueError("cross-target reliability experiments require independent B prompts")
        self.optimizers, self.schedulers = {}, {}
        for domain in model.prompts.domains:
            params = [model.prompts.class_context(domain),
                      model.prompts.domain_contexts[model.prompts.domain_keys[domain]]]
            optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
            self.optimizers[domain] = optimizer
            self.schedulers[domain] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates)
        self.updates = {d: 0 for d in model.prompts.domains}
        self.samples = {d: 0 for d in model.prompts.domains}
        self.step_calls = 0

    def zero_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

    def step(self, domains, sample_counts):
        for domain in domains:
            self.optimizers[domain].step()
            self.schedulers[domain].step()
            self.updates[domain] += 1
            self.samples[domain] += sample_counts[domain]
            self.step_calls += 1

    def state_dict(self):
        return {"optimizers": {d: o.state_dict() for d, o in self.optimizers.items()},
                "schedulers": {d: s.state_dict() for d, s in self.schedulers.items()},
                "updates": dict(self.updates), "samples": dict(self.samples), "step_calls": self.step_calls}

    def load_state_dict(self, state):
        for domain, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][domain])
            self.schedulers[domain].load_state_dict(state["schedulers"][domain])
        self.updates, self.samples = dict(state["updates"]), dict(state["samples"])
        self.step_calls = state["step_calls"]


def controlled_step(model, source_bank, registry, optimizers, source_batch, target_batches,
                    mode="base", beta=10.0, lambda_target=0.5, collect_diagnostics=False):
    if not target_batches or not set(target_batches) <= set(model.prompts.targets):
        raise ValueError("active target batches must be a nonempty subset of configured targets")
    model.train()
    optimizers.zero_grad()
    device = model.scale.device
    updated_domains, sample_counts = [], {}
    source_loss = None
    if source_batch is not None:
        raw, labels = (x.to(device) for x in source_batch)
        source_bank.update(raw, labels)
        source_text = model.forward_source_text()
        source_loss = F.cross_entropy(model.scale * F.normalize(raw, dim=-1) @ source_text.T, labels)
        if not torch.isfinite(source_loss):
            raise ValueError("nonfinite Source CE")
        source_loss.backward()
        teacher_source = source_text.detach()
        updated_domains.append(model.prompts.source)
        sample_counts[model.prompts.source] = len(raw)
    else:
        with torch.no_grad():
            teacher_source = model.forward_source_text()
    target_losses, diagnostics = {}, {}
    for domain, raw in target_batches.items():
        if not isinstance(raw, torch.Tensor):
            raise TypeError("target training accepts visual features only, never labels")
        raw = raw.to(device)
        probabilities, contribution = ensemble_spl(raw, model.base_features, teacher_source, model.scale,
                                                   source_bank, registry, domain, mode, beta)
        text = model.forward_target_text(domain)
        logits = model.scale * F.normalize(raw, dim=-1) @ text.T
        loss = soft_cross_entropy(logits, probabilities)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite Target SPL: {domain}")
        # Even a sequential stage retains B's lambda / all-three-domains gradient scale.
        (lambda_target * loss / len(model.prompts.targets)).backward()
        target_losses[domain] = float(loss.detach())
        updated_domains.append(domain)
        sample_counts[domain] = len(raw)
        if collect_diagnostics:
            base_probabilities, _ = ensemble_spl(raw, model.base_features, teacher_source, model.scale,
                                                 source_bank, registry, domain, "base", beta)
            diagnostics[domain] = {**prediction_statistics(probabilities.detach().cpu()),
                                   "teacher_student_agreement": float((probabilities.argmax(-1) == logits.detach().argmax(-1)).float().mean()),
                                   "base_cross_teacher_agreement": float((probabilities.argmax(-1) == base_probabilities.argmax(-1)).float().mean()),
                                   "contribution": weight_diagnostics(contribution)}
    for parameter in model.prompts.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError("nonfinite prompt gradient")
    if any(parameter.grad is not None for parameter in model.clip.parameters()):
        raise AssertionError("CLIP must stay frozen")
    optimizers.step(updated_domains, sample_counts)
    source_value = float(source_loss.detach()) if source_loss is not None else None
    return {"source_ce": source_value, "target_spl": target_losses,
            "total_loss": (source_value or 0) + lambda_target * sum(target_losses.values()) / len(model.prompts.targets),
            "diagnostics": diagnostics}


def freeze_target(model, domain):
    model.prompts.class_context(domain).requires_grad_(False)
    model.prompts.domain_contexts[model.prompts.domain_keys[domain]].requires_grad_(False)
