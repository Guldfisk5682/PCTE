"""Explicit prompt sharing and deterministic, domain-stable initialization."""

import hashlib

import torch
from torch import nn
from torch.nn import functional as F


def context_template(shape, seed, identity):
    digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") % (2**63))
    return torch.randn(shape, generator=generator) * 0.02


class MTDAPromptGenerator(nn.Module):
    def __init__(self, classnames, clip_model, tokenize, source, targets,
                 variant="A", seed=42, m1=16, m2=16):
        super().__init__()
        if variant not in ("A", "B", "C"):
            raise ValueError("variant must be A, B, or C")
        if not targets or source in targets or len(set(targets)) != len(targets):
            raise ValueError("source and nonempty unique targets must be disjoint")
        if variant == "C" and len(targets) != 1:
            raise ValueError("C requires exactly one independently initialized pair")
        if m1 < 1 or m2 < 1:
            raise ValueError("context lengths must be positive")
        self.source, self.targets, self.variant = source, tuple(targets), variant
        self.domains = (source, *targets)
        self.classnames = tuple(classnames)
        self.domain_keys = {domain: f"d{i}" for i, domain in enumerate(self.domains)}
        width = clip_model.ln_final.weight.numel()
        template = context_template((len(classnames), m1, width), seed, f"class:{source}")
        if variant == "B":
            self.class_contexts = nn.ParameterDict({
                self.domain_keys[d]: nn.Parameter(template.clone()) for d in self.domains
            })
        else:
            self.class_contexts = nn.ParameterDict({"shared": nn.Parameter(template.clone())})
        self.domain_contexts = nn.ParameterDict({
            self.domain_keys[d]: nn.Parameter(context_template(
                (m2, width), seed, f"domain:{d}:{'source' if d == source else 'target'}"
            )) for d in self.domains
        })
        prefix = " ".join(["X"] * (m1 + m2))
        tokens = tokenize([f"{prefix} {name.replace('_', ' ')}." for name in classnames])
        # EOT must follow every learned token and the entire fixed class name.
        if (tokens.argmax(-1) <= 1 + m1 + m2).any():
            raise ValueError("contexts leave no room for class names in CLIP token sequence")
        with torch.no_grad():
            embeddings = clip_model.token_embedding(tokens.to(clip_model.ln_final.weight.device)).float()
        self.register_buffer("tokenized_prompts", tokens)
        self.register_buffer("token_prefix", embeddings[:, :1].clone())
        self.register_buffer("token_suffix", embeddings[:, 1 + m1 + m2:].clone())

    def class_context(self, domain):
        key = self.domain_keys[domain] if self.variant == "B" else "shared"
        return self.class_contexts[key]

    def forward(self, domain):
        context = self.domain_contexts[self.domain_keys[domain]]
        return torch.cat((self.token_prefix, self.class_context(domain),
                          context.unsqueeze(0).expand(len(self.classnames), -1, -1),
                          self.token_suffix), dim=1)


class PromptModel(nn.Module):
    def __init__(self, clip_model, prompts, base_tokens):
        super().__init__()
        self.clip = clip_model.eval().requires_grad_(False)
        self.prompts = prompts
        with torch.no_grad():
            base = F.normalize(clip_model.encode_text(base_tokens), dim=-1)
        self.register_buffer("base_features", base)

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self

    @property
    def scale(self):
        return self.clip.logit_scale.exp().detach()

    def forward_text(self, domain):
        x = self.prompts(domain).to(self.clip.dtype)
        x = x + self.clip.positional_embedding.to(x.dtype)
        x = self.clip.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.clip.ln_final(x).to(self.clip.dtype)
        positions = self.prompts.tokenized_prompts.argmax(dim=-1)
        features = x[torch.arange(len(x), device=x.device), positions] @ self.clip.text_projection
        return F.normalize(features.float(), dim=-1)

    def forward_source_text(self):
        return self.forward_text(self.prompts.source)

    def forward_target_text(self, target_id):
        domain = self.prompts.targets[target_id] if isinstance(target_id, int) else target_id
        if domain not in self.prompts.targets:
            raise ValueError(f"not a target domain: {domain}")
        return self.forward_text(domain)
