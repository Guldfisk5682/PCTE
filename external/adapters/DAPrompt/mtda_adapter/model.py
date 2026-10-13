"""Upstream DAPL shared context + four domain vectors; own-domain inference."""
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from mtda.competitors.upstream import load_prompt_classes, encode_prompts


class DAPromptMTDA(nn.Module):
    def __init__(self, clip_model, clip_module, names, source, targets, threshold=.6):
        super().__init__()
        self.clip = clip_model.eval().requires_grad_(False)
        self.source, self.targets = source, tuple(targets)
        self.domains = (source, *targets)
        self.threshold = threshold
        self.classes = len(names)
        cfg = SimpleNamespace(INPUT=SimpleNamespace(SIZE=[clip_model.visual.input_resolution]),
            DATASET=SimpleNamespace(SOURCE_DOMAINS=[source.lower().replace(" ", "_")],
                                   TARGET_DOMAINS=[d.lower().replace(" ", "_") for d in targets]),
            TRAINER=SimpleNamespace(DAPL=SimpleNamespace(N_CTX=16,N_DMX=16,CSC=False)))
        classes = load_prompt_classes(Path(__file__).resolve().parents[1]/"trainers/dapl.py", ["PromptLearner"], clip_module)
        self.prompt_learner = classes["PromptLearner"](cfg,names,clip_model)
        for name in ("naive_embedding","tokenized_prompts"):
            value = getattr(self.prompt_learner,name)
            delattr(self.prompt_learner,name)
            self.prompt_learner.register_buffer(name,value)
        with torch.no_grad():
            tokens = self.prompt_learner.tokenized_prompts[-self.classes:]
            self.register_buffer("base_features",F.normalize(clip_model.encode_text(tokens).float(),dim=-1))

    def train(self,mode=True):
        super().train(mode)
        self.clip.eval()
        return self

    @property
    def scale(self):
        return self.clip.logit_scale.exp().detach()

    def text_features(self):
        prompts = self.prompt_learner()[:len(self.domains)*self.classes]
        tokens = self.prompt_learner.tokenized_prompts[:len(prompts)]
        return encode_prompts(self.clip,prompts,tokens).reshape(len(self.domains),self.classes,-1)

    def loss(self,source_batch,target_batches):
        raw,labels = source_batch
        texts = self.text_features()
        source_logits = self.scale * F.normalize(raw,dim=-1) @ texts[0].T
        source_loss = F.cross_entropy(source_logits,labels)
        target_losses, accepted = {}, {}
        for domain,features in target_batches.items():
            if not isinstance(features,torch.Tensor):
                raise TypeError("target training accepts features only, never ground truth")
            z = F.normalize(features,dim=-1)
            with torch.no_grad():
                probabilities = (self.scale*z@self.base_features.T).softmax(-1)
                confidence,pseudo = probabilities.max(-1)
                mask = confidence.ge(self.threshold)
            index = self.domains.index(domain)
            logits = self.scale*z@texts[index].T
            per_sample = F.cross_entropy(logits,pseudo,reduction="none")
            # Upstream divides by mask.sum(): preserve accepted-sample averaging,
            # but avoid its NaN when no sample meets the fixed threshold.
            loss = (per_sample*mask).sum()/mask.sum().clamp_min(1)
            target_losses[domain] = loss
            accepted[domain] = int(mask.sum())
        total = source_loss + torch.stack(list(target_losses.values())).mean()
        return total, {"source_ce":float(source_loss.detach()),
                       "target_ce":{d:float(v.detach()) for d,v in target_losses.items()},
                       "accepted_samples":accepted}

    @torch.no_grad()
    def evaluation_text(self,domain):
        return self.text_features()[self.domains.index(domain)]
