"""Official RN50 conditional prompts and losses, shared across three target domains."""
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from mtda.competitors.upstream import load_prompt_classes, encode_prompts
from mtda.competitors.grad_cache import text_gradient_cache


class ADCLIPMTDA(nn.Module):
    def __init__(self,clip_model,clip_module,names,source,targets,registration_fix=False):
        super().__init__()
        self.clip = clip_model.eval().requires_grad_(False)
        self.source,self.targets = source,tuple(targets)
        self.classes = len(names)
        cfg = SimpleNamespace(INPUT=SimpleNamespace(SIZE=[clip_model.visual.input_resolution]))
        classes = load_prompt_classes(Path(__file__).resolve().parents[1]/"trainers/adclip_rn50.py",
            ["AdaIN","domain_projector","image_projector","style_mapping_projector","PromptLearner"],clip_module)
        self.prompt_learner = classes["PromptLearner"](cfg,names,clip_model)
        # Native Python lists do not enter named_parameters(), the optimizer, or
        # checkpoints. Register all tensors; retain their native fixed-random
        # behavior in Code mode, or explicitly enable the registration repair.
        for module,attribute in ((self.prompt_learner.domain_tokens,"linear1"),
                                 (self.prompt_learner.image_tokens,"linear"),
                                 (self.prompt_learner.style_mapping_tokens,"linear1")):
            layers = nn.ModuleList(getattr(module,attribute))
            layers.requires_grad_(registration_fix)
            setattr(module,attribute,layers)
        tokens = self.prompt_learner.tokenized_prompts
        delattr(self.prompt_learner,"tokenized_prompts")
        self.prompt_learner.register_buffer("tokenized_prompts",tokens)
        self.registration_fix = registration_fix

    def train(self,mode=True):
        super().train(mode)
        self.clip.eval()
        return self

    @property
    def scale(self):
        return self.clip.logit_scale.exp().detach()

    def projected_tokens(self,statistics):
        domain,image,mapping = [],[],[]
        for i,statistics_i in enumerate(statistics):
            mu,sigma,gap = statistics_i.unbind(1)
            d = self.prompt_learner.domain_tokens
            domain.append(d.linear2[i](d.linear1[i]((mu+sigma)/2)))
            image.append(self.prompt_learner.image_tokens.linear[i](gap))
            m = self.prompt_learner.style_mapping_tokens
            pooled = F.adaptive_avg_pool1d(torch.cat([mu,sigma],dim=1),mu.shape[-1])
            mapping.append(m.linear2[i](m.relu(m.linear1[i](pooled))))
        domain = torch.stack(domain,dim=1)
        image = torch.stack(image,dim=1)
        # Upstream repeats the SAME lin(4 -> 1) decoder four times.
        content = self.prompt_learner.image_tokens.lin(image.transpose(1,2)).transpose(1,2).expand(-1,4,-1)
        return domain,content,torch.stack(mapping,dim=1)

    def encode_contexts(self,contexts):
        batch = len(contexts)
        p = self.prompt_learner
        ctx = contexts[:,None].expand(-1,self.classes,-1,-1)
        prefix = p.token_prefix[None].expand(batch,-1,-1,-1)
        suffix = p.token_suffix[None].expand(batch,-1,-1,-1)
        prompts = torch.cat([prefix,ctx,suffix],dim=2).flatten(0,1)
        return encode_prompts(self.clip,prompts,p.tokenized_prompts.repeat(batch,1),trim_causal_padding=True).reshape(batch,self.classes,-1)

    def pair_contexts(self,source_stats,target_stats):
        source_domain,source_content,source_mapping = self.projected_tokens(source_stats)
        source_contexts,target_contexts = [],[]
        for stats in target_stats.values():
            target_domain,target_content,_ = self.projected_tokens(stats)
            source_contexts.append(torch.cat([source_domain,target_domain,source_content],dim=1))
            target_contexts.append(torch.cat([source_domain,target_domain,target_content],dim=1))
        contexts = torch.cat([*source_contexts,*target_contexts])
        # Deliberately preserve the official Source-input mapping loss. Do not
        # silently change it to target-input mapping to match a different equation.
        style_loss = F.mse_loss(source_domain,source_mapping)
        return contexts,style_loss

    def feature_loss(self,texts,source_features,labels,target_features):
        domains,batch = len(target_features),len(source_features)
        source_text,target_text = texts.reshape(2,domains,batch,self.classes,-1).unbind(0)
        source_z,target_z = F.normalize(source_features,dim=-1),F.normalize(torch.stack(list(target_features.values())),dim=-1)
        source_logits = self.scale*torch.einsum("bd,tbkd->tbk",source_z,source_text)
        target_logits = self.scale*torch.einsum("tbd,tbkd->tbk",target_z,target_text)
        ce = F.cross_entropy(source_logits.flatten(0,1),labels.repeat(domains))
        # Native KL normalizes over CLASSES, then sums over K,D and divides by B.
        kl = F.kl_div(F.log_softmax(source_text,dim=2),F.softmax(target_text,dim=2),reduction="sum")/(domains*batch)
        probabilities = target_logits.softmax(-1)
        # Native entropy applies L2 normalization over the batch axis first.
        normalized = F.normalize(probabilities,dim=1)
        entropy = -(normalized*(normalized+1e-8).log()).sum(-1).mean()
        total = ce+kl+.01*entropy
        return total,{"source_ce":float(ce.detach()),"kl":float(kl.detach()),"entropy":float(entropy.detach())}

    def backward_loss(self,source_batch,target_batches,source_stats,target_stats,microbatch=4):
        if any(not isinstance(v,torch.Tensor) for v in target_batches.values()):
            raise TypeError("target training accepts features only, never ground truth")
        contexts,style_loss = self.pair_contexts(source_stats,target_stats)
        gradients,value,diagnostics = text_gradient_cache(contexts,self.encode_contexts,
            lambda features:self.feature_loss(features,*source_batch,target_batches),microbatch)
        torch.autograd.backward([contexts,.1*style_loss],[gradients,torch.ones_like(style_loss)])
        diagnostics["style_mapping"] = float(style_loss.detach())
        diagnostics["target_loss"] = diagnostics["kl"]+.01*diagnostics["entropy"]
        return value+.1*float(style_loss.detach()),diagnostics

    @torch.no_grad()
    def logits(self,features,statistics,microbatch=4):
        domain,content,mapped_source = self.projected_tokens(statistics)
        # Target-only inference uses the paper's style-mapping substitute for
        # unavailable source tokens. No source or other-target images are read.
        contexts = torch.cat([mapped_source,domain,content],dim=1)
        texts = torch.cat([self.encode_contexts(c) for c in contexts.split(microbatch)])
        return self.scale*torch.einsum("bd,bkd->bk",F.normalize(features,dim=-1),texts)
