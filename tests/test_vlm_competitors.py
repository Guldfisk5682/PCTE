from pathlib import Path
from types import SimpleNamespace
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mtda.competitors.experiment import load_adapter
from mtda.competitors.grad_cache import text_gradient_cache
from mtda.competitors.upstream import encode_prompts


def tokenize(prompts):
    if isinstance(prompts,str):
        prompts=[prompts]
    tokens=torch.zeros(len(prompts),77,dtype=torch.long)
    for row,prompt in enumerate(prompts):
        words=prompt.split()
        tokens[row,0]=1
        for i,word in enumerate(words):
            tokens[row,i+1]=2 if word=="X" else 3+(sum(word.encode())%100)
        tokens[row,len(words)+1]=127
    return tokens


TOKENIZER=SimpleNamespace(tokenize=tokenize,_tokenizer=SimpleNamespace(encode=lambda name:name.split()))


def upstream_adapter(repository,class_name):
    root=Path(__file__).resolve().parents[2]/repository
    if not (root/'mtda_adapter/model.py').is_file():
        pytest.skip(f'optional cloned competitor adapter is unavailable: {root}')
    return load_adapter(root,class_name)


class Block(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.linear=nn.Linear(width,width)
        self.attn_mask=torch.empty(77,77).fill_(float("-inf")).triu_(1)
    def forward(self,x):
        return self.linear(x).cumsum(0)


class Transformer(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.resblocks=nn.ModuleList([Block(width)])
    def forward(self,x):
        for block in self.resblocks:
            x=block(x)
        return x


class Clip(nn.Module):
    def __init__(self,width=8):
        super().__init__()
        self.visual=SimpleNamespace(input_resolution=224,output_dim=5)
        self.token_embedding=nn.Embedding(128,width)
        self.positional_embedding=nn.Parameter(torch.randn(77,width)*.01)
        self.transformer=Transformer(width)
        self.ln_final=nn.LayerNorm(width)
        self.text_projection=nn.Parameter(torch.randn(width,5)*.01)
        self.logit_scale=nn.Parameter(torch.tensor(1.))
    @property
    def dtype(self):
        return torch.float32
    def encode_text(self,tokens):
        return encode_prompts(self,self.token_embedding(tokens),tokens)


def dapl():
    torch.manual_seed(41)
    return upstream_adapter('DAPrompt','DAPromptMTDA')(
        Clip(),TOKENIZER,['one','two','three'],'s',('t1','t2','t3'),.6)


def test_dapl_shared_context_domain_rows_and_detached_teacher():
    model=dapl()
    assert model.prompt_learner.ctx.shape==(16,8)
    assert model.prompt_learner.domain_vectors.shape==(4,16,8)
    text=model.text_features()
    F.cross_entropy(model.scale*torch.randn(4,5)@text[0].T,torch.arange(4)%3).backward()
    assert model.prompt_learner.ctx.grad is not None
    assert torch.count_nonzero(model.prompt_learner.domain_vectors.grad[1:])==0
    assert all(p.grad is None for p in model.clip.parameters())
    assert not model.base_features.requires_grad


def test_dapl_empty_acceptance_is_finite_targets_cannot_supply_labels():
    model=dapl()
    model.threshold=1.
    batch=torch.randn(4,5),torch.arange(4)%3
    targets={d:torch.randn(4,5) for d in model.targets}
    loss,stats=model.loss(batch,targets)
    assert torch.isfinite(loss) and all(x==0 for x in stats['accepted_samples'].values())
    assert all(x==0 for x in stats['target_ce'].values())
    with pytest.raises(TypeError,match='never ground truth'):
        model.loss(batch,{'t1':(torch.randn(4,5),torch.arange(4))})


def test_causal_padding_trim_preserves_features_and_context_gradients():
    torch.manual_seed(10)
    clip=Clip().requires_grad_(False)
    tokens=tokenize(['X X one.','X X two.'])
    a=clip.token_embedding(tokens).detach().requires_grad_(True)
    b=a.detach().clone().requires_grad_(True)
    full=encode_prompts(clip,a,tokens)
    trimmed=encode_prompts(clip,b,tokens,True)
    torch.testing.assert_close(trimmed,full)
    full.sum().backward();trimmed.sum().backward()
    torch.testing.assert_close(a.grad,b.grad)
    assert clip.transformer.resblocks[0].attn_mask.shape==(77,77)


def test_gradient_cache_preserves_full_batch_entropy_kl_and_projector_gradients():
    torch.manual_seed(20)
    network=nn.Linear(5,7)
    projector=nn.Linear(3,5)
    features=torch.randn(12,3)
    def encode(x):
        return F.normalize(network(x).reshape(-1,1,7).expand(-1,3,-1),dim=-1)
    network.requires_grad_(False)
    def objective(text):
        source,target=text[:6],text[6:]
        p=(target.sum(-1)+torch.randn(1,3,generator=torch.Generator().manual_seed(1))).softmax(-1)
        normalized=F.normalize(p,dim=0)
        entropy=-(normalized*(normalized+1e-8).log()).sum(-1).mean()
        kl=F.kl_div(source.log_softmax(1),target.softmax(1),reduction='batchmean')
        return entropy+kl,{}
    direct=objective(encode(projector(features)))[0]
    direct.backward()
    expected=[p.grad.clone() for p in projector.parameters()]
    projector.zero_grad(set_to_none=True)
    contexts=projector(features)
    grads,value,_=text_gradient_cache(contexts,encode,objective,4)
    contexts.backward(grads)
    assert value==pytest.approx(float(direct.detach()))
    for actual,reference in zip(projector.parameters(),expected):
        torch.testing.assert_close(actual.grad,reference)


def test_adclip_cached_statistics_match_native_projectors_and_frozen_lists_checkpoint():
    torch.manual_seed(31)
    model=upstream_adapter('AD-CLIP','ADCLIPMTDA')(
        Clip(512),TOKENIZER,['one','two','three'],'s',('t1','t2','t3'),False)
    maps=[torch.randn(2,c,5,5) for c in (256,512,1024,2048)]
    native=model.prompt_learner
    stats=[torch.stack([native.domain_tokens.adain.mu(x),native.domain_tokens.adain.sigma(x),
                       F.adaptive_avg_pool2d(x,1).flatten(1)],dim=1) for x in maps]
    domain,content,mapping=model.projected_tokens(stats)
    torch.testing.assert_close(domain,native.domain_tokens(maps))
    torch.testing.assert_close(content,native.image_tokens(maps,4))
    torch.testing.assert_close(mapping,native.style_mapping_tokens(maps))
    assert all(not p.requires_grad for p in native.domain_tokens.linear1.parameters())
    assert 'domain_tokens.linear1.0.weight' in native.state_dict()
    saved=copy.deepcopy(native.state_dict())
    with torch.no_grad():
        native.domain_tokens.linear1[0].weight.add_(10)
    native.load_state_dict(saved)
    torch.testing.assert_close(model.projected_tokens(stats)[0],domain)


def test_adclip_vectorized_losses_match_native_single_target_reduction():
    torch.manual_seed(30)
    model=upstream_adapter('AD-CLIP','ADCLIPMTDA')(
        Clip(512),TOKENIZER,['one','two','three'],'s',('t1',),True)
    text=F.normalize(torch.randn(8,3,5),dim=-1).requires_grad_(True)
    source,target=text[:4],text[4:]
    raw,unlabeled=torch.randn(4,5),torch.randn(4,5)
    labels=torch.arange(4)%3
    actual,_=model.feature_loss(text,raw,labels,{'t1':unlabeled})
    source_logits=model.scale*torch.einsum('bd,bkd->bk',F.normalize(raw,dim=-1),source)
    target_probs=(model.scale*torch.einsum('bd,bkd->bk',F.normalize(unlabeled,dim=-1),target)).softmax(-1)
    norm=F.normalize(target_probs,dim=0)
    expected=F.cross_entropy(source_logits,labels)+F.kl_div(source.log_softmax(1),target.softmax(1),reduction='batchmean')
    expected=expected+.01*(-norm*(norm+1e-8).log()).sum(-1).mean()
    torch.testing.assert_close(actual,expected)


def test_conditional_evaluation_keeps_last_partial_batch(monkeypatch):
    from mtda.competitors import offline
    monkeypatch.setattr(offline,'SourceImages',lambda *args:SimpleNamespace(labels=torch.zeros(43,dtype=torch.long)))
    class Classifier:
        targets=('target',)
        scale=torch.tensor(1.)
        def eval(self):
            return self
        def logits(self,features,statistics,microbatch):
            assert len(features)==len(statistics[0])
            return torch.tensor([1.,0.]).expand(len(features),-1)
    cache={'target':{'features':torch.randn(43,5),'statistics':[torch.randn(43,3,4)]}}
    result,predictions=offline.evaluate(Classifier(),{'target':Path('/unused')},cache,['one','two'],text_microbatch=16,limit=37)
    assert result['target']['samples']==37 and len(predictions['target'])==37
    result,predictions=offline.evaluate(Classifier(),{'target':Path('/unused')},cache,['one','two'],text_microbatch=16)
    assert result['target']['samples']==43 and result['target']['accuracy']==1.
