"""Reuse audited upstream prompt classes without importing their training framework."""

import ast
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def load_prompt_classes(path, names, clip_module):
    # Upstream hardcodes a default GPU for a fixed embedding. Select the actual
    # embedding device instead; CPU construction and cuda:1 remain supported.
    source = Path(path).read_text().replace('torch.device("cuda")', 'clip_model.token_embedding.weight.device')
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError("upstream prompt classes differ from audited source")
    for node in nodes:
        for function in node.body:
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # The controlled benchmark uses FP32, matching B/R4. Do not
                # accidentally re-enable AMP via upstream's unconditional decorators.
                function.decorator_list = []
    namespace = {"torch": torch, "nn": nn, "F": F, "clip": clip_module,
                 "_tokenizer": clip_module._tokenizer, "device_cuda": "cpu"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def encode_prompts(clip_model, prompts, tokens, trim_causal_padding=False):
    length = int(tokens.argmax(-1).max()) + 1 if trim_causal_padding else prompts.shape[1]
    x = prompts[:, :length] + clip_model.positional_embedding[:length].to(prompts.dtype)
    blocks = list(clip_model.transformer.resblocks) if trim_causal_padding else []
    masks = [block.attn_mask for block in blocks]
    try:
        for block, mask in zip(blocks, masks):
            block.attn_mask = mask[:length, :length] if mask is not None else None
        x = clip_model.transformer(x.permute(1, 0, 2))
    finally:
        for block, mask in zip(blocks, masks):
            block.attn_mask = mask
    if isinstance(x, tuple):
        x = x[0]
    x = clip_model.ln_final(x.permute(1, 0, 2))
    raw = x[torch.arange(len(x), device=x.device), tokens.to(x.device).argmax(-1)] @ clip_model.text_projection
    return F.normalize(raw.float(), dim=-1)
