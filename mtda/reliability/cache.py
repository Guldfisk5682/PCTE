"""Stable, label-free views of deterministic frozen visual features."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from mtda.data import image_feature_cache, transform_signature


@dataclass
class VisualCache:
    domain: str
    sample_ids: tuple
    z_pre: torch.Tensor
    z: torch.Tensor
    fingerprint: str
    metadata: dict

    def __post_init__(self):
        if self.z_pre.ndim != 2 or self.z.shape != self.z_pre.shape:
            raise ValueError("cache must contain [samples, visual_dim] raw and normalized features")
        if len(self.sample_ids) != len(self.z_pre) or not torch.isfinite(self.z_pre).all():
            raise ValueError("invalid cache sample IDs or visual features")
        if self.z_pre.requires_grad or self.z.requires_grad:
            raise ValueError("frozen visual caches cannot require gradients")


def target_cache(dataset, domain, clip_model, cache_dir, checkpoint_sha,
                 batch_size=128, workers=4, backbone="RN50"):
    raw, upstream_key = image_feature_cache(dataset, clip_model, cache_dir, checkpoint_sha,
                                             batch_size, workers)
    root = Path(os.path.commonpath([str(p.parent) for p in dataset.paths]))
    identifiers = tuple(hashlib.sha256(p.relative_to(root).as_posix().encode()).hexdigest()
                        for p in dataset.paths)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sample IDs must be unique within a domain")
    manifest = hashlib.sha256("\n".join(identifiers).encode()).hexdigest()
    metadata = {"schema": 1, "domain": domain, "backbone": backbone,
                "checkpoint_sha256": checkpoint_sha, "preprocess": transform_signature(dataset.transform),
                "sample_manifest_sha256": manifest, "image_feature_fingerprint": upstream_key,
                "precision": "float32", "samples": len(raw)}
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / "domain_class" / f"{key}.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["metadata"] != metadata or tuple(state["sample_ids"]) != identifiers:
            raise ValueError("domain feature cache manifest mismatch")
        return VisualCache(domain, identifiers, state["z_pre"], state["z"], key, metadata)
    raw = raw.detach().float().cpu()
    normalized = F.normalize(raw, dim=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save({"metadata": metadata, "sample_ids": identifiers, "z_pre": raw, "z": normalized}, temporary)
    temporary.replace(path)
    return VisualCache(domain, identifiers, raw, normalized, key, metadata)
