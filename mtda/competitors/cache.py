"""Deterministic RN50 multiscale statistics, without labels or stored image tensors."""

import hashlib
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from mtda.data import image_feature_cache
from mtda.experiment import file_sha


def frozen_features(dataset, clip_model, cache_dir, checkpoint_sha, batch_size=128, workers=4, statistics=False):
    raw, key = image_feature_cache(dataset, clip_model, cache_dir, checkpoint_sha, batch_size, workers)
    if not statistics:
        return {"features": raw, "statistics": None, "fingerprint": key}
    metadata = {"backbone": "RN50", "image_feature_fingerprint": key, "precision": "float32",
                "statistics": "layer1..4 channel mean, native std(eps=1e-8), adaptive GAP",
                "implementation_sha256": file_sha(__file__), "samples": len(dataset), "extraction_batch_size":batch_size}
    stats_key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / "adclip_statistics" / f"{stats_key}.pt"
    if path.exists():
        state = torch.load(path, weights_only=True, map_location="cpu")
        if state["metadata"] != metadata or any(len(x) != len(raw) for x in state["statistics"]):
            raise ValueError("multiscale feature cache provenance mismatch")
        return {"features": raw, "statistics": state["statistics"], "fingerprint": stats_key}
    collected, current = [[] for _ in range(4)], {}
    def hook(index):
        def capture(module, inputs, output):
            mu = output.mean((2, 3))
            squared = (output - mu[:, :, None, None]).square().sum((2, 3))
            sigma = ((squared + 1e-8) / (output.shape[2] * output.shape[3])).sqrt()
            gap = F.adaptive_avg_pool2d(output, 1).flatten(1)
            current[index] = torch.stack([mu, sigma, gap], dim=1).float().cpu()
        return capture
    handles = [getattr(clip_model.visual, f"layer{i+1}").register_forward_hook(hook(i)) for i in range(4)]
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False, drop_last=False)
    cursor = 0
    try:
        with torch.no_grad():
            for images in loader:
                actual = clip_model.encode_image(images.to(clip_model.logit_scale.device)).float().cpu()
                torch.testing.assert_close(actual, raw[cursor:cursor+len(images)], atol=1e-5, rtol=1e-5)
                for i in range(4):
                    collected[i].append(current[i])
                cursor += len(images)
                if cursor % (20 * batch_size) < batch_size:
                    print(json.dumps({"event": "statistics_cache", "samples": cursor, "total": len(dataset)}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    statistics_rows = [torch.cat(rows) for rows in collected]
    if any(not torch.isfinite(x).all() for x in statistics_rows):
        raise ValueError("nonfinite multiscale statistics")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save({"metadata": metadata, "statistics": statistics_rows}, temporary)
    temporary.replace(path)
    return {"features": raw, "statistics": statistics_rows, "fingerprint": stats_key}
