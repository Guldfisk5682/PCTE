"""Independent labeled evaluation; no ground-truth values are returned to trainers."""

import torch
from torch.nn import functional as F
from mtda.data import SourceImages


@torch.no_grad()
def evaluate(model, domains, caches, names, batch_size=512, text_microbatch=4, limit=0):
    model.eval()
    device = model.scale.device
    results, predictions = {}, {}
    for domain in model.targets:
        cache = caches[domain]
        truth = SourceImages(domains[domain], None, names).labels
        if limit:
            truth = truth[:limit]
        outputs = []
        if cache["statistics"] is None:
            text = model.evaluation_text(domain)
            for raw in cache["features"][:len(truth)].split(batch_size):
                outputs.append((model.scale * F.normalize(raw.to(device), dim=-1) @ text.T).argmax(-1).cpu())
        else:
            for start in range(0, len(truth), text_microbatch):
                end=min(start+text_microbatch,len(truth))
                raw = cache["features"][start:end].to(device)
                stats = [x[start:end].to(device) for x in cache["statistics"]]
                outputs.append(model.logits(raw, stats, text_microbatch).argmax(-1).cpu())
        prediction = torch.cat(outputs)
        if len(prediction) != len(truth):
            raise ValueError("evaluation dropped a target batch")
        results[domain] = {"accuracy": float(prediction.eq(truth).double().mean()), "correct": int(prediction.eq(truth).sum()),
                           "samples": len(truth)}
        predictions[domain] = prediction
    return results, predictions
