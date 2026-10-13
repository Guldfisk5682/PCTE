"""Ground-truth target labels exist only inside final evaluation."""

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import SourceImages


@torch.no_grad()
def accuracy(features, labels, text_features, scale, batch_size=128):
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size,
                        shuffle=False, drop_last=False)
    correct, count = 0, 0
    for images, truth in loader:
        images = F.normalize(images.to(text_features.device), dim=-1)
        predictions = (scale * images @ text_features.T).argmax(-1).cpu()
        correct += int((predictions == truth).sum())
        count += len(truth)
    if count == 0:
        raise ValueError("evaluation requires at least one sample")
    return {"accuracy": correct / count, "correct": correct, "samples": count}


@torch.no_grad()
def evaluate_targets(model, domains, features, classnames, preprocess, batch_size=128):
    model.eval()
    results = {}
    for target in model.prompts.targets:
        # Instantiate the labeled target dataset exclusively in this module.
        evaluation_data = SourceImages(domains[target], preprocess, classnames)
        if len(evaluation_data) != len(features[target]):
            raise ValueError("evaluation and cached image ordering differ")
        current_text = model.forward_target_text(target)
        results[target] = accuracy(features[target], evaluation_data.labels, current_text, model.scale, batch_size)
        results[target]["zero_shot"] = accuracy(features[target], evaluation_data.labels,
                                              model.base_features, model.scale, batch_size)["accuracy"]
    return results
