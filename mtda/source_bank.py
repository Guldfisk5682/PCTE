"""Non-trainable cumulative means of unnormalized source image features."""

import torch
from torch import nn


class SourceCentroidBank(nn.Module):
    def __init__(self, classes, feature_dim):
        super().__init__()
        self.register_buffer("running_means", torch.zeros(classes, feature_dim, dtype=torch.float32))
        self.register_buffer("running_count", torch.zeros(classes, dtype=torch.long))

    @torch.no_grad()
    def update(self, features, labels):
        features = features.detach().float()
        if not torch.isfinite(features).all():
            raise ValueError("source features must be finite")
        if labels.min() < 0 or labels.max() >= len(self.running_count):
            raise ValueError("source label is outside the class vocabulary")
        for label in labels.unique():
            values = features[labels == label]
            count = self.running_count[label]
            self.running_means[label] = (
                self.running_means[label] * count + values.sum(0)
            ) / (count + len(values))
            self.running_count[label] += len(values)

    def weights(self, features):
        # Softmax over one source is identically one, including unseen classes.
        return torch.ones(len(features), 1, len(self.running_count), device=features.device)

    def diagnostics(self):
        return {"observed_classes": int((self.running_count > 0).sum()),
                "classes": len(self.running_count), "samples": int(self.running_count.sum())}
