"""Label-free target interfaces and checkpointable independent batch streams."""

import hashlib
import json
from pathlib import Path
import re

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff", ".webp"}
DOMAIN_ALIASES = {"art": "Art", "clipart": "Clipart", "product": "Product", "realworld": "Real World"}


def domain_identity(name):
    return name.casefold().replace("_", "").replace(" ", "").replace("-", "")


def discover_domains(root):
    root = Path(root).resolve()
    domains = {}
    for path in sorted(root.iterdir()):
        identity = domain_identity(path.name)
        if path.is_dir() and identity in DOMAIN_ALIASES:
            canonical = DOMAIN_ALIASES[identity]
            if canonical in domains:
                raise ValueError(f"duplicate Office-Home domain: {canonical}")
            domains[canonical] = path
    if set(domains) != set(DOMAIN_ALIASES.values()):
        raise ValueError(f"expected four Office-Home domains under {root}; found {list(domains)}")
    return domains


def resolve_domain(name, domains):
    canonical = DOMAIN_ALIASES.get(domain_identity(name))
    if canonical not in domains:
        raise ValueError(f"unknown domain {name!r}; choose from {list(domains)}")
    return canonical


def class_names(path):
    return sorted(p.name for p in Path(path).iterdir() if p.is_dir() and not p.name.startswith("."))


def validate_classes(domains, source, expected=65):
    names = class_names(domains[source])
    if expected is not None and len(names) != expected:
        raise ValueError(f"source has {len(names)} classes, expected {expected}")
    for domain, path in domains.items():
        if class_names(path) != names:
            raise ValueError(f"class vocabulary differs in {domain}")
    return names


class UnlabeledImages(Dataset):
    """Stores only paths: no ImageFolder, targets, labels, or inferred classes."""
    def __init__(self, root, transform):
        self.paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in EXTENSIONS)
        if not self.paths:
            raise ValueError(f"no images under {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class SourceImages(UnlabeledImages):
    def __init__(self, root, transform, classnames):
        super().__init__(root, transform)
        mapping = {name: i for i, name in enumerate(classnames)}
        self.labels = torch.tensor([mapping[p.relative_to(root).parts[0]] for p in self.paths])

    def __getitem__(self, index):
        return super().__getitem__(index), self.labels[index]


@torch.no_grad()
def image_feature_cache(dataset, clip_model, cache_dir, checkpoint_sha, batch_size=128, workers=4):
    """Cache deterministic frozen-image outputs without target labels."""
    metadata = {"checkpoint_sha256": checkpoint_sha, "precision": str(clip_model.dtype),
                "transform": transform_signature(dataset.transform),
                "files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in dataset.paths]}
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f"{fingerprint}.pt"
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if cache["fingerprint"] != fingerprint or len(cache["features"]) != len(dataset):
            raise ValueError(f"invalid feature cache: {path}")
        return cache["features"], fingerprint
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                        num_workers=workers, pin_memory=clip_model.logit_scale.is_cuda)
    rows = []
    for index, images in enumerate(loader):
        # Call only with UnlabeledImages, including when extracting source features.
        features = clip_model.encode_image(images.to(clip_model.logit_scale.device)).float().cpu()
        if not torch.isfinite(features).all():
            raise ValueError("nonfinite image features")
        rows.append(features)
        if index % 20 == 0:
            print(f"features: {min((index + 1) * batch_size, len(dataset))}/{len(dataset)}", flush=True)
    features = torch.cat(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{__import__('os').getpid()}.tmp")
    torch.save({"fingerprint": fingerprint, "features": features}, temporary)
    temporary.replace(path)
    return features, fingerprint


def transform_signature(transform):
    # Compose includes a process-dependent address for CLIP's RGB callable.
    return re.sub(r" at 0x[0-9a-fA-F]+", "", repr(transform))


class BatchStream:
    """Independent shuffled epochs; persist cursor, permutation, and generator."""
    def __init__(self, features, batch_size, seed, labels=None):
        if batch_size < 1 or len(features) < 1:
            raise ValueError("nonempty features and positive batch size required")
        self.features, self.labels, self.batch_size = features, labels, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(features), generator=self.generator)
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Match training drop_last, while allowing small synthetic datasets.
        size = min(self.batch_size, len(self.features))
        if self.cursor + size > len(self.order):
            raise StopIteration
        indices = self.order[self.cursor:self.cursor + size]
        self.cursor += size
        features = self.features[indices]
        return features if self.labels is None else (features, self.labels[indices])

    def next_batch(self):
        try:
            return next(self)
        except StopIteration:
            self.order = torch.randperm(len(self.features), generator=self.generator)
            self.cursor = 0
            return next(self)

    def state_dict(self):
        return {"order": self.order.clone(), "cursor": self.cursor,
                "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        if len(state["order"]) != len(self.features):
            raise ValueError("dataset length differs from checkpoint")
        self.order, self.cursor = state["order"].cpu(), state["cursor"]
        self.generator.set_state(state["generator"].cpu())
