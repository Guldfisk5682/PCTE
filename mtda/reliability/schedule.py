"""Label-free fixed domain ordering and explicit per-prompt update budgets."""

from dataclasses import dataclass
import math

from .bank import predict_cache

VARIANTS = ("R0M", "R1", "R2", "R3", "R4", "R5", "R6")
SEQUENTIAL = ("R1", "R2", "R3")
MODES = {"R0M": "base", "R1": "base", "R2": "uniform", "R3": "distance",
         "R4": "uniform", "R5": "distance", "R6": "distance"}


def difficulty_order(caches, base, source_text, scale, threshold=0.7, manual=None):
    # Fusing separately normalized prototype similarities equals this feature sum;
    # deliberately do not normalize the fused text features.
    teacher = 0.5 * base + 0.5 * source_text
    statistics = {}
    for domain, cache in caches.items():
        result = predict_cache(cache, teacher, scale)
        statistics[domain] = {"mean_entropy": float(result["entropy"].mean()),
                              "acceptance_rate": float(result["confidence"].ge(threshold).float().mean()),
                              "mean_max_probability": float(result["confidence"].mean()),
                              "samples": len(cache.z), "cache_fingerprint": cache.fingerprint}
    ordered = sorted(caches, key=lambda d: (statistics[d]["mean_entropy"],
                                            -statistics[d]["acceptance_rate"],
                                            -statistics[d]["mean_max_probability"], d))
    if manual is not None:
        if set(manual) != set(caches) or len(manual) != len(caches):
            raise ValueError("manual target order must contain every target exactly once")
        ordered = list(manual)
    return {"order": ordered, "statistics": statistics,
            "rule": "manual fixed order" if manual else "entropy ascending; ties: acceptance/max-confidence descending, name",
            "uses_target_labels": False, "confidence_threshold": threshold}


@dataclass
class BudgetSchedule:
    variant: str
    targets: tuple
    updates_per_target: int
    admission_fractions: tuple = (0.25, 0.5, 0.75)
    refresh_interval: int = 200

    def __post_init__(self):
        if self.variant not in VARIANTS or self.updates_per_target < 1 or not self.targets:
            raise ValueError("invalid schedule")
        if len(self.admission_fractions) != len(self.targets):
            raise ValueError("one admission milestone is required per target")
        if any(not 0 < f < 1 for f in self.admission_fractions) or list(self.admission_fractions) != sorted(set(self.admission_fractions)):
            raise ValueError("admission fractions must be distinct, ascending, and between zero and one")
        if self.refresh_interval < 1:
            raise ValueError("refresh interval must be positive")

    @property
    def sequential(self):
        return self.variant in SEQUENTIAL

    @property
    def total_steps(self):
        return self.updates_per_target * (len(self.targets) if self.sequential else 1)

    @property
    def admission_steps(self):
        return tuple(math.ceil(f * self.updates_per_target) for f in self.admission_fractions)

    def stage(self, step):
        return (step - 1) // self.updates_per_target if self.sequential else 0

    def active_targets(self, step):
        return (self.targets[self.stage(step)],) if self.sequential else self.targets

    def update_source(self, step):
        return not self.sequential or (step - 1) % len(self.targets) == 0

    def stage_end(self, step):
        return self.sequential and step % self.updates_per_target == 0

    def admissions(self, step):
        if self.variant in ("R2", "R3") and self.stage_end(step):
            return (self.targets[self.stage(step)],)
        if self.variant in ("R4", "R5", "R6"):
            return tuple(d for d, milestone in zip(self.targets, self.admission_steps) if step == milestone)
        return ()
