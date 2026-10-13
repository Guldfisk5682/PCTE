import copy
from types import SimpleNamespace

import pytest
import torch

from test_mtda import make_model
from test_b_reliability import cache, snapshot
from mtda.reliability.bank import TeacherRegistry, build_from_predictions
from mtda.reliability.teacher import ensemble_spl
from mtda.source_bank import SourceCentroidBank
from mtda.reliability_followup.references import validate_reference, METHODS
from scripts.agreement_metrics import error_partitions


def test_beta_zero_matches_uniform_when_all_centroids_valid_and_keeps_invalid_mask():
    raw = cache().z_pre
    text = torch.eye(3, 5)
    source = SourceCentroidBank(3, 5)
    source.update(raw, torch.arange(3))
    registry = TeacherRegistry(("t1", "t2", "t3"), 3, 5)
    bank = build_from_predictions(cache("t2"), torch.arange(3), torch.ones(3), 3, "t2:1", 1)
    registry.commit(snapshot("t2", text.flip(0)), bank)
    uniform, _ = ensemble_spl(raw, text, text, torch.tensor(3.), source, registry, "t1", "uniform")
    zero, info = ensemble_spl(raw, text, text, torch.tensor(3.), source, registry, "t1", "distance", 0.)
    torch.testing.assert_close(zero, uniform, atol=0, rtol=0)
    assert torch.all(info["weights"] == .5)
    bank["counts"][1] = 0
    bank["valid_mask"][1] = False
    registry.commit(snapshot("t2", text.flip(0)), bank)
    zero, info = ensemble_spl(raw, text, text, torch.tensor(3.), source, registry, "t1", "distance", 0.)
    assert torch.isfinite(zero).all() and torch.all(info["weights"][:, 1, 1] == 0)
    assert torch.all(info["weights"][:, 0, 1] == 1)


def test_error_taxonomy_separates_corrections_damage_and_unanimous_wrong():
    truth = torch.tensor([0, 0, 0, 0, 0])
    base, cross = torch.tensor([0, 1, 0, 1, 1]), torch.tensor([0, 0, 1, 1, 2])
    result = error_partitions(base, cross, truth, torch.ones(5), {"t2": torch.tensor([0, 0, 1, 1, 2])})
    assert result["corrected_count"] == result["damaged_count"] == 1
    assert result["net_accuracy_gain"] == 0
    assert result["same_wrong_rate"] == pytest.approx(.2)
    assert result["different_wrong_rate"] == pytest.approx(.2)
    assert result["unanimous_wrong_rate"] == pytest.approx(.2)
    assert result["other_teachers"]["t2"]["corrected_support_fraction"] == 1


def test_unanimous_wrong_votes_can_still_be_corrected_by_similarity_fusion():
    # Argmax agreement is not equivalent to the actual similarity-level ensemble.
    result = error_partitions(torch.tensor([1]),torch.tensor([0]),torch.tensor([0]),
                              torch.tensor([.9]),{"t2":torch.tensor([1]),"t3":torch.tensor([1])})
    assert result["unanimous_wrong_rate"] == 1
    assert result["unanimous_and_cross_same_wrong_rate"] == 0
    assert result["unanimous_but_cross_correct_rate"] == 1


def test_reference_rejects_seed_budget_ground_truth_and_cache_mismatch():
    model = make_model("B")
    args = SimpleNamespace(seed=42, steps=12, batch_size=4, m1=2, m2=2,
                           learning_rate=.003, lambda_target=.5, admission_fractions=[.25, .5, .75])
    caches = {d: cache(d) for d in model.prompts.domains}
    order = {"order": list(model.prompts.targets)}
    cfg = {"variant": "R4", "source": "s", "seed": 42, "steps": 12, "batch_size": 4,
           "m1": 2, "m2": 2, "learning_rate": .003, "lambda_target": .5, "threshold": .7,
           "checkpoint_sha256": "sha", "domain_order": order, "admission_fractions": [.25, .5, .75],
           "classnames": list(model.prompts.classnames), "target_training_ground_truth": False,
           "checkpoint_selection": "fixed final budget",
           "domain_cache_fingerprints": {d: c.fingerprint for d, c in caches.items()}}
    validate_reference(cfg, args, "s", model, caches, order, "sha")
    for key, value in (("seed", 43), ("steps", 13), ("target_training_ground_truth", True),
                       ("domain_cache_fingerprints", {})):
        changed = copy.deepcopy(cfg)
        changed[key] = value
        with pytest.raises(ValueError, match="reference"):
            validate_reference(changed, args, "s", model, caches, order, "sha")
    assert METHODS["FStartRef"]["snapshot"] == "reference"
    assert METHODS["FStartInit"]["snapshot"] == "online"
