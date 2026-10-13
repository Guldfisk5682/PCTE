"""Offline error partitions and teacher support; never imported by trainers."""

import torch


def error_partitions(base, cross, truth, confidence, other_predictions=None, threshold=.7):
    base_ok, cross_ok = base.eq(truth), cross.eq(truth)
    corrected, damaged = ~base_ok & cross_ok, base_ok & ~cross_ok
    same_wrong = ~base_ok & ~cross_ok & base.eq(cross)
    different_wrong = ~base_ok & ~cross_ok & base.ne(cross)
    votes = torch.stack([base, *(other_predictions or {}).values()])
    unanimous = votes.eq(votes[0]).all(0)
    unanimous_wrong = unanimous & ~base_ok
    accepted = confidence.ge(threshold)
    count = len(truth)
    def rate(mask):
        return float(mask.float().mean())
    result = {"samples": count, "base_accuracy": rate(base_ok), "cross_accuracy": rate(cross_ok),
              "corrected_count": int(corrected.sum()), "damaged_count": int(damaged.sum()),
              "correction_rate": rate(corrected), "damage_rate": rate(damaged),
              "net_accuracy_gain": rate(cross_ok) - rate(base_ok),
              "both_correct_rate": rate(base_ok & cross_ok), "same_wrong_rate": rate(same_wrong),
              "different_wrong_rate": rate(different_wrong), "base_cross_agreement": rate(base.eq(cross)),
              "unanimous_vote_rate": rate(unanimous), "unanimous_wrong_rate": rate(unanimous_wrong),
              "unanimous_and_cross_same_wrong_rate": rate(unanimous_wrong & cross.eq(base)),
              "unanimous_but_cross_correct_rate": rate(unanimous_wrong & cross_ok),
              "accepted_unanimous_wrong_rate": rate(unanimous_wrong & accepted),
              "unanimous_vote_precision": float(base_ok[unanimous].float().mean()) if unanimous.any() else None,
              "corrected_supported_by_other_teacher_rate":
                  float(votes[1:].eq(truth).any(0)[corrected].float().mean()) if corrected.any() and len(votes) > 1 else None,
              "vote_scope": "original Base+Source SPL prediction plus each eligible other-target prompt prediction",
              "other_teachers": {}}
    assert int(corrected.sum()) - int(damaged.sum()) == int(cross_ok.sum()) - int(base_ok.sum())
    for domain, prediction in (other_predictions or {}).items():
        result["other_teachers"][domain] = {
            "prompt_accuracy": rate(prediction.eq(truth)), "agreement_with_base": rate(prediction.eq(base)),
            "agreement_with_cross": rate(prediction.eq(cross)),
            "corrected_support_count": int((corrected & prediction.eq(truth)).sum()),
            "damaged_wrong_vote_count": int((damaged & prediction.eq(cross)).sum()),
            "corrected_support_fraction": float(prediction[corrected].eq(truth[corrected]).float().mean()) if corrected.any() else None}
    return result
