import copy

import pytest
import torch
from torch.nn import functional as F

from test_mtda import make_model
from mtda.data import BatchStream
from mtda.source_bank import SourceCentroidBank
from mtda.spl import compute_spl_soft_targets
from mtda.trainer import train_step
from mtda.reliability.bank import (TeacherSnapshot, TeacherRegistry, build_from_predictions,
                                    rebuild_target_bank)
from mtda.reliability.cache import VisualCache
from mtda.reliability.experiment import save_state, load_state
from mtda.reliability.schedule import BudgetSchedule, difficulty_order, VARIANTS
from mtda.reliability.teacher import ensemble_spl
from mtda.reliability.training import BranchOptimizers, controlled_step, freeze_target


def cache(domain="t1", duplicate=False):
    raw = torch.tensor([[2., 0., 0., 0., 0.], [0., 2., 0., 0., 0.], [4., 0., 0., 0., 0.]])
    ids = ("a", "b", "a" if duplicate else "c")
    return VisualCache(domain, ids, raw, F.normalize(raw, dim=-1), "fingerprint", {})


def snapshot(domain, text, step=1):
    return TeacherSnapshot(domain, step, f"{domain}:{step}", text.detach().clone(), torch.zeros(1), torch.zeros(1))


def test_rebuild_is_unique_threshold_inclusive_and_replacement_only():
    c = cache(duplicate=True)
    result = build_from_predictions(c, torch.tensor([0, 1, 0]), torch.tensor([0.7, 0.69, 0.99]), 3, "v1", 1)
    assert result["counts"].tolist() == [1, 0, 0]
    assert result["duplicates_skipped"] == 1
    torch.testing.assert_close(result["centroids"][0], c.z_pre[0])
    registry = TeacherRegistry(("t1", "t2", "t3"), 3, 5)
    registry.commit(snapshot("t1", torch.eye(3, 5)), {**result, "teacher_version": "t1:1"})
    updated = build_from_predictions(cache(), torch.tensor([1, 1, 1]), torch.tensor([.8, .9, .95]), 3, "t1:2", 2)
    registry.commit(snapshot("t1", torch.eye(3, 5), 2), updated)
    assert registry.bank.counts[0].tolist() == [0, 3, 0]
    assert registry.bank.valid_mask[0].tolist() == [False, True, False]
    assert registry.bank.last_refresh_step[0] == 2
    assert registry.bank.teacher_version["t1"] == registry.snapshots["t1"].version


@pytest.mark.parametrize("mode", ["base", "uniform", "distance"])
def test_unadmitted_teacher_exactly_reduces_to_original_spl(mode):
    raw = torch.randn(7, 5)
    base, source = F.normalize(torch.randn(3, 5), dim=-1), F.normalize(torch.randn(3, 5), dim=-1)
    scale = torch.tensor(3.)
    p, info = ensemble_spl(raw, base, source, scale, SourceCentroidBank(3, 5),
                           TeacherRegistry(("t1", "t2", "t3"), 3, 5), "t1", mode)
    original = compute_spl_soft_targets(F.normalize(raw, dim=-1), base, source, scale)
    torch.testing.assert_close(p, original, rtol=0, atol=0)
    assert info["domains"] == ["Source"] and torch.equal(info["weights"], torch.ones(7, 1, 3))


def test_uniform_teacher_formula_self_exclusion_and_detachment():
    raw = torch.randn(7, 5, requires_grad=True)
    base, source, other = [F.normalize(torch.randn(3, 5), dim=-1).requires_grad_() for _ in range(3)]
    registry = TeacherRegistry(("t1", "t2", "t3"), 3, 5)
    registry.commit(snapshot("t1", -other))
    registry.commit(snapshot("t2", other))
    p, info = ensemble_spl(raw, base, source, torch.tensor(3., requires_grad=True), SourceCentroidBank(3, 5), registry, "t1", "uniform")
    z = F.normalize(raw, dim=-1)
    expected = (3 * (0.5 * z @ base.T + 0.25 * (z @ source.T + z @ other.T))).softmax(-1)
    torch.testing.assert_close(p, expected)
    assert info["domains"] == ["Source", "t2"] and not p.requires_grad
    torch.testing.assert_close(info["weights"].sum(1), torch.ones(7, 3))
    assert torch.all(info["weights"] == 0.5)


def test_distance_masks_invalid_classes_normalizes_domain_axis_and_source_fallback():
    source_bank = SourceCentroidBank(3, 5)
    source_bank.update(torch.tensor([[2., 0., 0., 0., 0.], [0., 2., 0., 0., 0.]]), torch.tensor([0, 1]))
    registry = TeacherRegistry(("t1", "t2", "t3"), 3, 5)
    text = F.normalize(torch.randn(3, 5), dim=-1)
    result = build_from_predictions(cache("t2"), torch.tensor([0, 0, 2]), torch.ones(3), 3, "t2:1", 1)
    registry.commit(snapshot("t2", text), result)
    # The own bank is deliberately very attractive and must still be excluded.
    own = build_from_predictions(cache(), torch.tensor([0, 1, 2]), torch.ones(3), 3, "t1:1", 1)
    registry.commit(snapshot("t1", text), own)
    p, info = ensemble_spl(cache().z_pre, text, text, torch.tensor(3.), source_bank, registry, "t1", "distance", beta=.2)
    assert info["domains"] == ["Source", "t2"]
    weights = info["weights"]
    torch.testing.assert_close(weights.sum(1), torch.ones(3, 3))
    assert torch.all(weights[:, 1, 1] == 0)  # invalid t2 centroid never competes
    assert torch.isinf(info["distances"][:, 1, 1]).all()  # no raw distance to an invalid zero centroid
    assert torch.all(weights[:, 0, 2] == 1) and torch.all(weights[:, 1, 2] == 0)  # unseen source fallback
    expected = (-.2 * info["distances"][:, :, 0]).softmax(1)
    torch.testing.assert_close(weights[:, :, 0], expected)
    assert torch.isfinite(p).all()


def test_snapshot_is_immutable_and_refresh_changes_after_student_update():
    model = make_model("B")
    first = TeacherSnapshot.capture(model, "t1", 1)
    saved = first.text.clone()
    old_bank = rebuild_target_bank(cache(), first, model.scale, 3, threshold=.1)
    with torch.no_grad():
        model.prompts.class_context("t1").add_(torch.randn_like(model.prompts.class_context("t1")) * 10)
    second = TeacherSnapshot.capture(model, "t1", 2)
    new_bank = rebuild_target_bank(cache(), second, model.scale, 3, threshold=.1)
    assert torch.equal(first.text, saved) and first.version != second.version
    assert not torch.equal(first.text, second.text)
    assert not torch.equal(old_bank["confidence"], new_bank["confidence"])
    assert not first.text.requires_grad


@pytest.mark.parametrize("variant", VARIANTS)
def test_exact_effective_update_budgets_and_admission_order(variant):
    schedule = BudgetSchedule(variant, ("t3", "t1", "t2"), 12, refresh_interval=5)
    source_steps, updates, admissions = 0, {d: 0 for d in schedule.targets}, []
    for step in range(1, schedule.total_steps + 1):
        source_steps += schedule.update_source(step)
        for domain in schedule.active_targets(step):
            updates[domain] += 1
        admissions.extend(schedule.admissions(step))
    assert source_steps == 12 and set(updates.values()) == {12}
    assert schedule.total_steps == (36 if schedule.sequential else 12)
    assert admissions == ([] if variant in ("R0M", "R1") else list(schedule.targets))


def test_matched_joint_exactly_matches_original_b_adamw_parameters():
    original, matched = make_model("B"), make_model("B")
    bank, other_bank = SourceCentroidBank(3, 5), SourceCentroidBank(3, 5)
    optimizer = torch.optim.AdamW(original.prompts.parameters(), lr=.003, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    branches = BranchOptimizers(matched, 5)
    registry = TeacherRegistry(matched.prompts.targets, 3, 5)
    for _ in range(5):
        batch = torch.randn(6, 5), torch.arange(6) % 3
        targets = {d: torch.randn(6, 5) for d in original.prompts.targets}
        train_step(original, bank, optimizer, batch, targets)
        scheduler.step()
        controlled_step(matched, other_bank, registry, branches, batch, targets)
    for name, parameter in original.prompts.state_dict().items():
        torch.testing.assert_close(parameter, matched.prompts.state_dict()[name], rtol=0, atol=0)
    assert all(v == 5 for v in branches.updates.values()) and branches.step_calls == 20


def test_target_loss_never_trains_source_or_teacher_and_target_labels_rejected():
    model = make_model("B")
    registry = TeacherRegistry(model.prompts.targets, 3, 5)
    registry.commit(TeacherSnapshot.capture(model, "t2", 1))
    before = {n: p.clone() for n, p in model.prompts.named_parameters()}
    branches = BranchOptimizers(model, 5)
    bank = SourceCentroidBank(3, 5)
    controlled_step(model, bank, registry, branches, None, {"t1": torch.randn(5, 5)}, "uniform")
    for domain in ("s", "t2", "t3"):
        assert model.prompts.class_context(domain).grad is None
        for prefix in ("class_contexts", "domain_contexts"):
            key = f"{prefix}.{model.prompts.domain_keys[domain]}"
            assert torch.equal(before[key], model.prompts.state_dict()[key])
    assert registry.snapshots["t2"].text.grad is None and all(p.grad is None for p in model.clip.parameters())
    with pytest.raises(TypeError, match="never labels"):
        controlled_step(model, bank, registry, branches, None, {"t1": (torch.randn(5, 5), torch.zeros(5))})


def test_order_is_label_free_deterministic_and_manual_override_supported():
    caches = {d: cache(d) for d in ("t1", "t2", "t3")}
    text = torch.eye(3, 5)
    order = difficulty_order(caches, text, text, torch.tensor(3.))
    assert order["order"] == ["t1", "t2", "t3"] and not order["uses_target_labels"]
    manual = difficulty_order(caches, text, text, torch.tensor(3.), manual=["t3", "t1", "t2"])
    assert manual["order"] == ["t3", "t1", "t2"]
    assert all(not hasattr(c, "labels") for c in caches.values())


def test_new_checkpoint_restores_registry_banks_optimizers_and_exact_resume(tmp_path):
    raw, labels = torch.randn(11, 5), torch.arange(11) % 3
    def setup():
        model = make_model("B")
        bank = SourceCentroidBank(3, 5)
        registry = TeacherRegistry(model.prompts.targets, 3, 5)
        branches = BranchOptimizers(model, 4)
        streams = {d: BatchStream(raw, 4, i, labels if d == "s" else None)
                   for i, d in enumerate(model.prompts.domains)}
        return model, bank, registry, branches, streams
    def step(objects):
        model, bank, registry, branches, streams = objects
        controlled_step(model, bank, registry, branches, streams["s"].next_batch(),
                        {d: streams[d].next_batch() for d in model.prompts.targets}, "distance")
    original = setup()
    step(original)
    snap = TeacherSnapshot.capture(original[0], "t2", 1)
    result = rebuild_target_bank(cache("t2"), snap, original[0].scale, 3, threshold=.1)
    original[2].commit(snap, result)
    save_state(tmp_path / "last.pt", *original, 1, {"seed": 42}, 1., [])
    step(original)
    restored = setup()
    state = load_state(tmp_path / "last.pt", *restored, {"seed": 42})
    assert state["step"] == 1 and restored[2].snapshots["t2"].version == snap.version
    step(restored)
    for name, value in original[0].prompts.state_dict().items():
        torch.testing.assert_close(value, restored[0].prompts.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(original[2].bank.centroids, restored[2].bank.centroids, rtol=0, atol=0)
    assert original[3].updates == restored[3].updates


def test_finished_sequential_target_stays_frozen():
    model = make_model("B")
    branches = BranchOptimizers(model, 4)
    before = model.prompts.class_context("t1").detach().clone()
    freeze_target(model, "t1")
    controlled_step(model, SourceCentroidBank(3, 5), TeacherRegistry(model.prompts.targets, 3, 5), branches,
                    (torch.randn(5, 5), torch.arange(5) % 3), {"t2": torch.randn(5, 5)})
    assert torch.equal(before, model.prompts.class_context("t1"))


def test_b_with_fixed_source_teacher_is_invariant_to_joint_sequential_and_reverse_order():
    """Independent B targets cannot obtain ordering benefits from each other's parameters."""
    raw = torch.randn(13, 5)
    def setup():
        model = make_model("B")
        optimizers = BranchOptimizers(model, 5)
        streams = {d: BatchStream(raw, 4, i) for i, d in enumerate(model.prompts.targets)}
        return model, optimizers, streams, SourceCentroidBank(3, 5), TeacherRegistry(model.prompts.targets, 3, 5)
    def update(objects, active):
        model, optimizers, streams, bank, registry = objects
        controlled_step(model, bank, registry, optimizers, None, {d: streams[d].next_batch() for d in active})
    joint, sequential, reverse = setup(), setup(), setup()
    for _ in range(5):
        update(joint, joint[0].prompts.targets)
    for objects, order in ((sequential, sequential[0].prompts.targets),
                            (reverse, tuple(reversed(reverse[0].prompts.targets)))):
        for domain in order:
            for _ in range(5):
                update(objects, (domain,))
    for objects in (sequential, reverse):
        for name, value in joint[0].prompts.state_dict().items():
            torch.testing.assert_close(value, objects[0].prompts.state_dict()[name], rtol=0, atol=0)
