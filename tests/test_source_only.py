import torch

from mtda.source_bank import SourceCentroidBank
from mtda.trainer import train_step
from train_source_only import SourcePredictor, source_parameters, source_step
from test_mtda import make_model


def test_source_control_matches_b_source_teacher_without_target_updates():
    source_model = make_model("A")
    b_model = make_model("B")
    parameters = source_parameters(source_model)
    assert len(parameters) == 2
    assert {p for p in source_model.parameters() if p.requires_grad} == set(parameters)
    optimizer = torch.optim.AdamW(parameters, lr=0.003)
    b_optimizer = torch.optim.AdamW(b_model.prompts.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    b_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(b_optimizer, T_max=5)
    bank, b_bank = SourceCentroidBank(3, 5), SourceCentroidBank(3, 5)
    untouched = {d: p.detach().clone() for d, p in source_model.prompts.domain_contexts.items()}
    for _ in range(5):
        features = torch.randn(7, 5)
        labels = torch.arange(7) % 3
        source_step(source_model, bank, optimizer, (features, labels))
        train_step(b_model, b_bank, b_optimizer, (features, labels),
                   {d: torch.randn(7, 5) for d in b_model.prompts.targets})
        scheduler.step()
        b_scheduler.step()
        torch.testing.assert_close(source_model.prompts.class_context("s"),
                                   b_model.prompts.class_context("s"), rtol=0, atol=0)
        torch.testing.assert_close(source_model.prompts.domain_contexts["d0"],
                                   b_model.prompts.domain_contexts["d0"], rtol=0, atol=0)
    for domain in ("d1", "d2", "d3"):
        assert not source_model.prompts.domain_contexts[domain].requires_grad
        assert source_model.prompts.domain_contexts[domain].grad is None
        torch.testing.assert_close(source_model.prompts.domain_contexts[domain], untouched[domain], rtol=0, atol=0)
    torch.testing.assert_close(bank.running_means, b_bank.running_means, rtol=0, atol=0)
    assert all(p.grad is None for p in source_model.clip.parameters())


def test_source_predictor_always_uses_source_prototypes():
    model = make_model("A")
    expected = model.forward_source_text().detach()
    predictor = SourcePredictor(model)
    def forbidden(*args):
        raise AssertionError("source-only evaluation called a target prompt")
    model.forward_target_text = forbidden
    predictor.eval()
    for target in model.prompts.targets:
        torch.testing.assert_close(predictor.forward_target_text(target), expected)
    assert not model.clip.training
    assert torch.equal(predictor.base_features, model.base_features)
