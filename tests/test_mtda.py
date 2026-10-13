import copy
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mtda.data import BatchStream, UnlabeledImages, discover_domains, validate_classes, transform_signature
from mtda.evaluate import accuracy
from mtda.experiment import load_checkpoint, save_checkpoint, seed_everything
from mtda.prompts import MTDAPromptGenerator, PromptModel
from mtda.source_bank import SourceCentroidBank
from mtda.spl import compute_spl_soft_targets, soft_cross_entropy
from mtda.trainer import train_step
from scripts.run_officehome_mtda import experiment_matrix
from scripts.summarize_mtda import summarize

torch.set_num_threads(1)


def tokenize(names):
    tokens = torch.zeros(len(names), 12, dtype=torch.long)
    tokens[:, 0] = 1
    tokens[:, 1:5] = 2
    for i in range(len(names)):
        tokens[i, 5:10] = 3 + i
    tokens[:, 10] = 127
    return tokens


class CumulativeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x).cumsum(0)


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(128, 8)
        self.ln_final = nn.LayerNorm(8)
        self.transformer = CumulativeTransformer()
        self.positional_embedding = nn.Parameter(torch.randn(12, 8) * 0.01)
        self.text_projection = nn.Parameter(torch.randn(8, 5))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    @property
    def dtype(self):
        return torch.float32

    def encode_text(self, tokens):
        x = self.token_embedding(tokens) + self.positional_embedding
        x = self.ln_final(self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2))
        return x[torch.arange(len(x)), tokens.argmax(-1)] @ self.text_projection


def make_model(variant="A", targets=("t1", "t2", "t3"), seed=42):
    torch.manual_seed(10)
    clip = TinyCLIP()
    names = ["one", "two", "three"]
    prompts = MTDAPromptGenerator(names, clip, tokenize, "s", targets, variant, seed, 2, 2)
    return PromptModel(clip, prompts, tokenize(names))


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_shapes_and_frozen_encoder(variant):
    model = make_model(variant, ("t1",) if variant == "C" else ("t1", "t2", "t3"))
    assert model.prompts.class_context("s").shape == (3, 2, 8)
    assert all(p.shape == (2, 8) for p in model.prompts.domain_contexts.values())
    assert model.prompts("s").shape == (3, 12, 8)
    assert model.forward_text("s").shape == (3, 5)
    model.train()
    assert not model.clip.training
    assert all(not p.requires_grad for p in model.clip.parameters())
    assert set(model.prompts.parameters()) == {p for p in model.parameters() if p.requires_grad}


def test_sharing_and_initial_values_across_variants():
    a, b = make_model("A"), make_model("B")
    c1, c2 = make_model("C", ("t1",)), make_model("C", ("t2",))
    for domain in a.prompts.domains:
        assert a.prompts.class_context(domain) is a.prompts.class_context("s")
        torch.testing.assert_close(a.prompts.class_context(domain), b.prompts.class_context(domain), rtol=0, atol=0)
    assert len({id(b.prompts.class_context(d)) for d in b.prompts.domains}) == 4
    assert len({b.prompts.class_context(d).data_ptr() for d in b.prompts.domains}) == 4
    for c in (c1, c2):
        assert c.prompts.class_context("s") is c.prompts.class_context(c.prompts.targets[0])
        assert c.prompts.class_context("s") is not c1.prompts.class_context("s") or c is c1
        torch.testing.assert_close(a.prompts.class_context("s"), c.prompts.class_context("s"), rtol=0, atol=0)
        for domain in c.prompts.domains:
            torch.testing.assert_close(a.prompts.domain_contexts[a.prompts.domain_keys[domain]],
                                       c.prompts.domain_contexts[c.prompts.domain_keys[domain]], rtol=0, atol=0)
    with torch.no_grad():
        c1.prompts.class_context("s").add_(1)
    torch.testing.assert_close(c2.prompts.class_context("s"), a.prompts.class_context("s"))


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_loss_gradient_routes(variant):
    model = make_model(variant, ("t1",) if variant == "C" else ("t1", "t2", "t3"))
    images = F.normalize(torch.randn(7, 5), dim=-1)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    for domain in model.prompts.domains:
        model.zero_grad(set_to_none=True)
        text = model.forward_text(domain)
        if domain == "s":
            loss = F.cross_entropy(model.scale * images @ text.T, labels)
        else:
            teacher = compute_spl_soft_targets(images, model.base_features, model.forward_source_text(), model.scale)
            loss = soft_cross_entropy(model.scale * images @ text.T, teacher)
        loss.backward()
        for name in model.prompts.domains:
            class_gradient = model.prompts.class_context(name).grad
            expected = variant != "B" or name == domain
            assert (class_gradient is not None) == expected
            if expected:
                assert class_gradient.abs().sum() > 0
            domain_gradient = model.prompts.domain_contexts[model.prompts.domain_keys[name]].grad
            assert (domain_gradient is not None) == (name == domain)
        assert all(p.grad is None for p in model.clip.parameters())


def test_bank_matches_direct_mean_and_single_source_degeneracy():
    bank = SourceCentroidBank(4, 5)
    assert bank.running_means.count_nonzero() == 0
    assert bank.running_count.count_nonzero() == 0
    torch.manual_seed(1)
    x = torch.randn(9, 5, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 2, 1, 0, 2, 0, 2])
    bank.update(x[:4], labels[:4])
    bank.update(x[4:], labels[4:])
    for k in range(3):
        torch.testing.assert_close(bank.running_means[k], x.detach()[labels == k].mean(0))
    assert bank.running_count.tolist() == [4, 2, 3, 0]
    assert bank.running_means[3].count_nonzero() == 0
    assert not bank.running_means.requires_grad
    assert torch.equal(bank.weights(x), torch.ones(9, 1, 4))
    model = make_model()
    p = compute_spl_soft_targets(F.normalize(x, dim=-1), model.base_features, model.forward_source_text(), model.scale)
    bank.running_means.fill_(10000)
    q = compute_spl_soft_targets(F.normalize(x, dim=-1), model.base_features, model.forward_source_text(), model.scale)
    torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_spl_matches_logits_fusion_and_soft_ce_formula():
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    base = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    source = torch.tensor([[0.6, 0.8], [-1.0, 0.0]], requires_grad=True)
    scale = torch.tensor(3.0, requires_grad=True)
    p = compute_spl_soft_targets(z, base, source, scale)
    expected = torch.softmax(3 * (0.5 * z @ base.T + 0.5 * z @ source.T), -1)
    torch.testing.assert_close(p, expected)
    assert not p.requires_grad
    assert not torch.allclose(p, 0.5 * (3 * z @ base.T).softmax(-1) + 0.5 * (3 * z @ source.T).softmax(-1))
    logits = torch.tensor([[0.2, 0.8], [0.7, 0.3]], requires_grad=True)
    loss = soft_cross_entropy(logits, p)
    manual = -sum((p[i] * torch.log_softmax(logits[i], 0)).sum() for i in range(2)) / 2
    torch.testing.assert_close(loss, manual)
    loss.backward()
    assert logits.grad is not None
    assert z.grad is None and base.grad is None and source.grad is None and scale.grad is None


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_accumulated_step_matches_single_total_loss(variant):
    model = make_model(variant, ("t1",) if variant == "C" else ("t1", "t2", "t3"))
    reference = copy.deepcopy(model)
    source = (torch.randn(7, 5), torch.tensor([0, 1, 2, 0, 1, 2, 0]))
    targets = {d: torch.randn(7, 5) for d in model.prompts.targets}
    optimizer = torch.optim.SGD(model.prompts.parameters(), lr=0.1)
    other = torch.optim.SGD(reference.prompts.parameters(), lr=0.1)
    actual = train_step(model, SourceCentroidBank(3, 5), optimizer, source, targets)
    source_text = reference.forward_source_text()
    total = F.cross_entropy(reference.scale * F.normalize(source[0], dim=-1) @ source_text.T, source[1])
    target_losses = []
    for d, features in targets.items():
        z = F.normalize(features, dim=-1)
        p = compute_spl_soft_targets(z, reference.base_features, source_text, reference.scale)
        target_losses.append(soft_cross_entropy(reference.scale * z @ reference.forward_target_text(d).T, p))
    total = total + 0.5 * sum(target_losses) / len(targets)
    total.backward()
    other.step()
    assert actual["loss"] == pytest.approx(float(total.detach()))
    for p, q in zip(model.prompts.parameters(), reference.prompts.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)


def test_target_dataset_and_stream_never_expose_labels(tmp_path):
    from PIL import Image
    for domain in ("art", "clipart", "product", "real_world"):
        for i in range(65):
            (tmp_path / domain / f"class_{i:02}").mkdir(parents=True)
    Image.new("RGB", (3, 3)).save(tmp_path / "clipart" / "class_00" / "image.png")
    domains = discover_domains(tmp_path)
    assert len(validate_classes(domains, "Art")) == 65
    dataset = UnlabeledImages(domains["Clipart"], lambda image: torch.zeros(3, 3, 3))
    assert set(vars(dataset)) == {"paths", "transform"}
    assert isinstance(dataset[0], torch.Tensor)
    stream = BatchStream(torch.randn(7, 5), 3, 1)
    assert stream.labels is None and isinstance(stream.next_batch(), torch.Tensor)
    (tmp_path / "product" / "class_64").rmdir()
    with pytest.raises(ValueError, match="vocabulary differs"):
        validate_classes(domains, "Art")


def test_evaluation_counts_tail_batch_and_uses_given_text():
    features = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.], [1., 0.]])
    labels = torch.tensor([0, 1, 0, 1, 1])
    result = accuracy(features, labels, torch.eye(2), torch.tensor(1.), batch_size=2)
    assert result == {"accuracy": 0.8, "correct": 4, "samples": 5}
    switched = accuracy(features, labels, torch.eye(2).flip(0), torch.tensor(1.), batch_size=2)
    assert switched["accuracy"] == 0.2


def test_checkpoint_round_trip_and_exact_training_resume(tmp_path):
    seed_everything(7)
    features = torch.randn(11, 5)
    labels = torch.arange(11) % 3

    def setup():
        model = make_model("B")
        bank = SourceCentroidBank(3, 5)
        opt = torch.optim.AdamW(model.prompts.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
        streams = {d: BatchStream(features, 4, i, labels if d == "s" else None)
                   for i, d in enumerate(model.prompts.domains)}
        return model, bank, opt, scheduler, streams

    def step(objects):
        model, bank, opt, scheduler, streams = objects
        train_step(model, bank, opt, streams["s"].next_batch(),
                   {d: streams[d].next_batch() for d in model.prompts.targets})
        scheduler.step()

    original = setup()
    step(original)
    model, bank, opt, scheduler, streams = original
    save_checkpoint(tmp_path / "last.pt", *original, 1, {"seed": 7}, 1.0)
    before = model.forward_target_text("t1").detach().clone()
    step(original)
    step(original)
    restored = setup()
    state = load_checkpoint(tmp_path / "last.pt", *restored, {"seed": 7})
    assert state["step"] == 1
    torch.testing.assert_close(restored[0].forward_target_text("t1"), before, rtol=0, atol=0)
    step(restored)
    step(restored)
    for key, value in original[0].prompts.state_dict().items():
        torch.testing.assert_close(value, restored[0].prompts.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(original[1].running_means, restored[1].running_means, rtol=0, atol=0)
    assert original[3].state_dict() == restored[3].state_dict()
    for domain in streams:
        torch.testing.assert_close(original[4][domain].next_batch()[0] if domain == "s" else original[4][domain].next_batch(),
                                   restored[4][domain].next_batch()[0] if domain == "s" else restored[4][domain].next_batch())
    with pytest.raises(ValueError, match="configuration differs"):
        load_checkpoint(tmp_path / "last.pt", *restored, {"seed": 8})


@pytest.mark.parametrize("seeds", [[42], [42, 43]])
def test_matrix_and_complete_summary(tmp_path, seeds):
    domains = ["Art", "Clipart", "Product", "Real World"]
    jobs = list(experiment_matrix(domains, seeds))
    assert len(jobs) == 20 * len(seeds)
    assert [sum(job[0] == v for job in jobs) for v in "ABC"] == [4 * len(seeds), 4 * len(seeds), 12 * len(seeds)]
    for i, (variant, source, target, seed) in enumerate(jobs):
        root = tmp_path / "runs" / str(i)
        root.mkdir(parents=True)
        cfg = {"seed": seed, "variant": variant, "source": source, "steps": 1000, "batch_size": 30,
               "m1": 16, "m2": 16, "learning_rate": 0.003, "lambda_target": 0.5,
               "checkpoint_sha256": "fake", "implementation_sha256": "fake", "upstream_sha": "fake",
               "weight_decay": 0.01, "data_root": "fake", "clip_checkpoint": "fake"}
        targets = [target] if target else [d for d in domains if d != source]
        metrics = {"config": cfg, "targets": {d: {"accuracy": 0.5 + "ABC".index(variant) * 0.1 + (seed - 42) * 0.02,
                   "zero_shot": 0.4} for d in targets}, "training_seconds": 1, "peak_memory_bytes": 0,
                   "bank": {"observed_classes": 65, "classes": 65, "samples": 30000}}
        (root / "metrics.json").write_text(json.dumps(metrics))
    result = summarize(tmp_path / "runs", tmp_path / "reports")
    assert result["runs"] == 20 * len(seeds) and result["task_results"] == 36 * len(seeds) and len(result["tasks"]) == 12
    offset = 1 if len(seeds) > 1 else 0
    assert result["summary"][-1]["A"] == pytest.approx(50 + offset)
    assert result["summary"][-1]["B"] == pytest.approx(60 + offset)
    assert result["summary"][-1]["C"] == pytest.approx(70 + offset)
    assert result["summary"][-1]["zero_shot"] == pytest.approx(40)
    assert result["summary"][-1]["A_complete_seeds"] == len(seeds)
    if len(seeds) > 1:
        assert result["summary"][-1]["A_std"] == pytest.approx(2**0.5)
    else:
        assert result["summary"][-1]["A_std"] is None


def test_empty_summary_has_no_fabricated_macro_scores(tmp_path):
    result = summarize(tmp_path / "empty", tmp_path / "reports")
    assert result["runs"] == 0 and result["task_results"] == 0
    assert len(result["tasks"]) == 12
    assert all(result["summary"][-1][v] is None for v in "ABC")
    assert "not yet established" in (tmp_path / "reports" / "officehome_mtda_report.md").read_text()


def test_c_rejects_joint_targets_and_classname_truncation():
    with pytest.raises(ValueError, match="exactly one"):
        make_model("C")
    with pytest.raises(ValueError, match="no room"):
        MTDAPromptGenerator(["x"], TinyCLIP(), tokenize, "s", ["t"], m1=7, m2=3)


def test_cache_signature_ignores_callable_addresses():
    class Transform:
        def __init__(self, address):
            self.address = address

        def __repr__(self):
            return f"Compose(Resize(224), <function rgb at {self.address}>, Normalize(0.1))"

    assert transform_signature(Transform("0xabcdef")) == transform_signature(Transform("0x123456"))
    assert "Resize(224)" in transform_signature(Transform("0xabcdef"))
