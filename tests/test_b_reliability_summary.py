import json

import pytest

from scripts.summarize_b_reliability import DOMAINS, METHODS, summarize


def fixtures(tmp_path, seeds=(42, 43)):
    runs, original = tmp_path / "runs", tmp_path / "original"
    for seed in seeds:
        for source in DOMAINS:
            targets = [d for d in DOMAINS if d != source]
            for index, method in enumerate(METHODS):
                root = (original / "B" if method == "R0" else runs / method) / source.replace(" ", "_") / f"seed_{seed}"
                root.mkdir(parents=True)
                config = {"variant": "B" if method == "R0" else method, "source": source,
                          "targets": targets, "seed": seed, "steps": 2, "batch_size": 30,
                          "m1": 16, "m2": 16, "learning_rate": .003, "lambda_target": .5,
                          "checkpoint_sha256": "fixed", "data_root": "fixed",
                          "target_training_ground_truth": False, "checkpoint_selection": "fixed final budget",
                          "domain_order": {"order": targets, "rule": "fixed label-free"}}
                result = {"config": config, "targets": {d: {"accuracy": .6 + .01 * index + .02 * (seed - 42),
                          "samples": 5} for d in targets}, "training_seconds": 1, "peak_memory_bytes": 0,
                          "effective_updates": {d: 2 for d in [source, *targets]}}
                (root / "metrics.json").write_text(json.dumps(result))
    return runs, original


def test_summary_seed_level_macro_and_paired_deltas(tmp_path):
    runs, original = fixtures(tmp_path)
    result = summarize(runs, original, tmp_path / "report", (42, 43))
    assert result["complete"]
    row = result["summary"][-1]
    assert row["R0M"] == pytest.approx(62.)
    assert row["R0M_std"] == pytest.approx(2 ** .5)
    delta = next(r for r in result["comparisons"] if r["question"] == "E2H_bank" and r["source"] == "All")
    assert delta["delta_pp"] == pytest.approx(1.)
    assert delta["std_pp"] == pytest.approx(0., abs=1e-12)
    assert all(r["positive_sources"] == 4 for r in result["candidates"])
    selected = summarize(runs, original, tmp_path / "selected", (42, 43), selected_methods=("R0", "R1", "R2"))
    assert selected["complete"] and "R3" not in selected["summary"][-1]


def test_summary_missing_task_stays_incomplete_and_rejects_ground_truth(tmp_path):
    runs, original = fixtures(tmp_path)
    path = runs / "R6" / "Product" / "seed_43" / "metrics.json"
    state = json.loads(path.read_text())
    state["targets"].pop("Art")
    path.write_text(json.dumps(state))
    result = summarize(runs, original, tmp_path / "report", (42, 43))
    assert not result["complete"] and result["summary"][-1]["R6_seeds"] == 1
    state["config"]["target_training_ground_truth"] = True
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="ground-truth"):
        summarize(runs, original, tmp_path / "report", (42, 43))
