import json

import pytest

from scripts.run_officehome_mtda import experiment_matrix
from scripts.summarize_mtda_baselines import DOMAINS, summarize_baselines


def fixture_results(root, seeds=(41, 42, 43)):
    common = {"steps": 1000, "batch_size": 30, "m1": 16, "m2": 16, "learning_rate": 0.003,
              "weight_decay": 0.01, "scheduler": "cosine_per_update", "precision": "float32",
              "checkpoint_sha256": "weights", "data_root": "data", "classnames": ["a", "b"],
              "feature_fingerprints": {d: d for d in DOMAINS},
              "implementation_sha256": "core", "upstream_sha": "upstream"}
    for i, (variant, source, target, seed) in enumerate(experiment_matrix(DOMAINS, seeds)):
        cfg = {**common, "variant": variant, "source": source, "seed": seed}
        targets = [target] if target else [d for d in DOMAINS if d != source]
        metrics = {"config": cfg, "training_seconds": 1, "peak_memory_bytes": 0,
                   "targets": {d: {"accuracy": 0.6 + "ABC".index(variant) * 0.03 + (seed - 41) * 0.01,
                                   "zero_shot": 0.5} for d in targets}}
        path = root / "abc" / str(i)
        path.mkdir(parents=True)
        (path / "metrics.json").write_text(json.dumps(metrics))
    for seed in seeds:
        for source in DOMAINS:
            cfg = {**common, "variant": "SourceOnly", "source": source, "seed": seed,
                   "objective": "source_cross_entropy_only", "target_training_images": 0,
                   "implementation_sha256": "control", "baseline_implementation_sha256": "core"}
            metrics = {"config": cfg, "training_seconds": 1, "peak_memory_bytes": 0,
                       "evaluation_fingerprints": {d: d for d in DOMAINS if d != source},
                       "targets": {d: {"accuracy": 0.55 + (seed - 41) * 0.01, "zero_shot": 0.5}
                                   for d in DOMAINS if d != source}}
            path = root / "source" / source / str(seed)
            path.mkdir(parents=True)
            (path / "metrics.json").write_text(json.dumps(metrics))


def test_matched_summary_uses_seed_level_macros_and_paired_differences(tmp_path):
    fixture_results(tmp_path)
    result = summarize_baselines(tmp_path / "abc", tmp_path / "source", tmp_path / "reports")
    assert result["complete"] and result["abc_runs"] == 60 and result["source_runs"] == 12
    assert len(result["detail"]) == 36 and len(result["tasks"]) == 12
    macro = result["source_summary"][-1]
    assert macro["source_only"] == pytest.approx(56)
    assert macro["A"] == pytest.approx(61)
    assert macro["A_std"] == pytest.approx(1)
    assert macro["A_minus_source_only"] == pytest.approx(5)
    # Correlated seed shifts cancel in paired differences; do not add stds.
    assert macro["A_minus_source_only_std"] == pytest.approx(0, abs=1e-12)
    assert macro["zero_shot"] == pytest.approx(50)
    assert macro["zero_shot_std"] == pytest.approx(0)
    assert all(row["A_seeds"] == 3 for row in result["target_summary"])


def test_missing_control_seed_is_pending_and_protocol_mismatch_fails(tmp_path):
    fixture_results(tmp_path)
    path = tmp_path / "source" / "Art" / "43" / "metrics.json"
    preserved = path.read_text()
    path.unlink()
    result = summarize_baselines(tmp_path / "abc", tmp_path / "source", tmp_path / "reports")
    assert not result["complete"]
    assert result["source_summary"][-1]["source_only_seeds"] == 2
    assert "incomplete" in (tmp_path / "reports" / "baseline_report.md").read_text()
    metric = json.loads(preserved)
    metric["config"]["target_training_images"] = 30
    path.write_text(json.dumps(metric))
    with pytest.raises(ValueError, match="consumed target"):
        summarize_baselines(tmp_path / "abc", tmp_path / "source", tmp_path / "reports")
