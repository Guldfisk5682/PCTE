#!/usr/bin/env python3
"""Audit follow-up budgets, immutable references, teacher masking, and old baselines."""

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from mtda.experiment import write_json, file_sha
from mtda.reliability_followup.provenance import implementation_hashes


def audit(runs, baseline, output):
    _, runtime = implementation_hashes()
    records = []
    banks_by_reference = {}
    for path in sorted(runs.rglob("metrics.json")):
        run = json.loads(path.read_text())
        cfg, root = run["config"], path.parent
        assert cfg["implementation_sha256"] == runtime
        assert not cfg["target_training_ground_truth"] and cfg["checkpoint_selection"] == "fixed final budget"
        assert run["effective_updates"] == {d: cfg["steps"] for d in [cfg["source"], *cfg["targets"]]}
        assert run["sample_exposures"] == {d: cfg["steps"] * cfg["batch_size"] for d in [cfg["source"], *cfg["targets"]]}
        assert run["optimizer_step_calls"] == cfg["steps"] * 4
        state = torch.load(root / "last.pt", weights_only=False, map_location="cpu")
        reference = torch.load(baseline / "R4" / cfg["source"].replace(" ", "_") / f"seed_{cfg['seed']}" / "last.pt",
                               weights_only=False, map_location="cpu")
        for k in ("class_contexts.d0", "domain_contexts.d0"):
            assert torch.equal(state["prompts"][k], reference["prompts"][k]), "Source trajectory changed"
        for k, value in state["source_bank"].items():
            assert torch.equal(value, reference["source_bank"][k])
        exact = cfg["variant"] == "FUniform"
        if exact:
            assert all(torch.equal(v, reference["prompts"][k]) for k, v in state["prompts"].items()), "uniform control differs from R4"
        for d, snapshot in state["registry"]["snapshots"].items():
            if cfg["snapshot_policy"] == "reference":
                provenance = cfg["reference_provenance"]["snapshots"][d]
                assert file_sha(provenance["path"]) == provenance["sha256"]
                original = torch.load(provenance["path"], weights_only=True, map_location="cpu")["snapshot"]
                assert snapshot["version"] == original["version"]
                for field in ("text", "class_context", "domain_context"):
                    assert torch.equal(snapshot[field], original[field])
            else:
                assert snapshot["step"] == 0
        if cfg["teacher_mode"] == "distance":
            for d, bank in state["registry"]["bank"]["details"].items():
                version = bank["teacher_version"]
                assert version == state["registry"]["snapshots"][d]["version"]
                assert torch.equal(bank["valid_mask"], bank["counts"] > 0)
                assert bank["unique_samples"] == len(set(bank["sample_ids"])) and bank["duplicates_skipped"] == 0
                assert int(bank["counts"].sum()) == int(bank["accepted"].sum())
                key = (cfg["source"], cfg["seed"], d)
                if key in banks_by_reference:
                    for field in ("centroids", "counts", "accepted", "prediction", "confidence"):
                        assert torch.equal(bank[field], banks_by_reference[key][field]), "beta scan changed reference bank"
                banks_by_reference[key] = bank
        rows = list(csv.DictReader((root / "history.csv").open()))
        assert len(rows) == cfg["steps"] * 3
        for d in cfg["targets"]:
            selected = [r for r in rows if r["target"] == d]
            assert [int(r["target_update_count"]) for r in selected] == list(range(1, cfg["steps"] + 1))
        for row in rows:
            for field in ("source_ce", "target_spl", "total_loss"):
                assert torch.isfinite(torch.tensor(float(row[field])))
            versions = json.loads(row["teacher_version"])
            available = [d for d in versions if d != "Source" and d != row["target"]]
            assert row["target"] not in available
            if cfg["admission_policy"] == "start":
                assert len(available) == 2
            else:
                expected = [d for d, step in zip(cfg["domain_order"]["order"], cfg["admission_steps"])
                            if int(row["step"]) > step and d != row["target"]]
                assert set(expected) == set(available)
        for line in (root / "diagnostics.jsonl").read_text().splitlines():
            r = json.loads(line)
            assert r["target"] not in r["contribution"]["teachers"]
        records.append({"run": str(root), "variant": cfg["variant"], "source": cfg["source"], "seed": cfg["seed"],
                        "source_exact_R4": True, "fixed_reference_exact": cfg["snapshot_policy"] == "reference",
                        "uniform_all_parameters_exact_R4": exact, "equal_student_budgets": True,
                        "finite_losses": True, "self_teacher_excluded": True})
    result = {"completed_runs": len(records), "implementation_sha256": runtime, "runs": records,
              "fixed_reference_beta_banks_identical": True}
    write_json(output, result)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs-dir", default="runs/b_followup")
    cli.add_argument("--baseline-dir", default="runs/b_reliability")
    cli.add_argument("--output", default="reports/b_followup/validation.json")
    args = cli.parse_args()
    result = audit(Path(args.runs_dir), Path(args.baseline_dir), Path(args.output))
    print(json.dumps({"validated_runs": result["completed_runs"]}))


if __name__ == "__main__":
    main()
