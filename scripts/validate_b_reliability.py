#!/usr/bin/env python3
"""Audit budgets, source isolation, teacher versions and preserved B equivalence."""

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from mtda.reliability.experiment import implementation_hashes
from mtda.reliability.schedule import SEQUENTIAL
from mtda.experiment import write_json


def audit(runs_dir, r0_dir, output, expected_steps=1000, current_code=True):
    sealed, code = implementation_hashes()
    proofs, total_updates = [], 0
    for path in sorted(Path(runs_dir).rglob("metrics.json")):
        run = json.loads(path.read_text())
        cfg = run["config"]
        root, source, variant = path.parent, cfg["source"], cfg["variant"]
        if variant not in ("R0M", "R1", "R2", "R3", "R4", "R5", "R6"):
            continue
        steps = cfg["steps"]
        assert steps == expected_steps
        assert cfg["baseline_implementation_sha256"] == sealed
        if current_code:
            assert cfg["implementation_sha256"] == code
        assert not cfg["target_training_ground_truth"]
        state = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
        assert state["config"] == cfg
        assert set(state["branches"]["updates"].values()) == {steps}
        assert state["branches"]["step_calls"] == 4 * steps
        assert set(state["branches"]["samples"].values()) == {cfg["batch_size"] * steps}
        assert int(state["source_bank"]["running_count"].sum()) == cfg["batch_size"] * steps
        assert int((state["source_bank"]["running_count"] > 0).sum()) == 65
        history = list(csv.DictReader((root / "history.csv").open()))
        assert len(history) == 3 * steps
        for target in cfg["targets"]:
            rows = [r for r in history if r["target"] == target]
            assert [int(r["target_update_count"]) for r in rows] == list(range(1, steps + 1))
            assert all(torch.isfinite(torch.tensor(float(r["target_spl"]))) and torch.isfinite(torch.tensor(float(r["total_loss"]))) for r in rows)
        registry = state["registry"]
        for target, bank_version in registry["bank"]["teacher_version"].items():
            if bank_version is not None:
                assert registry["snapshots"][target]["version"] == bank_version
                detail = registry["bank"]["details"][target]
                assert detail["duplicates_skipped"] == 0 and len(set(detail["sample_ids"])) == len(detail["sample_ids"])
                assert torch.equal(detail["counts"], torch.bincount(detail["prediction"][detail["accepted"]], minlength=65))
                assert torch.equal(detail["valid_mask"], detail["counts"] > 0)
        for record in (root / "diagnostics.jsonl").read_text().splitlines():
            record = json.loads(record)
            assert record["target"] not in record["contribution"]["teachers"]
        frozen_accuracies = {}
        if variant in SEQUENTIAL:
            assert len(state["frozen_targets"]) == 3
            for line in (root / "evaluations.jsonl").read_text().splitlines():
                point = json.loads(line)
                for domain, value in frozen_accuracies.items():
                    assert point["targets"][domain]["accuracy"] == value
                if point["event"] == "stage_end":
                    domain = cfg["domain_order"]["order"][point["stage"]]
                    frozen_accuracies[domain] = point["targets"][domain]["accuracy"]
        b_root = Path(r0_dir) / "B" / source.replace(" ", "_") / f"seed_{cfg['seed']}"
        original = torch.load(b_root / "last.pt", map_location="cpu", weights_only=False)
        assert original["config"]["steps"] == steps
        for name in ("class_contexts.d0", "domain_contexts.d0"):
            assert torch.equal(state["prompts"][name], original["prompts"][name]), name
        assert all(torch.equal(value, original["bank"][key]) for key, value in state["source_bank"].items())
        if variant == "R0M":
            assert all(torch.equal(value, original["prompts"][key]) for key, value in state["prompts"].items())
        total_updates += sum(state["branches"]["updates"].values())
        proofs.append({"variant": variant, "source": source, "seed": cfg["seed"],
                       "updates_each_branch": steps, "original_b_source_parameters_exact": True,
                       "source_bank_exact": True, "r0m_all_parameters_exact": variant == "R0M",
                       "self_teacher_exclusion": True, "target_bank_versions_consistent": True,
                       "unique_bank_samples": True, "finite_training": True,
                       "completed_target_accuracy_unchanged": variant in SEQUENTIAL})
    result = {"completed_runs_audited": len(proofs), "effective_branch_updates": total_updates,
              "baseline_implementation_sha256": sealed, "implementation_sha256": code,
              "training_target_ground_truth": False, "proofs": proofs}
    write_json(output, result)
    print(f"Audited {len(proofs)} runs and {total_updates} effective branch updates: exact source/B controls, budgets, versions, self exclusion and frozen accuracy.")
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs-dir", default="runs/b_reliability")
    cli.add_argument("--r0-dir", default="runs/mtda_spl")
    cli.add_argument("--output", default="reports/b_reliability/artifact_audit.json")
    cli.add_argument("--steps", type=int, default=1000)
    cli.add_argument("--allow-older-code", action="store_true")
    args = cli.parse_args()
    audit(args.runs_dir, args.r0_dir, args.output, args.steps, not args.allow_older_code)


if __name__ == "__main__":
    main()
