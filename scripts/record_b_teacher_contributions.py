#!/usr/bin/env python3
"""Recover actual in-training teacher weights from immutable caches and versioned banks.

This offline audit performs no optimizer updates and reads no target ground truth.
Source cumulative means and target batch streams are deterministic; text prototypes
do not enter centroid weights. Thus historical contribution statistics can be
reproduced without pretending that teachers admitted later taught a frozen target.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from mtda.data import BatchStream, SourceImages, discover_domains
from mtda.experiment import write_json
from mtda.source_bank import SourceCentroidBank
from mtda.reliability.bank import TeacherSnapshot, TeacherRegistry
from mtda.reliability.teacher import ensemble_spl, weight_diagnostics


def batch_seed(seed, domain):
    return int.from_bytes(hashlib.sha256(f"{seed}:batch:{domain}".encode()).digest()[:8], "big") % (2**63)


def record(root, cache_dir, device="cpu", interval=100):
    root = Path(root)
    cfg = json.loads((root / "config.json").read_text())
    domains, source = discover_domains(cfg["data_root"]), cfg["source"]
    caches = {d: torch.load(Path(cache_dir) / "domain_class" / f"{key}.pt", map_location="cpu", weights_only=True)
              for d, key in cfg["domain_cache_fingerprints"].items()}
    # Source labels only; target caches contain features/IDs and no labels.
    labels = SourceImages(domains[source], None, cfg["classnames"]).labels
    source_stream = BatchStream(caches[source]["z_pre"], cfg["batch_size"], batch_seed(cfg["seed"], source), labels)
    streams = {d: BatchStream(caches[d]["z_pre"], cfg["batch_size"], batch_seed(cfg["seed"], d)) for d in cfg["targets"]}
    source_bank = SourceCentroidBank(len(cfg["classnames"]), caches[source]["z_pre"].shape[-1]).to(device)
    snapshots, banks = {}, {}
    for path in (root / "teacher_snapshots").glob("*.pt"):
        state = torch.load(path, map_location="cpu", weights_only=True)
        snapshot = state["snapshot"]
        snapshots[snapshot["version"]] = snapshot
    for path in (root / "bank_snapshots").glob("*.pt"):
        state = torch.load(path, map_location="cpu", weights_only=True)
        banks[state["bank"]["teacher_version"]] = state["bank"]
    registry = TeacherRegistry(cfg["targets"], len(cfg["classnames"]), caches[source]["z_pre"].shape[-1], device)
    events = {json.loads(line)["step"] for line in (root / "events.jsonl").read_text().splitlines()}
    source_updates = 0
    results = []
    # Only weights are inspected. Normalized placeholders for Source/Base text
    # keep the reusable fusion API well-formed; its discarded probabilities are
    # NOT pseudo-label-quality measurements, predictions, or training signals.
    placeholder = F.normalize(torch.ones(len(cfg["classnames"]), caches[source]["z_pre"].shape[-1], device=device), dim=-1)
    scale = torch.tensor(1., device=device)
    for row in csv.DictReader((root / "history.csv").open()):
        step, target = int(row["step"]), row["target"]
        requested = int(row["source_update_count"])
        while source_updates < requested:
            raw, truth = source_stream.next_batch()
            source_bank.update(raw.to(device), truth.to(device))
            source_updates += 1
        versions = json.loads(row["teacher_version"])
        actual = {d: version for d, version in versions.items() if d != "Source"}
        for domain, version in actual.items():
            if domain not in registry.snapshots or registry.snapshots[domain].version != version:
                registry.commit(TeacherSnapshot.from_state(snapshots[version], device), banks.get(version))
        # Advance each target stream at EVERY effective update, not just audit intervals.
        raw = streams[target].next_batch()
        local_update = int(row["target_update_count"])
        if local_update == 1 or local_update % interval == 0 or step in events or local_update == cfg["steps"]:
            _, info = ensemble_spl(raw.to(device), placeholder, placeholder, scale, source_bank,
                                    registry, target, cfg["teacher_mode"], cfg["beta"])
            result = {"source": source, "target": target, "variant": cfg["variant"], "seed": cfg["seed"],
                      "step": step, "stage": int(row["stage"]), "teacher_version": versions,
                      "target_effective_update": local_update, "source_updates": requested,
                      "replay_device": device,
                      "scope": "actual active training batch; not post-hoc final ensemble",
                      **weight_diagnostics(info)}
            assert target not in result["teachers"]
            results.append(result)
    saved = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
    for key, value in source_bank.state_dict().items():
        if device.startswith("cuda"):
            assert torch.equal(value.cpu(), saved["source_bank"][key]), "source bank replay differs"
        else:
            torch.testing.assert_close(value.cpu(), saved["source_bank"][key], atol=2e-5, rtol=2e-5)
    write_json(root / "actual_teacher_contributions.json", results)
    return results


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs-dir", default="runs/b_reliability")
    cli.add_argument("--cache-dir", default="runs/feature_cache")
    cli.add_argument("--device", default="cuda:0", help="same GPU arithmetic for exact source-bank replay")
    cli.add_argument("--interval", type=int, default=100)
    args = cli.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    for path in sorted(Path(args.runs_dir).rglob("metrics.json")):
        root = path.parent
        if (root / "actual_teacher_contributions.json").exists():
            existing = json.loads((root / "actual_teacher_contributions.json").read_text())
            if existing and existing[0].get("replay_device", "cpu") == args.device:
                continue
        rows = record(root, args.cache_dir, args.device, args.interval)
        print(f"Recorded {len(rows)} actual active-training contributions: {root}", flush=True)


if __name__ == "__main__":
    main()
