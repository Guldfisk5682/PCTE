#!/usr/bin/env python3
"""Train the independent source-only control for four sources and every seed."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import queue
import re
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", required=True)
    cli.add_argument("--clip-checkpoint", required=True)
    cli.add_argument("--devices", nargs="+", default=["cuda:0"])
    cli.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    cli.add_argument("--steps", type=int, default=1000)
    cli.add_argument("--output-dir", default="runs/source_prompt_only")
    cli.add_argument("--cache-dir", default="runs/feature_cache")
    cli.add_argument("--retry-failed", action="store_true")
    cli.add_argument("--dry-run", action="store_true")
    args = cli.parse_args()
    from mtda.data import discover_domains
    from mtda.experiment import file_sha
    from train_source_only import baseline_digest
    domains = discover_domains(args.data_root)
    jobs = [(source, seed) for seed in args.seeds for source in domains]
    if len(set(args.devices)) != len(args.devices) or len(set(args.seeds)) != len(args.seeds):
        cli.error("devices and seeds must be unique")
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    weight_sha = file_sha(args.clip_checkpoint)
    implementation = hashlib.sha256((file_sha(PROJECT / "train_source_only.py") + baseline_digest()).encode()).hexdigest()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    devices = queue.Queue()
    for device in args.devices:
        devices.put(device)

    def execute(job):
        source, seed = job
        run = root / source.replace(" ", "_") / f"seed_{seed}"
        if (run / "metrics.json").exists():
            metric = json.loads((run / "metrics.json").read_text())
            cfg = metric["config"]
            if (metric["status"] != "complete" or cfg["implementation_sha256"] != implementation
                    or cfg["checkpoint_sha256"] != weight_sha or cfg["steps"] != args.steps
                    or cfg["data_root"] != str(Path(args.data_root).resolve())):
                raise ValueError(f"existing source-only configuration differs: {run}")
            print(f"SKIP {job}", flush=True)
            return 0
        command = [sys.executable, "train_source_only.py", "--source", source, "--seed", str(seed),
                   "--data-root", args.data_root, "--clip-checkpoint", args.clip_checkpoint,
                   "--steps", str(args.steps), "--output-dir", args.output_dir, "--cache-dir", args.cache_dir]
        if (run / "config.json").exists():
            if not args.retry_failed or not (run / "last.pt").exists():
                raise FileExistsError(f"existing source-only run needs a checkpoint and --retry-failed: {run}")
            command.append("--resume")
        device = devices.get()
        try:
            command += ["--device", device]
            path = root / f"{source.replace(' ', '_')}_{seed}.log"
            print(f"START SourceOnly {job} on {device}", flush=True)
            with path.open("a", buffering=1) as log:
                log.write(json.dumps({"command": command}) + "\n")
                process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                estimated = False
                for line in process.stdout:
                    log.write(line)
                    match = re.search(r"step (\d+)/(\d+):.*elapsed=([\d.]+)s", line)
                    if match and int(match[1]) >= 20 and not estimated:
                        step, total, seconds = int(match[1]), int(match[2]), float(match[3])
                        finish = datetime.now(timezone(timedelta(hours=8))) + timedelta(seconds=seconds / step * (total - step))
                        print(f"ETA SourceOnly {job}: ≈{finish:%m-%d %H:%M:%S} UTC+8 ({seconds / step:.3f}s/update)", flush=True)
                        estimated = True
                code = process.wait()
            print(f"{'DONE' if code == 0 else 'FAILED'} SourceOnly {job}", flush=True)
            return code
        finally:
            devices.put(device)

    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        codes = list(pool.map(execute, jobs))
    if any(codes):
        raise SystemExit("source-only runs failed; inspect logs and retry")


if __name__ == "__main__":
    main()
