#!/usr/bin/env python3
"""Run all 20 independent experiments per seed, with one process per device."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def experiment_matrix(domains, seeds):
    for seed in seeds:
        for variant in ("A", "B", "C"):
            for source in domains:
                if variant == "C":
                    for target in domains:
                        if source != target:
                            yield variant, source, target, seed
                else:
                    yield variant, source, None, seed


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", required=True)
    cli.add_argument("--clip-checkpoint", required=True)
    cli.add_argument("--devices", nargs="+", default=["cuda:0"])
    cli.add_argument("--seeds", nargs="+", type=int, default=[42])
    cli.add_argument("--steps", type=int, default=1000)
    cli.add_argument("--batch-size", type=int, default=30)
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--output-dir", default="runs/mtda_spl")
    cli.add_argument("--cache-dir", default="runs/feature_cache")
    cli.add_argument("--retry-failed", action="store_true", help="resume incomplete runs with checkpoints")
    cli.add_argument("--dry-run", action="store_true")
    args = cli.parse_args()
    if len(set(args.devices)) != len(args.devices) or len(set(args.seeds)) != len(args.seeds):
        cli.error("devices and seeds must be unique")
    from mtda.data import discover_domains
    from mtda.experiment import file_sha
    import hashlib
    domains = discover_domains(args.data_root)
    common = ["--data-root", args.data_root, "--clip-checkpoint", args.clip_checkpoint,
              "--steps", str(args.steps), "--batch-size", str(args.batch_size),
              "--workers", str(args.workers), "--output-dir", args.output_dir,
              "--cache-dir", args.cache_dir]
    jobs = list(experiment_matrix(domains, args.seeds))
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    checkpoint_sha = file_sha(args.clip_checkpoint)
    code_files = [PROJECT / "train_mtda.py", *sorted((PROJECT / "mtda").glob("*.py"))]
    code_digest = hashlib.sha256("".join(file_sha(p) for p in code_files).encode()).hexdigest()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "matrix.log").open("a", buffering=1) as log:
        prepare = [sys.executable, "train_mtda.py", *common, "--device", args.devices[0], "--prepare-cache-only"]
        log.write(json.dumps({"prepare_command": prepare}) + "\n")
        subprocess.run(prepare, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
    devices = queue.Queue()
    for device in args.devices:
        devices.put(device)
    event_lock = threading.Lock()
    matrix_started = time.monotonic()
    completed = set()
    first_estimates = {}

    def record_event(event):
        with event_lock:
            with (root / "matrix_events.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")

    def execute(job):
        variant, source, target, seed = job
        run = root / variant / source.replace(" ", "_")
        if target:
            run /= target.replace(" ", "_")
        run /= f"seed_{seed}"
        if (run / "metrics.json").exists():
            metrics = json.loads((run / "metrics.json").read_text())
            cfg = metrics["config"]
            if (metrics["status"] != "complete" or cfg["steps"] != args.steps
                    or cfg["batch_size"] != args.batch_size or cfg["checkpoint_sha256"] != checkpoint_sha
                    or cfg["data_root"] != str(Path(args.data_root).resolve())
                    or cfg["implementation_sha256"] != code_digest
                    or cfg["source"] != source or cfg["variant"] != variant or cfg["seed"] != seed):
                raise ValueError(f"existing run has different configuration: {run}")
            print(f"SKIP completed {run}", flush=True)
            with event_lock:
                completed.add(job)
            return 0
        command = [sys.executable, "train_mtda.py", *common, "--variant", variant,
                   "--source", source, "--seed", str(seed)]
        if target:
            command += ["--target", target]
        if (run / "config.json").exists():
            if not args.retry_failed:
                raise FileExistsError(f"incomplete run needs --retry-failed: {run}")
            if (run / "last.pt").exists():
                command.append("--resume")
            else:
                # A failure before the first checkpoint has no recoverable state.
                archive = root / "failed_attempts" / f"{run.relative_to(root).as_posix().replace('/', '_')}_{time.time_ns()}"
                archive.parent.mkdir(parents=True, exist_ok=True)
                run.rename(archive)
                print(f"RESTART {job}; preserved pre-checkpoint attempt in {archive}", flush=True)
        device = devices.get()
        try:
            command += ["--device", device]
            log_path = root / f"{variant}_{source.replace(' ', '_')}_{(target or 'joint').replace(' ', '_')}_{seed}.log"
            with log_path.open("a", buffering=1) as log:
                log.write(json.dumps({"command": command}) + "\n")
                print(f"START {job} on {device}", flush=True)
                record_event({"event": "start", "job": job, "device": device,
                              "time": datetime.now(timezone.utc).isoformat()})
                process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                estimate_reported = False
                for line in process.stdout:
                    log.write(line)
                    progress = re.search(r"step (\d+)/(\d+):.*elapsed=([\d.]+)s", line)
                    if progress and int(progress[1]) >= 20 and not estimate_reported:
                        step, total, elapsed = int(progress[1]), int(progress[2]), float(progress[3])
                        remaining = elapsed / step * (total - step)
                        now = datetime.now(timezone(timedelta(hours=8)))
                        run_eta = now + timedelta(seconds=remaining)
                        branches = 2 if variant == "C" else 4
                        with event_lock:
                            first_estimates[job] = elapsed / step / branches
                            per_branch = sum(first_estimates.values()) / len(first_estimates)
                            remaining_work = sum(2 if j[0] == "C" else 4 for j in jobs if j not in completed)
                            matrix_remaining = remaining_work * args.steps * per_branch / len(args.devices)
                        matrix_eta = now + timedelta(seconds=matrix_remaining)
                        print(f"ETA {job}: run ≈{run_eta:%H:%M:%S}, matrix ≈{matrix_eta:%H:%M:%S} UTC+8 "
                              f"({elapsed / step:.3f}s/update; setup/evaluation excluded)", flush=True)
                        record_event({"event": "initial_eta", "job": job, "step": step,
                                      "seconds_per_update": elapsed / step,
                                      "run_eta": run_eta.isoformat(), "matrix_eta": matrix_eta.isoformat()})
                        estimate_reported = True
                returncode = process.wait()
            with event_lock:
                if returncode == 0:
                    completed.add(job)
            print(f"{'DONE' if returncode == 0 else 'FAILED'} {job}", flush=True)
            record_event({"event": "complete" if returncode == 0 else "failed", "job": job,
                          "returncode": returncode, "matrix_elapsed_seconds": time.monotonic() - matrix_started})
            return returncode
        finally:
            devices.put(device)

    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        results = list(pool.map(execute, jobs))
    subprocess.run([sys.executable, "scripts/summarize_mtda.py", "--runs-dir", args.output_dir],
                   cwd=PROJECT, check=True)
    if any(results):
        raise SystemExit("Some runs failed; inspect per-run logs and retry with --retry-failed")


if __name__ == "__main__":
    main()
