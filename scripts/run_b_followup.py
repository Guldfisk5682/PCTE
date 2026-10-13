#!/usr/bin/env python3
"""Event-driven fixed-teacher follow-up matrix with completion ETAs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import queue
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from mtda.data import discover_domains, resolve_domain
from mtda.experiment import file_sha
from mtda.reliability_followup.provenance import implementation_hashes
from mtda.reliability_followup.references import METHODS
VARIANTS = tuple(METHODS)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", required=True)
    cli.add_argument("--clip-checkpoint", required=True)
    cli.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    cli.add_argument("--sources", nargs="+", default=["Art", "Clipart", "Product", "Real World"])
    cli.add_argument("--seeds", nargs="+", type=int, default=[42])
    cli.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    cli.add_argument("--steps", type=int, default=1000)
    cli.add_argument("--refresh-interval", type=int, default=200)
    cli.add_argument("--beta", type=float, default=10.)
    cli.add_argument("--threshold", type=float, default=.7)
    cli.add_argument("--admission-fractions", nargs=3, type=float, default=[.25, .5, .75])
    cli.add_argument("--eval-every", type=int, default=200)
    cli.add_argument("--output-dir", default="runs/b_followup")
    cli.add_argument("--cache-dir", default="runs/feature_cache")
    cli.add_argument("--retry-failed", action="store_true")
    cli.add_argument("--reference-root", default="runs/b_reliability")
    args = cli.parse_args()
    _, implementation = implementation_hashes()
    weight_sha = file_sha(args.clip_checkpoint)
    domains = discover_domains(args.data_root)
    sources = [resolve_domain(d, domains) for d in args.sources]
    if len(set(args.devices)) != len(args.devices) or len(set(sources)) != len(sources):
        cli.error("devices and sources must be unique")
    jobs = [(variant, source, seed) for seed in args.seeds for variant in args.variants for source in sources]
    available = queue.Queue()
    for device in args.devices:
        available.put(device)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    base_command = [sys.executable, str(PROJECT / "train_b_followup.py"), "--data-root", str(Path(args.data_root).resolve()),
                    "--clip-checkpoint", str(Path(args.clip_checkpoint).resolve()), "--output-dir", str(root.resolve()),
                    "--cache-dir", str(Path(args.cache_dir).resolve()), "--steps", str(args.steps),
                    "--refresh-interval", str(args.refresh_interval), "--beta", str(args.beta),
                    "--threshold", str(args.threshold), "--eval-every", str(args.eval_every),
                    "--reference-root", str(Path(args.reference_root).resolve()),
                    "--admission-fractions", *[str(f) for f in args.admission_fractions]]
    # Prepare one immutable order per source before any formal target prompt updates.
    for source in sources:
        path = root / "orders" / f"{source.replace(' ', '_')}.json"
        if not path.exists():
            command = [*base_command, "--source", source, "--seed", "42", "--device", args.devices[0], "--prepare-order-only"]
            subprocess.run(command, cwd=PROJECT, check=True, stdout=subprocess.DEVNULL)
        record = json.loads(path.read_text())
        print(f"ORDER {source}: {' -> '.join(record['order'])}; {record['rule']}", flush=True)

    def execute(job):
        variant, source, seed = job
        run = root / variant / source.replace(" ", "_") / f"seed_{seed}"
        expected = {"variant": variant, "source": source, "seed": seed, "steps": args.steps,
                    "implementation_sha256": implementation, "checkpoint_sha256": weight_sha,
                    "data_root": str(Path(args.data_root).resolve()), "refresh_interval": args.refresh_interval,
                    "threshold": args.threshold, "beta": METHODS[variant].get("beta", 0.), "admission_fractions": args.admission_fractions,
                    "eval_every": args.eval_every}
        if (run / "metrics.json").exists():
            config = json.loads((run / "metrics.json").read_text())["config"]
            if any(config.get(k) != v for k, v in expected.items()):
                raise ValueError(f"completed configuration mismatch: {run}")
            print(f"SKIP {variant} {source} seed={seed}", flush=True)
            return 0
        device = available.get()
        try:
            command = [*base_command, "--variant", variant, "--source", source, "--seed", str(seed), "--device", device]
            if (run / "config.json").exists():
                if args.retry_failed and (run / "last.pt").exists():
                    command.append("--resume")
                else:
                    raise FileExistsError(f"unfinished run requires --retry-failed and a checkpoint: {run}")
            run.mkdir(parents=True, exist_ok=True)
            print(f"START {variant} {source} seed={seed} device={device}", flush=True)
            estimated = False
            with (run / "process.log").open("a", buffering=1) as handle:
                process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1)
                for line in process.stdout:
                    handle.write(line)
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") == "progress" and event["step"] >= 20 and not estimated:
                        seconds = event["elapsed_seconds"] / event["step"] * (event["total_steps"] - event["step"])
                        eta = datetime.now(timezone(timedelta(hours=8))) + timedelta(seconds=seconds)
                        print(f"ETA {variant} {source} seed={seed}: ~{eta:%m-%d %H:%M:%S} UTC+8", flush=True)
                        estimated = True
                code = process.wait()
            print(f"{'DONE' if code == 0 else 'FAILED'} {variant} {source} seed={seed}", flush=True)
            if code:
                print((run / "process.log").read_text()[-3000:], flush=True)
            return code
        finally:
            available.put(device)
    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        codes = list(pool.map(execute, jobs))
    if any(codes):
        raise SystemExit("matrix failed; inspect preserved logs")


if __name__ == "__main__":
    main()
