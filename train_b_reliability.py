#!/usr/bin/env python3
"""Controlled B-only E2H, cross-target teacher, and target-bank experiments."""

import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--variant", choices=["R0", "R0M", "R1", "R2", "R3", "R4", "R5", "R6"], default="R0M")
    cli.add_argument("--source", required=True)
    cli.add_argument("--data-root", required=True)
    cli.add_argument("--clip-checkpoint", required=True)
    cli.add_argument("--device", default="cuda:0")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--steps", type=int, default=1000, help="effective optimizer updates per target and source")
    cli.add_argument("--batch-size", type=int, default=30)
    cli.add_argument("--m1", type=int, default=16)
    cli.add_argument("--m2", type=int, default=16)
    cli.add_argument("--learning-rate", type=float, default=0.003)
    cli.add_argument("--lambda-target", type=float, default=0.5)
    cli.add_argument("--threshold", type=float, default=0.7, help="fixed first-round bank confidence threshold")
    cli.add_argument("--beta", type=float, default=10.0)
    cli.add_argument("--refresh-interval", type=int, default=200)
    cli.add_argument("--admission-fractions", nargs=3, type=float, default=[0.25, 0.5, 0.75])
    cli.add_argument("--target-order", nargs=3, help="manual fixed order; default: unlabeled source-teacher entropy")
    cli.add_argument("--order-file")
    cli.add_argument("--order-teacher-root", default="runs/source_prompt_only")
    cli.add_argument("--order-teacher-seed", type=int, default=42)
    cli.add_argument("--prepare-order-only", action="store_true")
    cli.add_argument("--output-dir", default="runs/b_reliability",
                     help="separate result tree so original recursive A/B/C summarizers stay compatible")
    cli.add_argument("--r0-runs-dir", default="runs/mtda_spl")
    cli.add_argument("--cache-dir", default="runs/feature_cache")
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--cpu-threads", type=int, default=4)
    cli.add_argument("--feature-batch-size", type=int, default=128)
    cli.add_argument("--eval-batch-size", type=int, default=512)
    cli.add_argument("--eval-every", type=int, default=200)
    cli.add_argument("--log-every", type=int, default=100)
    cli.add_argument("--checkpoint-every", type=int, default=200)
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--stop-after", type=int, help="global step to pause at; sequential total is 3 * steps")
    return cli


def main():
    cli = parser()
    args = cli.parse_args()
    for name in ("steps", "batch_size", "m1", "m2", "refresh_interval", "cpu_threads", "feature_batch_size",
                 "eval_batch_size", "eval_every", "log_every", "checkpoint_every"):
        if getattr(args, name) < 1:
            cli.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.learning_rate <= 0 or args.lambda_target < 0 or args.beta < 0 or not 0 < args.threshold <= 1:
        cli.error("invalid worker, learning rate, loss scale, beta or threshold")
    if args.stop_after is not None and args.stop_after < 1:
        cli.error("--stop-after must be positive")
    if args.variant == "R0":
        # R0 is literally the unchanged upstream-compatible B entry, never a redefinition.
        from pathlib import Path
        import json
        from mtda.data import discover_domains, resolve_domain
        from mtda.experiment import run_experiment, file_sha
        from mtda.reliability.experiment import implementation_hashes
        source = resolve_domain(args.source, discover_domains(args.data_root))
        root = Path(args.r0_runs_dir) / "B" / source.replace(" ", "_") / f"seed_{args.seed}"
        if (root / "metrics.json").exists():
            cfg = json.loads((root / "metrics.json").read_text())["config"]
            baseline, _ = implementation_hashes()
            checks = {"steps": args.steps, "batch_size": args.batch_size, "m1": args.m1, "m2": args.m2,
                      "learning_rate": args.learning_rate, "lambda_target": args.lambda_target,
                      "implementation_sha256": baseline, "checkpoint_sha256": file_sha(args.clip_checkpoint),
                      "data_root": str(Path(args.data_root).resolve())}
            if any(cfg[k] != v for k, v in checks.items()):
                cli.error("preserved R0 configuration mismatch")
            print(json.dumps({"event": "preserved_r0", "run": str(root)}))
            return
        args.variant, args.target, args.output_dir = "B", [], args.r0_runs_dir
        args.smoke, args.prepare_cache_only = False, False
        run_experiment(args)
    else:
        from mtda.reliability.experiment import run_experiment
        run_experiment(args)


if __name__ == "__main__":
    main()
