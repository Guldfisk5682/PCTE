#!/usr/bin/env python3
"""Fixed-R4-snapshot distance-temperature and admission controls."""

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
    from mtda.reliability_followup.references import METHODS
    cli = parser()
    for action in cli._actions:
        if action.dest == "variant":
            action.choices = list(METHODS)
            action.default = "FUniform"
        if action.dest == "output_dir":
            action.default = "runs/b_followup"
    cli.add_argument("--reference-root", default="runs/b_reliability",
                     help="completed R4 snapshots; never target-label-selected")
    args = cli.parse_args()
    specification = METHODS[args.variant]
    args.beta = specification.get("beta", 0.)
    for name in ("steps", "batch_size", "m1", "m2", "cpu_threads", "feature_batch_size",
                 "eval_batch_size", "eval_every", "log_every", "checkpoint_every", "refresh_interval"):
        if getattr(args, name) < 1:
            cli.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.threshold != 0.7 or args.learning_rate <= 0 or args.lambda_target < 0:
        cli.error("follow-up retains threshold 0.7 and valid learning rate/loss scale")
    if args.stop_after is not None and args.stop_after < 1:
        cli.error("--stop-after must be positive")
    from mtda.reliability_followup.experiment import run_experiment
    run_experiment(args)


if __name__ == "__main__":
    main()
