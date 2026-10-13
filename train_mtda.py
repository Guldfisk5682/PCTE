#!/usr/bin/env python3
"""Train an A/B joint baseline or one independent C pair."""

import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--variant", choices=["A", "B", "C"], default="A")
    cli.add_argument("--source", help="one labeled Office-Home domain")
    cli.add_argument("--target", nargs="+", default=[], help="default: other three domains; C requires one")
    cli.add_argument("--data-root", required=True, help="directory containing four domain folders")
    cli.add_argument("--clip-checkpoint", required=True, help="local official CLIP checkpoint, e.g. RN50.pt")
    cli.add_argument("--device", default="cuda:0")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--steps", type=int, default=1000)
    cli.add_argument("--batch-size", type=int, default=30)
    cli.add_argument("--m1", type=int, default=16)
    cli.add_argument("--m2", type=int, default=16)
    cli.add_argument("--learning-rate", type=float, default=0.003)
    cli.add_argument("--lambda-target", type=float, default=0.5)
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--cpu-threads", type=int, default=4)
    cli.add_argument("--feature-batch-size", type=int, default=128)
    cli.add_argument("--eval-batch-size", type=int, default=128)
    cli.add_argument("--cache-dir", default="runs/feature_cache")
    cli.add_argument("--output-dir", default="runs/mtda_spl")
    cli.add_argument("--checkpoint-every", type=int, default=100)
    cli.add_argument("--log-every", type=int, default=20)
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--stop-after", type=int, help="save and stop at this update without final evaluation")
    cli.add_argument("--prepare-cache-only", action="store_true", help="extract all four frozen image caches")
    cli.add_argument("--smoke", action="store_true", help="allow a single target for an A/B smoke test")
    return cli


def main():
    cli = parser()
    args = cli.parse_args()
    if not args.source and not args.prepare_cache_only:
        cli.error("--source is required for training")
    for name in ("steps", "batch_size", "m1", "m2", "cpu_threads", "feature_batch_size",
                 "eval_batch_size", "checkpoint_every", "log_every"):
        if getattr(args, name) < 1:
            cli.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.learning_rate <= 0 or args.lambda_target < 0:
        cli.error("workers and lambda-target must be nonnegative; learning-rate must be positive")
    if args.stop_after is not None and args.stop_after < 1:
        cli.error("--stop-after must be positive")
    from mtda.experiment import run_experiment
    try:
        run_experiment(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    main()
