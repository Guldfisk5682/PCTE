#!/usr/bin/env python3
"""Frozen RN50, one source and three simultaneous targets: official VLM DA adapters."""

import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def parser(default_method="DAPrompt"):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--method",choices=["DAPrompt","DAPrompt-CodeTau","ADCLIP-Code","ADCLIP-Registered"],default=default_method)
    cli.add_argument("--source",required=True)
    cli.add_argument("--data-root",required=True)
    cli.add_argument("--clip-checkpoint",required=True)
    cli.add_argument("--device",default="cuda:0")
    cli.add_argument("--seed",type=int,default=42)
    cli.add_argument("--steps",type=int,default=1000)
    cli.add_argument("--batch-size",type=int,default=30)
    cli.add_argument("--output-dir",default="runs/vlm_competitors")
    cli.add_argument("--cache-dir",default="runs/feature_cache")
    cli.add_argument("--adclip-repo",default="/home/tangxunchi/workspace/AD-CLIP")
    cli.add_argument("--daprompt-repo",default="/home/tangxunchi/workspace/DAPrompt")
    cli.add_argument("--text-microbatch",type=int,default=16)
    cli.add_argument("--feature-batch-size",type=int,default=128)
    cli.add_argument("--eval-batch-size",type=int,default=512)
    cli.add_argument("--workers",type=int,default=4)
    cli.add_argument("--cpu-threads",type=int,default=4)
    cli.add_argument("--eval-every",type=int,default=0,help="zero: initial and final; expensive image-conditioned AD-CLIP evaluation")
    cli.add_argument("--evaluation-limit",type=int,default=0,help="positive: smoke-only first N samples per target; never a formal result")
    cli.add_argument("--log-every",type=int,default=100)
    cli.add_argument("--checkpoint-every",type=int,default=100)
    cli.add_argument("--prepare-cache-only",action="store_true")
    cli.add_argument("--resume",action="store_true")
    cli.add_argument("--stop-after",type=int)
    return cli


def main(default_method="DAPrompt"):
    cli=parser(default_method)
    args=cli.parse_args()
    for key in ("steps","batch_size","text_microbatch","feature_batch_size","eval_batch_size","cpu_threads","log_every","checkpoint_every"):
        if getattr(args,key)<1:
            cli.error(f"--{key.replace('_','-')} must be positive")
    if args.workers<0 or args.eval_every<0 or args.evaluation_limit<0:
        cli.error("worker and evaluation intervals cannot be negative")
    from mtda.competitors.experiment import run
    run(args)


if __name__=="__main__":
    main()
