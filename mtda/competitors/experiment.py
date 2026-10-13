"""Shared budget/data/evaluation harness; algorithms remain in their cloned repositories."""

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

from mtda.data import BatchStream, SourceImages, UnlabeledImages, discover_domains, resolve_domain, validate_classes
from mtda.experiment import PROJECT, file_sha, seed_everything, rng_state, restore_rng, write_json
from mtda.reliability.experiment import implementation_hashes as preserved_hashes
from .cache import frozen_features
from .offline import evaluate

UPSTREAM = {"AD-CLIP": "06a30ce32ab7a0dcc29a5ac72e66c4e868f70c89",
            "DAPrompt": "f24b0c16c06b585104ba97615f89cd5df4def872"}
HISTORY = ("method","source","seed","step","source_updates","target_updates","source_samples",
           "target_samples","source_ce_views","optimizer_steps","learning_rate","total_loss","components","elapsed_seconds")


def code_hash(args, root):
    baseline, reliability = preserved_hashes()
    if reliability != "f33008a054906a428f079d001a7eb84df87a97c8bd6789d231662b39c6ff5bb2":
        raise ValueError("preserved SPL reliability runtime changed")
    sha = subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    if sha != UPSTREAM[root.name]:
        raise ValueError("external repository HEAD differs from audited official version")
    subprocess.run(["git","diff","--exit-code",sha,"--","trainers","clip","configs","datasets"],cwd=root,check=True,stdout=subprocess.DEVNULL)
    files = [PROJECT/"train_vlm_competitors.py",*sorted(Path(__file__).parent.glob("*.py")),
             *sorted((root/"mtda_adapter").glob("*.py")),root/"train_mtda.py"]
    implementation = hashlib.sha256((baseline+reliability+sha+"".join(file_sha(p) for p in files)).encode()).hexdigest()
    return sha,implementation,baseline,reliability


def load_adapter(root, class_name):
    spec = importlib.util.spec_from_file_location(f"external_{class_name}",root/"mtda_adapter/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module,class_name)


def batch_seed(seed,domain):
    return int.from_bytes(hashlib.sha256(f"{seed}:batch:{domain}".encode()).digest()[:8],"big") % (2**63)


def learning_rate(step,steps,warmup,maximum):
    if step < warmup:
        return 1e-5+(maximum-1e-5)*step/max(1,warmup)
    return maximum*.5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))


def append(path,record):
    with path.open("a",buffering=1) as handle:
        handle.write(json.dumps(record,ensure_ascii=False)+"\n")


def run(args):
    from clip_custom import clip
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    conditional = args.method.startswith("ADCLIP")
    repository = Path(args.adclip_repo if conditional else args.daprompt_repo).resolve()
    upstream,implementation,baseline,reliability = code_hash(args,repository)
    domains=discover_domains(args.data_root)
    source=resolve_domain(args.source,domains)
    targets=[d for d in domains if d != source]
    names=validate_classes(domains,source)
    checkpoint=Path(args.clip_checkpoint).resolve()
    checkpoint_sha=file_sha(checkpoint)
    clip_model,preprocess=clip.load(str(checkpoint),device="cpu",jit=False)
    clip_model=clip_model.float().eval().requires_grad_(False)
    if conditional:
        model=load_adapter(repository,"ADCLIPMTDA")(clip_model,clip,names,source,targets,args.method=="ADCLIP-Registered")
    else:
        threshold=.5 if args.method=="DAPrompt-CodeTau" else .6
        model=load_adapter(repository,"DAPromptMTDA")(clip_model,clip,names,source,targets,threshold)
    model=model.to(args.device)
    caches={d:frozen_features(UnlabeledImages(path,preprocess),model.clip,args.cache_dir,checkpoint_sha,
                             args.feature_batch_size,args.workers,conditional) for d,path in domains.items()}
    if args.prepare_cache_only:
        print(json.dumps({"event":"cache_ready","method":args.method,"fingerprints":{d:c["fingerprint"] for d,c in caches.items()}}),flush=True)
        return
    # Only Source labels enter training. Target streams are opaque sample indices.
    source_labels=SourceImages(domains[source],None,names).labels
    streams={d:BatchStream(torch.arange(len(c["features"]))[:,None],args.batch_size,batch_seed(args.seed,d),
                           source_labels if d==source else None) for d,c in caches.items()}
    maximum_lr=.01 if conditional else .003
    warmup=min(max(1,len(source_labels)//args.batch_size),max(1,args.steps//2))
    parameters=[p for p in model.prompt_learner.parameters() if p.requires_grad]
    optimizer=(torch.optim.Adam(parameters,lr=1e-5,weight_decay=5e-4) if conditional else
               torch.optim.SGD(parameters,lr=1e-5,momentum=.9,weight_decay=5e-4))
    config={"method":args.method,"source":source,"targets":targets,"seed":args.seed,"steps":args.steps,
            "batch_size":args.batch_size,"backbone":"RN50","precision":"float32",
            "preprocess":"deterministic official CLIP resize/center crop; identical to SPL comparison",
            "data_root":str(Path(args.data_root).resolve()),"clip_checkpoint":str(checkpoint),
            "checkpoint_sha256":checkpoint_sha,"classnames":names,"cache_fingerprints":{d:c["fingerprint"] for d,c in caches.items()},
            "upstream_sha":upstream,"upstream_repository":str(repository),"implementation_sha256":implementation,
            "preserved_baseline_sha256":baseline,"preserved_reliability_sha256":reliability,
            "optimizer":"Adam" if conditional else "SGD","learning_rate":maximum_lr,"weight_decay":5e-4,
            "momentum":None if conditional else .9,"scheduler":"one Source epoch warmup, then per-update cosine to fixed budget",
            "warmup_steps":warmup,"target_aggregation":"equal mean of three native target/pair losses",
            "source_ce_views_per_step":len(targets) if conditional else 1,
            "source_budget":args.steps,"target_budgets":{d:args.steps for d in targets},
            "target_training_ground_truth":False,"checkpoint_selection":"fixed final budget",
            "evaluation_limit":args.evaluation_limit,"eval_every":args.eval_every,"text_microbatch":args.text_microbatch,
            "learnable_parameters":sum(p.numel() for p in parameters),
            "frozen_random_projectors":conditional and args.method=="ADCLIP-Code",
            "style_mapping_training_input":"Source (official code)" if conditional else None,
            "inference":"target-only mapped style + own target content" if conditional else "own target domain prompt only",
            "threshold":None if conditional else model.threshold,
            "loss":"CE + 0.1 Source mapping MSE + mean(KL + 0.01 batch-L2 entropy)" if conditional else
                   "Source CE + mean(accepted zero-shot hard pseudo-label CE); U=1,T=1"}
    root=Path(args.output_dir)/args.method/source.replace(" ","_")/f"seed_{args.seed}"
    if (root/"metrics.json").exists():
        raise FileExistsError(f"completed competitor run preserved: {root}")
    if (root/"config.json").exists() and not args.resume:
        raise FileExistsError(f"unfinished run needs explicit resume: {root}")
    root.mkdir(parents=True,exist_ok=True)
    step,prior_elapsed=0,0.
    if args.resume:
        state=torch.load(root/"last.pt",weights_only=False,map_location="cpu")
        if state["config"]!=config:
            raise ValueError("resume competitor configuration changed")
        model.prompt_learner.load_state_dict(state["prompts"])
        optimizer.load_state_dict(state["optimizer"])
        for d,s in streams.items():
            s.load_state_dict(state["streams"][d])
        restore_rng(state["rng"])
        step,prior_elapsed=state["step"],state["elapsed_seconds"]
        # Checkpoints are atomic; discard rows written after the restored step.
        if (root/"history.csv").exists():
            rows=[r for r in csv.DictReader((root/"history.csv").open()) if int(r["step"])<=step]
            with (root/"history.csv").open("w",newline="") as h:
                writer=csv.DictWriter(h,fieldnames=HISTORY);writer.writeheader();writer.writerows(rows)
    else:
        write_json(root/"config.json",config)
        torch.save(model.prompt_learner.state_dict(),root/"initialization.pt")
    write_json(root/"environment.json",{"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,
        "device":args.device,"gpu":torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "command":[sys.executable,*sys.argv]})
    started=time.monotonic()
    def elapsed():
        return prior_elapsed+time.monotonic()-started
    def diagnose(current,event):
        results,predictions=evaluate(model,domains,caches,names,args.eval_batch_size,args.text_microbatch,args.evaluation_limit)
        append(root/"evaluations.jsonl",{"method":args.method,"source":source,"seed":args.seed,"step":current,
            "event":event,"targets":results,"elapsed_seconds":elapsed(),"evaluation_limit":args.evaluation_limit})
        return results,predictions
    def save(current):
        state={"prompts":model.prompt_learner.state_dict(),"optimizer":optimizer.state_dict(),"step":current,
               "streams":{d:s.state_dict() for d,s in streams.items()},"config":config,"rng":rng_state(),"elapsed_seconds":elapsed()}
        temporary=root/"last.tmp"
        torch.save(state,temporary);temporary.replace(root/"last.pt")
    if not step:
        diagnose(0,"initial")
    stop=min(args.steps,args.stop_after) if args.stop_after else args.steps
    with (root/"history.csv").open("a",newline="",buffering=1) as handle:
        writer=csv.DictWriter(handle,fieldnames=HISTORY)
        if handle.tell()==0:
            writer.writeheader()
        for current in range(step+1,stop+1):
            source_indices,labels=streams[source].next_batch()
            indices={source:source_indices.flatten().long(),**{d:streams[d].next_batch().flatten().long() for d in targets}}
            raw={d:caches[d]["features"][ix].to(args.device) for d,ix in indices.items()}
            labels=labels.to(args.device)
            lr=learning_rate(current-1,args.steps,warmup,maximum_lr)
            for group in optimizer.param_groups:
                group["lr"]=lr
            optimizer.zero_grad(set_to_none=True)
            model.train()
            if conditional:
                stats={d:[s[ix].to(args.device) for s in caches[d]["statistics"]] for d,ix in indices.items()}
                value,components=model.backward_loss((raw[source],labels),{d:raw[d] for d in targets},stats[source],
                                                    {d:stats[d] for d in targets},args.text_microbatch)
            else:
                loss,components=model.loss((raw[source],labels),{d:raw[d] for d in targets})
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite DAPL training loss")
                loss.backward();value=float(loss.detach())
            if any(p.grad is not None for p in model.clip.parameters()):
                raise AssertionError("frozen CLIP received gradients")
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError("nonfinite competitor prompt/projector gradient")
            optimizer.step()
            writer.writerow({"method":args.method,"source":source,"seed":args.seed,"step":current,
                "source_updates":current,"target_updates":json.dumps({d:current for d in targets}),
                "source_samples":current*args.batch_size,"target_samples":json.dumps({d:current*args.batch_size for d in targets}),
                "source_ce_views":current*args.batch_size*(len(targets) if conditional else 1),
                "optimizer_steps":current,"learning_rate":lr,"total_loss":value,"components":json.dumps(components),"elapsed_seconds":elapsed()})
            if current in (1,5,20) or current%args.log_every==0 or current==stop:
                print(json.dumps({"event":"progress","method":args.method,"source":source,"step":current,"total_steps":args.steps,
                    "elapsed_seconds":elapsed(),"loss":value,"components":components}),flush=True)
            if args.eval_every and current%args.eval_every==0 and current != args.steps:
                diagnose(current,"interval")
            if current%args.checkpoint_every==0 or current==stop:
                save(current)
    if stop==args.steps:
        results,predictions=diagnose(stop,"final")
        # Restore into the model and reproduce a small fixed prediction sample,
        # without running expensive full AD-CLIP evaluation twice.
        reload=torch.load(root/"last.pt",weights_only=False,map_location="cpu")
        model.prompt_learner.load_state_dict(reload["prompts"])
        sample_results,sample_predictions=evaluate(model,domains,caches,names,args.eval_batch_size,args.text_microbatch,
                                                   min(args.evaluation_limit or 37,37))
        for d in targets:
            if not torch.equal(sample_predictions[d],predictions[d][:len(sample_predictions[d])]):
                raise AssertionError("checkpoint restoration changes classifier predictions")
        metrics={"status":"smoke" if args.evaluation_limit else "complete","config":config,"step":stop,"targets":results,
                 "macro_accuracy":sum(r["accuracy"] for r in results.values())/len(targets),
                 "effective_updates":{d:stop for d in [source,*targets]},"sample_exposures":{d:stop*args.batch_size for d in [source,*targets]},
                 "optimizer_step_calls":stop,"source_ce_sample_views":stop*args.batch_size*(len(targets) if conditional else 1),
                 "training_seconds":elapsed(),"checkpoint_prediction_roundtrip":True,
                 "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else None}
        torch.save(predictions,root/"final_predictions.pt")
        write_json(root/"metrics.json",metrics)
        print(json.dumps({"event":"complete","run":str(root),"macro_accuracy":metrics["macro_accuracy"]}),flush=True)
    else:
        print(json.dumps({"event":"paused","run":str(root),"step":stop}),flush=True)
