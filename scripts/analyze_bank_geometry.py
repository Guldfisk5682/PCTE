#!/usr/bin/env python3
"""Offline check: does centroid proximity select a teacher that predicts correctly?"""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from mtda.data import SourceImages, discover_domains
from mtda.experiment import write_json
from scripts.analyze_r4_agreement import verify_samples
from scripts.summarize_b_reliability import flat_csv


def analyze(root, source_text_path, cache_dir):
    cfg = json.loads((root/"config.json").read_text())
    state = torch.load(root/"last.pt",weights_only=False,map_location="cpu")
    trajectory = torch.load(source_text_path,weights_only=True,map_location="cpu")
    source_text = trajectory["texts"][cfg["steps"]]
    bank = state["registry"]["bank"]
    source_centroids = state["source_bank"]["running_means"]
    source_valid = state["source_bank"]["running_count"] > 0
    domains = discover_domains(cfg["data_root"])
    rows = []
    for target in cfg["targets"]:
        cache = torch.load(Path(cache_dir)/"domain_class"/f"{cfg['domain_cache_fingerprints'][target]}.pt",weights_only=True,map_location="cpu")
        data = SourceImages(domains[target],None,cfg["classnames"])
        verify_samples(data,cache)
        truth = data.labels
        others = [d for d in cfg["targets"] if d != target]
        references = ["Source",*others]
        texts = torch.stack([source_text,*[state["registry"]["snapshots"][d]["text"] for d in others]])
        votes = torch.einsum("bd,nkd->bnk",cache["z"],texts).argmax(-1)
        correct = votes.eq(truth[:,None])
        mismatch = ~correct[:,0] & correct[:,1:].any(-1)
        indices = [cfg["targets"].index(d) for d in others]
        centroids = torch.cat([source_centroids[None],bank["centroids"][indices]])
        valid = torch.cat([source_valid[None],bank["valid_mask"][indices]])
        # Only evaluate the true-class axis here, strictly in this offline program.
        selected = centroids.flatten(0,1)[valid.flatten()]
        computed = (cache["z_pre"].square().sum(-1)[:,None] + selected.square().sum(-1)[None]
                    - 2*cache["z_pre"]@selected.T).clamp_min(0)
        distances = torch.full((len(truth),len(references),len(cfg["classnames"])), torch.inf)
        distances.flatten(1)[:,valid.flatten()] = computed
        wrong_class = ~F.one_hot(truth,len(cfg["classnames"])).bool()
        for beta in (0.,.1,1.,10.):
            inputs = (-beta*distances).masked_fill(~valid[None],-torch.inf)
            fallback = ~valid[0]
            inputs[:,:,fallback] = -torch.inf
            inputs[:,0,fallback] = 0
            full_weights = inputs.softmax(1)
            weights = full_weights[torch.arange(len(truth)),:,truth]
            correct_mass = (weights*correct.float()).sum(1)
            rows.append({"source":cfg["source"],"target":target,"seed":cfg["seed"],"beta":beta,
                         "teacher_set":references,"samples":len(truth),
                         "source_prompt_accuracy":float(correct[:,0].float().mean()),
                         "other_prompt_accuracies":{d:float(correct[:,i+1].float().mean()) for i,d in enumerate(others)},
                         "source_wrong_other_teacher_correct_rate":float(mismatch.float().mean()),
                         "source_weight_on_true_class":float(weights[:,0].mean()),
                         "source_weight_on_wrong_classes":float(full_weights[:,0,:][wrong_class].mean()),
                         "source_weight_when_source_wrong_other_correct":float(weights[mismatch,0].mean()) if mismatch.any() else None,
                         "weighted_correct_teacher_vote_mass_at_true_class":float(correct_mass.mean()),
                         "correct_vote_mass_when_source_wrong_other_correct":float(correct_mass[mismatch].mean()) if mismatch.any() else None,
                         "scope":"offline true-class diagnostic; no label feedback; vote ability is not a calibrated probability"})
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs-dir",default="runs/b_followup/FBeta0")
    cli.add_argument("--source-text-dir",default="runs/b_followup_analysis/final_source")
    cli.add_argument("--cache-dir",default="runs/feature_cache")
    cli.add_argument("--output-dir",default="reports/b_followup")
    args = cli.parse_args()
    torch.set_num_threads(4)
    rows = []
    for path in sorted(Path(args.runs_dir).glob("*/seed_42/metrics.json")):
        cfg=json.loads(path.read_text())["config"]
        rows.extend(analyze(path.parent,Path(args.source_text_dir)/cfg["source"].replace(" ","_")/"seed_42.pt",args.cache_dir))
    output=Path(args.output_dir)
    write_json(output/"geometry_mismatch.json",rows)
    flat_csv(output/"geometry_mismatch.csv",rows)
    print(json.dumps({"offline_geometry_records":len(rows)}))


if __name__=="__main__":
    main()
