#!/usr/bin/env python3
"""Restore into fresh models; verify all saved tensors and spread-out predictions."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
import torch
from torch.nn import functional as F
from clip_custom import clip
from mtda.competitors.experiment import load_adapter
from mtda.experiment import seed_everything,write_json


@torch.no_grad()
def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--runs-dir',default='runs/vlm_competitors')
    cli.add_argument('--cache-dir',default='runs/feature_cache')
    cli.add_argument('--device',default='cuda:0')
    cli.add_argument('--methods',nargs='+',default=['DAPrompt','DAPrompt-CodeTau','ADCLIP-Code','ADCLIP-Registered'])
    cli.add_argument('--output-file',default='reports/vlm_competitors/fresh_checkpoint_verification.json')
    args=cli.parse_args();torch.set_num_threads(4)
    rows=[]
    for method in args.methods:
        for path in sorted((Path(args.runs_dir)/method).glob('*/seed_42/metrics.json')):
            cfg=json.loads(path.read_text())['config']
            state=torch.load(path.parent/'last.pt',weights_only=False,map_location='cpu')
            saved_predictions=torch.load(path.parent/'final_predictions.pt',weights_only=True,map_location='cpu')
            seed_everything(cfg['seed']+12345)
            frozen,_=clip.load(cfg['clip_checkpoint'],device='cpu',jit=False)
            frozen=frozen.float().eval().requires_grad_(False)
            root=Path(cfg['upstream_repository']);conditional=method.startswith('ADCLIP')
            if conditional:
                model=load_adapter(root,'ADCLIPMTDA')(frozen,clip,cfg['classnames'],cfg['source'],cfg['targets'],method=='ADCLIP-Registered')
            else:
                model=load_adapter(root,'DAPromptMTDA')(frozen,clip,cfg['classnames'],cfg['source'],cfg['targets'],cfg['threshold'])
            if all(torch.equal(v,state['prompts'][k]) for k,v in model.prompt_learner.state_dict().items()):
                raise AssertionError('fresh model unexpectedly matches trained initialization')
            model.prompt_learner.load_state_dict(state['prompts'])
            if any(not torch.equal(v,state['prompts'][k]) for k,v in model.prompt_learner.state_dict().items()):
                raise AssertionError('a learned/fixed prompt tensor was not restored exactly')
            model=model.to(args.device).eval()
            checks={}
            for domain in model.targets:
                fingerprint=cfg['cache_fingerprints'][domain]
                if conditional:
                    stat=torch.load(Path(args.cache_dir)/'adclip_statistics'/f'{fingerprint}.pt',weights_only=True,map_location='cpu')
                    raw_key=stat['metadata']['image_feature_fingerprint'];statistics=stat['statistics']
                else:
                    raw_key=fingerprint;statistics=None
                raw=torch.load(Path(args.cache_dir)/f'{raw_key}.pt',weights_only=True,map_location='cpu')['features']
                # Spread across the domain, while preserving original AD text
                # batch shapes; this reaches many classes rather than the first folder.
                usable=len(raw)-len(raw)%cfg['text_microbatch'] if conditional else len(raw)
                ids=torch.linspace(0,usable-1,96).long()
                if conditional:
                    outputs=[]
                    for ix in ids.split(cfg['text_microbatch']):
                        outputs.append(model.logits(raw[ix].to(args.device),[s[ix].to(args.device) for s in statistics],cfg['text_microbatch']).argmax(-1).cpu())
                    actual=torch.cat(outputs)
                else:
                    text=model.evaluation_text(domain)
                    actual=(model.scale*F.normalize(raw[ids].to(args.device),dim=-1)@text.T).argmax(-1).cpu()
                disagreement=int(actual.ne(saved_predictions[domain][ids]).sum())
                if disagreement:
                    raise AssertionError(f'fresh checkpoint prediction mismatch: {method} {cfg["source"]} {domain}: {disagreement}/96')
                checks[domain]={'samples':len(ids),'prediction_disagreements':disagreement}
            rows.append({'method':method,'source':cfg['source'],'all_saved_tensors_exact':True,'different_fresh_initialization':True,'targets':checks})
            print(json.dumps(rows[-1]),flush=True)
            del model,frozen
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
    write_json(Path(args.output_file),{'device':args.device,'runs_verified':len(rows),'rows':rows})


if __name__=='__main__':
    main()
