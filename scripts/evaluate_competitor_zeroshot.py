#!/usr/bin/env python3
"""Independent offline check of native DAPL and common zero-shot templates."""
import argparse
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
import torch
from torch.nn import functional as F
from clip_custom import clip
from mtda.data import SourceImages,discover_domains
from mtda.experiment import file_sha,write_json


@torch.no_grad()
def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference-config',default='runs/b_reliability/R4/Art/seed_42/config.json')
    cli.add_argument('--cache-dir',default='runs/feature_cache')
    cli.add_argument('--output-dir',default='reports/vlm_competitors')
    args=cli.parse_args()
    cfg=json.loads(Path(args.reference_config).read_text())
    torch.set_num_threads(4)
    model,_=clip.load(cfg['clip_checkpoint'],device='cpu',jit=False)
    model=model.float().eval().requires_grad_(False)
    if file_sha(cfg['clip_checkpoint'])!=cfg['checkpoint_sha256']:
        raise ValueError('weight provenance changed')
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    domains=discover_domains(cfg['data_root'])
    results={}
    templates={'CLIP Zero-shot':'A photo of a {}','CLIP Zero-shot (DAPL template)':'a photo of a {}.'}
    for method,template in templates.items():
        tokens=clip.tokenize([template.format(n.replace('_',' ')) for n in cfg['classnames']])
        text=F.normalize(model.encode_text(tokens).float(),dim=-1)
        per_domain={}
        for domain,path in domains.items():
            fingerprint=cfg['feature_fingerprints'][domain]
            cache=torch.load(Path(args.cache_dir)/f'{fingerprint}.pt',weights_only=True,map_location='cpu')
            if cache['fingerprint']!=fingerprint:
                raise ValueError('visual cache provenance changed')
            raw=cache['features']
            truth=SourceImages(path,None,cfg['classnames']).labels
            pred=(model.logit_scale.exp()*F.normalize(raw,dim=-1)@text.T).argmax(-1)
            per_domain[domain]={'accuracy':float(pred.eq(truth).double().mean()),'samples':len(truth)}
        results[method]={'template':template,'domains':per_domain,
            'macro_accuracy':sum(r['accuracy'] for r in per_domain.values())/4}
    write_json(output/'zero_shot_templates.json',{'checkpoint_sha256':cfg['checkpoint_sha256'],
        'cache_fingerprints':cfg['feature_fingerprints'],'offline_only':True,'results':results})
    print(json.dumps(results))


if __name__=='__main__':
    main()
