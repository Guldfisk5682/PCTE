#!/usr/bin/env python3
"""Event-driven, one process per GPU, fixed-budget VLM MTDA comparison."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from mtda.competitors.experiment import code_hash
from mtda.data import discover_domains,resolve_domain
from mtda.experiment import file_sha,write_json

METHODS=("DAPrompt","DAPrompt-CodeTau","ADCLIP-Code","ADCLIP-Registered")
ZONE=timezone(timedelta(hours=8))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data-root',required=True)
    cli.add_argument('--clip-checkpoint',required=True)
    cli.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1'])
    cli.add_argument('--sources',nargs='+',default=['Art','Clipart','Product','Real World'])
    cli.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    cli.add_argument('--seed',type=int,default=42)
    cli.add_argument('--steps',type=int,default=1000)
    cli.add_argument('--batch-size',type=int,default=30)
    cli.add_argument('--text-microbatch',type=int,default=16)
    cli.add_argument('--output-dir',default='runs/vlm_competitors')
    cli.add_argument('--cache-dir',default='runs/feature_cache')
    cli.add_argument('--adclip-repo',default='/home/tangxunchi/workspace/AD-CLIP')
    cli.add_argument('--daprompt-repo',default='/home/tangxunchi/workspace/DAPrompt')
    cli.add_argument('--resume-failed',action='store_true')
    args=cli.parse_args()
    domains=discover_domains(args.data_root)
    sources=[resolve_domain(s,domains) for s in args.sources]
    if len(set(args.devices))!=len(args.devices) or len(set(sources))!=len(sources):
        cli.error('sources and devices must be unique')
    if min(args.steps,args.batch_size,args.text_microbatch)<1:
        cli.error('budgets and batch sizes must be positive')
    hashes={m:code_hash(args,Path(args.adclip_repo if m.startswith('ADCLIP') else args.daprompt_repo))[1] for m in args.methods}
    checkpoint_sha=file_sha(args.clip_checkpoint)
    root=Path(args.output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    jobs=[(m,s) for m in args.methods for s in sources]
    devices=queue.Queue()
    for d in args.devices:
        devices.put(d)
    lock=threading.Lock();states={}
    def update(key,**values):
        with lock:
            states.setdefault(key,{}).update(values)
            write_json(root/'matrix_status.json',{'seed':args.seed,'steps':args.steps,'jobs':states})
    for m,s in jobs:
        update(f'{m}/{s}',state='queued')
    base=[sys.executable,str(PROJECT/'train_vlm_competitors.py'),'--data-root',str(Path(args.data_root).resolve()),
          '--clip-checkpoint',str(Path(args.clip_checkpoint).resolve()),'--output-dir',str(root),
          '--cache-dir',str(Path(args.cache_dir).resolve()),'--steps',str(args.steps),'--batch-size',str(args.batch_size),
          '--seed',str(args.seed),'--text-microbatch',str(args.text_microbatch),'--adclip-repo',args.adclip_repo,
          '--daprompt-repo',args.daprompt_repo]
    def execute(job):
        method,source=job;key=f'{method}/{source}'
        run=root/method/source.replace(' ','_')/f'seed_{args.seed}'
        expected={'method':method,'source':source,'seed':args.seed,'steps':args.steps,'batch_size':args.batch_size,
                  'implementation_sha256':hashes[method],'checkpoint_sha256':checkpoint_sha,
                  'text_microbatch':args.text_microbatch,'evaluation_limit':0}
        if (run/'metrics.json').exists():
            record=json.loads((run/'metrics.json').read_text())
            if record['status']!='complete' or any(record['config'].get(k)!=v for k,v in expected.items()):
                raise ValueError(f'preserved completed configuration differs: {run}')
            update(key,state='complete',macro_accuracy=record['macro_accuracy'],skipped=True)
            print(f'SKIP {key}',flush=True);return 0
        device=devices.get()
        try:
            command=[*base,'--method',method,'--source',source,'--device',device]
            if (run/'config.json').exists():
                if args.resume_failed and (run/'last.pt').exists():
                    command.append('--resume')
                else:
                    raise FileExistsError(f'unfinished run needs --resume-failed and a checkpoint: {run}')
            run.mkdir(parents=True,exist_ok=True)
            update(key,state='running',device=device,started=datetime.now(ZONE).isoformat())
            print(f'START {key} seed={args.seed} {device}',flush=True)
            first=None;estimated=set()
            with (run/'process.log').open('a',buffering=1) as log:
                process=subprocess.Popen(command,cwd=PROJECT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in process.stdout:
                    log.write(line)
                    try:
                        event=json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get('event')=='progress':
                        if first is None:
                            first=event
                        milestone=20 if event['step']<500 else 500
                        if event['step']>=milestone and milestone not in estimated and event['step']>first['step']:
                            rate=(event['elapsed_seconds']-first['elapsed_seconds'])/(event['step']-first['step'])
                            final_eval=max(0,first['elapsed_seconds']-rate*first['step'])
                            eta=datetime.now(ZONE)+timedelta(seconds=rate*(args.steps-event['step'])+final_eval)
                            update(key,step=event['step'],seconds_per_step=rate,eta=eta.isoformat())
                            print(f'ETA {key}: ~{eta:%m-%d %H:%M:%S} UTC+8 ({rate:.3f}s/update)',flush=True)
                            estimated.add(milestone)
                code=process.wait()
            if code:
                update(key,state='failed',exit_code=code)
                print(f'FAILED {key}\n{(run/"process.log").read_text()[-2500:]}',flush=True)
            else:
                record=json.loads((run/'metrics.json').read_text())
                update(key,state='complete',step=args.steps,macro_accuracy=record['macro_accuracy'],completed=datetime.now(ZONE).isoformat())
                print(f'DONE {key}: {100*record["macro_accuracy"]:.4f}%',flush=True)
            return code
        finally:
            devices.put(device)
    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        results=list(pool.map(execute,jobs))
    if any(results):
        raise SystemExit('matrix failed; inspect preserved per-run logs')
    print('COMPLETE: all requested four-source comparisons',flush=True)


if __name__=='__main__':
    main()
