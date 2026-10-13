#!/usr/bin/env python3
"""Validate fixed-budget runs and report all 12 directed Office-Home tasks."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from mtda.experiment import write_json

DOMAINS=('Art','Clipart','Product','Real World')
COMPETITORS=('DAPrompt','DAPrompt-CodeTau','ADCLIP-Code','ADCLIP-Registered')
METHODS=('CLIP Zero-shot','CLIP Zero-shot (DAPL template)','Source Prompt Only','B / R0M','R2','R4',*COMPETITORS)


def csv_file(path,rows):
    if not rows:
        return
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def finite(value):
    if isinstance(value,dict):
        return all(finite(v) for v in value.values())
    if isinstance(value,list):
        return all(finite(v) for v in value)
    return not isinstance(value,(int,float)) or math.isfinite(value)


def summarize(args):
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    tasks=[];resources=[];runs=[]
    def add(method,source,targets):
        for target,result in targets.items():
            tasks.append({'method':method,'source':source,'target':target,'seed':42,'accuracy':100*result['accuracy'],
                          'samples':result['samples']})
    zs=json.loads((output/'zero_shot_templates.json').read_text())
    sample_counts={d:zs['results']['CLIP Zero-shot']['domains'][d]['samples'] for d in DOMAINS}
    for method,result in zs['results'].items():
        for source in DOMAINS:
            add(method,source,{d:r for d,r in result['domains'].items() if d!=source})
    for method,folder in [('Source Prompt Only',Path(args.source_only_dir)),
                          ('B / R0M',Path(args.baseline_dir)/'R0M'),
                          ('R2',Path(args.baseline_dir)/'R2'),('R4',Path(args.baseline_dir)/'R4')]:
        for source in DOMAINS:
            record=json.loads((folder/source.replace(' ','_')/'seed_42/metrics.json').read_text())
            if record['config']['steps']!=1000 or record['config']['seed']!=42:
                raise ValueError('reference budget/seed differs')
            add(method,source,record['targets'])
    for method in COMPETITORS:
        for source in DOMAINS:
            root=Path(args.runs_dir)/method/source.replace(' ','_')/'seed_42'
            if not (root/'metrics.json').exists():
                if args.allow_partial:
                    continue
                raise FileNotFoundError(f'incomplete formal matrix: {root}')
            record=json.loads((root/'metrics.json').read_text());cfg=record['config']
            expected={d:1000 for d in DOMAINS}
            if record['status']!='complete' or record['step']!=1000 or cfg['seed']!=42 or cfg['batch_size']!=30:
                raise ValueError(f'not the formal fixed-budget protocol: {root}')
            if record['effective_updates']!=expected or record['sample_exposures']!={d:30000 for d in DOMAINS}:
                raise ValueError(f'per-domain budget differs: {root}')
            if cfg['target_training_ground_truth'] or cfg['checkpoint_selection']!='fixed final budget' or cfg['evaluation_limit']:
                raise ValueError(f'target-label feedback or incomplete evaluation: {root}')
            if cfg['checkpoint_sha256']!=zs['checkpoint_sha256'] or not record['checkpoint_prediction_roundtrip']:
                raise ValueError(f'weight provenance/restoration differs: {root}')
            history=list(csv.DictReader((root/'history.csv').open()))
            if len(history)!=1000 or [int(r['step']) for r in history]!=list(range(1,1001)):
                raise ValueError(f'training history is incomplete: {root}')
            for row in history:
                step=int(row['step'])
                if int(row['source_updates'])!=step or json.loads(row['target_updates'])!={d:step for d in cfg['targets']}:
                    raise ValueError(f'update counters differ: {root}')
                if not math.isfinite(float(row['total_loss'])) or not finite(json.loads(row['components'])):
                    raise ValueError(f'nonfinite history: {root}')
            if any(result['samples']!=sample_counts[d] for d,result in record['targets'].items()):
                raise ValueError(f'evaluation drops samples: {root}')
            add(method,source,record['targets']);runs.append(str(root))
            resources.append({'method':method,'source':source,'seed':42,'source_updates':1000,'updates_per_target':1000,
                'source_draws':30000,'draws_per_target':30000,'source_ce_sample_views':record['source_ce_sample_views'],
                'optimizer_steps':record['optimizer_step_calls'],'seconds_including_evaluation':record['training_seconds'],
                'peak_memory_gib':record['peak_cuda_memory_bytes']/2**30,'learnable_parameters':cfg['learnable_parameters'],
                'optimizer':cfg['optimizer'],'learning_rate':cfg['learning_rate'],'threshold':cfg['threshold'],
                'upstream_sha':cfg['upstream_sha'],'implementation_sha256':cfg['implementation_sha256'],
                'source_ce_first_100':statistics.mean(json.loads(r['components'])['source_ce'] for r in history[:100]),
                'source_ce_last_100':statistics.mean(json.loads(r['components'])['source_ce'] for r in history[-100:])})
    values={(r['method'],r['source'],r['target']):r['accuracy'] for r in tasks}
    if len(values)!=len(tasks):
        raise ValueError('duplicate task results')
    summary=[];source_rows=[]
    for method in METHODS:
        keys=[(method,s,t) for s in DOMAINS for t in DOMAINS if s!=t]
        if not all(k in values for k in keys):
            continue
        row={'method':method,'macro_accuracy':statistics.mean(values[k] for k in keys)}
        row['clipart_target_accuracy']=statistics.mean(values[method,s,'Clipart'] for s in DOMAINS if s!='Clipart')
        row['gain_vs_clip_zero_shot_pp']=row['macro_accuracy']-statistics.mean(values['CLIP Zero-shot',s,t] for s in DOMAINS for t in DOMAINS if s!=t)
        row['gain_vs_B_pp']=row['macro_accuracy']-statistics.mean(values['B / R0M',s,t] for s in DOMAINS for t in DOMAINS if s!=t)
        row['gain_vs_R4_pp']=row['macro_accuracy']-statistics.mean(values['R4',s,t] for s in DOMAINS for t in DOMAINS if s!=t)
        summary.append(row)
        for source in DOMAINS:
            source_rows.append({'method':method,'source':source,'macro_accuracy':statistics.mean(values[method,source,t] for t in DOMAINS if t!=source)})
    complete=[r['method'] for r in summary]
    directed=[]
    for source in DOMAINS:
        for target in DOMAINS:
            if source!=target:
                directed.append({'source':source,'target':target,**{m:values[m,source,target] for m in complete}})
    csv_file(output/'accuracy_tasks.csv',tasks);csv_file(output/'accuracy_12_tasks.csv',directed)
    csv_file(output/'macro_summary.csv',summary);csv_file(output/'source_summary.csv',source_rows)
    csv_file(output/'resources.csv',resources)
    write_json(output/'summary.json',{'seed':42,'backbone':'RN50','completed_competitor_runs':len(runs),
        'expected_competitor_runs':16,'methods':summary,'source_summary':source_rows,'resources':resources})
    write_json(output/'validation.json',{'completed_runs':len(runs),'expected_runs':16,'histories_finite':True,
        'updates_per_domain':1000,'samples_per_domain':30000,'full_target_evaluation_counts':sample_counts,
        'restored_predictions_verified':True,'fixed_final_checkpoint':True,'seed':42,'backbone':'RN50',
        'native_losses_preserved':True,'native_long_budgets_run':False,'runtime_hashes':sorted({r['implementation_sha256'] for r in resources})})
    sr={(r['method'],r['source']):r['macro_accuracy'] for r in source_rows}
    lines=['# VLM MTDA controlled comparison','',
        f'Seed 42, RN50, one source and three simultaneous targets, 1,000 updates and batch 30 per domain. Completed competitor runs: {len(runs)}/16.',
        '', 'All values below are accuracy percentages. Each source column averages its three targets; Overall averages all 12 directed tasks. Final checkpoints only.',
        '', '| Method | Art source | Clipart source | Product source | Real World source | Overall | Clipart target |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        method=row['method']
        lines.append(f'| {method} | '+' | '.join(f'{sr[method,s]:.2f}' for s in DOMAINS)+f' | {row["macro_accuracy"]:.2f} | {row["clipart_target_accuracy"]:.2f} |')
    lines+=['','## All 12 tasks','', '| Source → Target | '+' | '.join(complete)+' |', '|---|'+'---:|'*len(complete)]
    for row in directed:
        lines.append(f'| {row["source"]} → {row["target"]} | '+' | '.join(f'{row[m]:.2f}' for m in complete)+' |')
    lines+=['','## Interpretation and limits','',
        'The cross-target teacher effect is isolated by R4 minus B/R0M: **+1.6483 pp** at seed 42. Both use the same frozen CLIP, prompt structure, source supervision, per-domain updates and evaluation. Their established three-seed paired gain is +1.7857 ± 0.1571 pp. Competitor ranking supplies a second check at the selected budget; differing native losses, optimizers, parameter sharing and model sizes prevent assigning every ranking difference to teachers.',
        '', 'DAPrompt uses threshold 0.6 from the Office-Home paper; CodeTau uses 0.5 from the released RN50 config. The period in its native zero-shot template is evaluated separately. DAPL accepted-sample CE and shared generic context are preserved.',
        '', 'ADCLIP-Code retains unupdated random first-layer projections from the official Python lists. ADCLIP-Registered explicitly repairs their optimizer registration. Both preserve the official source-input mapping loss and the nonstandard KL/entropy reductions. Target-only inference uses the paper-described mapping substitute because the released RN50 evaluation API does not provide a working single-image path. These are audited code adaptations, **not a reproduction of the paper numbers**.',
        '', 'AD-CLIP sees 30,000 source draws but 90,000 conditional source CE views, because the same source batch participates in three target pairs. Draws include repeated images. Every method receives 30,000 draws and 1,000 updates for each target. Native epoch budgets, random augmentation and AMP are outside this controlled experiment; the user selected the 1,000-step comparison.',
        '', 'Only one competitor seed has been run. Report practical performance under this protocol; do not infer statistical significance or superiority at longer native budgets.',
        '', '## Artifacts and reproduction','',
        'See [protocol](../../docs/vlm_competitors_protocol.md), [12-task CSV](accuracy_12_tasks.csv), [source means](source_summary.csv), [resources and provenance](resources.csv), and [validation](validation.json).',
        '', '```bash','python scripts/run_vlm_competitors.py --data-root /home/tangxunchi/office_home --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt --devices cuda:0 cuda:1','python scripts/evaluate_competitor_zeroshot.py','python scripts/summarize_vlm_competitors.py','```','']
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps({'completed_runs':len(runs),'macro':summary},ensure_ascii=False))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--runs-dir',default='runs/vlm_competitors')
    cli.add_argument('--baseline-dir',default='runs/b_reliability')
    cli.add_argument('--source-only-dir',default='runs/source_prompt_only')
    cli.add_argument('--output-dir',default='reports/vlm_competitors')
    cli.add_argument('--allow-partial',action='store_true')
    summarize(cli.parse_args())


if __name__=='__main__':
    main()
