#!/usr/bin/env python3
"""Standalone comparison figure from completed, validated fixed-budget results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reports-dir',default='reports/vlm_competitors')
    args=cli.parse_args()
    root=Path(args.reports_dir)
    report=json.loads((root/'summary.json').read_text())
    if report['completed_competitor_runs']!=report['expected_competitor_runs']:
        raise ValueError('complete the controlled matrix before plotting final comparisons')
    methods=('CLIP Zero-shot','B / R0M','R4','DAPrompt','DAPrompt-CodeTau','ADCLIP-Code','ADCLIP-Registered')
    names=('CLIP zero-shot','B / R0M','R4 cross-target teachers','DAPL, tau=0.6','DAPL, code tau=0.5','AD-CLIP, code projections','AD-CLIP, registered projections')
    values={r['method']:r for r in report['methods']}
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.1),layout='constrained',sharey=True)
    for ax,key,title in zip(axes,('macro_accuracy','clipart_target_accuracy'),('All 12 directed tasks','Clipart as target (3 sources)')):
        points=[values[m][key] for m in methods]
        for i,(method,value) in enumerate(zip(methods,points)):
            ax.scatter(value,i,color='#2878B5' if method=='R4' else '#777777',s=55,zorder=3)
            ax.text(value+.13,i,f'{value:.2f}',va='center',fontsize=9)
        ax.set_xlim(min(points)-.7,max(points)+1.1)
        ax.set_yticks(range(len(methods)),names,fontsize=9)
        ax.set_xlabel('Final accuracy (%)');ax.set_title(title)
        ax.spines[['top','right','left']].set_visible(False)
        ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle('Office-Home MTDA: seed 42, frozen RN50, 1,000 updates per domain\nNative competitor losses and optimizers; controlled budget, not paper reproduction',fontsize=11)
    for suffix in ('png','pdf'):
        fig.savefig(root/f'vlm_comparison.{suffix}',dpi=220,bbox_inches='tight')


if __name__=='__main__':
    main()
