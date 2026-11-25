"""Render figures from the verified full counts, distributions and node scores."""
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out,run=map(Path,sys.argv[1:])
figdir=out/'FIGURES';figdir.mkdir(exist_ok=True)
def records(name):
    with (out/name).open() as f:yield from csv.DictReader(f)
def save(fig,name):
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(figdir/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
summary={int(r['length']):r for r in records('COMPOSITION_SUMMARY.csv')}
fig,axes=plt.subplots(2,3,figsize=(13,7))
longtail=[]
for k in (1,2,3):
    ax=axes[0,k-1];hist=Counter(int(r['train_frequency']) for r in records('ACTION_COUNTS.csv' if k==1 else f'COMPOSITION_COUNTS/length_{k}.csv'))
    rank=1;xs=[];ys=[]
    for freq,n in sorted(hist.items(),reverse=True):
        longtail.append((k,rank,rank+n-1,freq,n))
        xs.extend([rank,rank+n-1]);ys.extend([freq,freq]);rank+=n
    ax.plot(xs,ys,lw=1.2);ax.set(xscale='log',yscale='symlog',title=f'{k} action'+('s' if k>1 else ''),xlabel='Type rank',ylabel='Training instances')
    ax=axes[1,k-1];r=summary[k]
    for y,scope in enumerate(('known_semantics','observable_signatures')):
        total=int(r['candidate_'+scope]);left=0
        for cat,color in [('Seen','#147d74'),('Zero_shot','#da8138'),('Potential','#b5bfd0')]:
            frac=int(r[cat+'_'+scope])/total
            ax.barh(y,frac,left=left,color=color,label=cat if y==0 else None);left+=frac
        ax.text(.01,y,f"Seen {100*int(r['Seen_'+scope])/total:.3g}%",va='center',bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    ax.set(xlim=(0,1),yticks=[0,1],yticklabels=['Known','Observable'],xlabel='Share of candidate space')
    if k==1:ax.legend(loc='lower left',bbox_to_anchor=(0,-.6),ncol=3,fontsize=8)
with (figdir/'figure1_rank_intervals.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['length','first_rank','last_rank','training_frequency','types']);w.writerows(longtail)
save(fig,'figure1')
fig,axes=plt.subplots(2,3,figsize=(13,7))
cor={int(r['length']):r for r in records('ACTION_DISTRIBUTIONS/CORRELATIONS.csv') if r['range']=='all_categories'}
for k in (1,2,3):
    ax=axes[0,k-1];data=list(records(f'ACTION_DISTRIBUTIONS/length_{k}_top25.csv'))
    for field,label in [('All_fraction','All'),('Zero_shot_fraction','Zero-shot')]:
        ax.plot([int(r['rank']) for r in data],[float(r[field]) if r[field]!='NA' else np.nan for r in data],marker='.',label=label)
    ax.set(title=f"Length {k}, full Pearson {float(cor[k]['pearson']):.3f}" if cor[k]['status']=='ok' else f"Length {k}, Pearson undefined",xlabel='Action rank in All (first 25)',ylabel='Share of Action positions');ax.legend()
    ax=axes[1,k-1];data=list(records(f'ACTION_DISTRIBUTIONS/length_{k}_zero_shot_groups.csv'))
    for i,field in enumerate(('type_fraction','instance_fraction')):
        ax.bar(np.arange(len(data))+.34*i,[float(r[field]) if r[field]!='NA' else np.nan for r in data],width=.34,label=['Types','Instances'][i])
    ax.set(xticks=np.arange(3)+.17,xticklabels=['Composition','Action unseen','Unknown'],ylim=(0,1),ylabel='Share of zero-shot');ax.tick_params(axis='x',labelrotation=15);ax.legend()
save(fig,'figure2')
fig,axes=plt.subplots(2,2,figsize=(11,8))
groups=['All','Seen','Zero-shot','Other']
for k in (2,3):
    data={r['group']:r for r in records('DETECTION_PERFORMANCE.csv') if int(r['length'])==k}
    rs=[data[g] for g in groups];ax=axes[0,k-2]
    bars=ax.bar(groups,[100*float(r['FPR']) if r['FPR']!='NA' else np.nan for r in rs],color=['#355d80','#147d74','#da8138','#9aa4aa'])
    ax.bar_label(bars,labels=[f"{r['FP']} / {int(r['nodes']):,}" for r in rs],fontsize=8,padding=3);ax.margins(y=.2);ax.set(ylabel='False positive rate (%)',title=f'Length {k}: FP / normal nodes')
    ax=axes[1,k-2];positions=[];boxes=[]
    for i,r in enumerate(rs):
        if int(r['nodes']):
            positions.append(i+1);boxes.append({a:float(r[b]) for a,b in [('med','score_median'),('q1','score_p25'),('q3','score_p75'),('whislo','score_min'),('whishi','score_max')]})
    ax.bxp(boxes,positions=positions,showfliers=False);ax.set(xticks=range(1,5),xticklabels=groups,yscale='symlog',ylabel='Anomaly score (min–max, quartiles)')
    ax.axhline(float(rs[0]['threshold']),ls='--',color='#ac4735',label='Frozen threshold');ax.legend()
save(fig,'figure3')
phases=json.loads((out/'PHASE_TIMES.json').read_text());windows=list(records('WINDOW_AUDIT.csv'))
slow=max(windows,key=lambda r:float(r['total_s'])+float(r['write_s']))
resources=out/'resources.log'
samples=[r.split() for r in resources.read_text().splitlines() if len(r.split())==5] if resources.exists() else []
metrics={'phase_seconds':phases,'workers':2,'peak_process_rss_mib':max(float(r['peak_rss_mb']) for r in windows),
    'slowest_window_compute_plus_write':slow['window'],'slowest_s':float(slow['total_s'])+float(slow['write_s']),
    'spool_total_bytes':sum(int(r['spool_bytes']) for r in windows),'counts_db_bytes':(out.parent/'counts.sqlite').stat().st_size,
    'sampled_process_tree_rss_sum_mib':max((int(r[3])/1024 for r in samples),default=None),
    'minimum_sampled_free_disk_gib':min((int(r[4])/1048576 for r in samples),default=None)}
(out/'PERFORMANCE_REPORT.md').write_text('# Formal run performance\n\n'+json.dumps(metrics,indent=2)+'\n\nProcess RSS sums include shared pages. Sampling interval: 15 s. Phase timers cover the counting process.\n')

with (out/'SUMMARY.md').open('a') as f:
    f.write('\n## Findings 草稿\n')
    for k,r in summary.items():
        f.write(f"\n- 长度 {k}，语义完整空间 {int(r['candidate_known_semantics']):,} 种，训练覆盖率 {float(r['training_coverage_known_semantics']):.6%}；训练未见而正常测试实际出现 {int(r['Zero_shot_known_semantics']):,} 种。")
    f.write(' 理论空间表示端点结构可接。未观察到的候选还缺少实际执行证据。\n')
    for k in (2,3):
        source=list(records(f'ACTION_DISTRIBUTIONS/length_{k}_zero_shot_groups.csv'))
        f.write(f"\n- 长度 {k} 的未见测试实例数按来源拆分为 "+'、'.join(f"{r['group']} {int(r['instances']):,}" for r in source)+f"；与全部正常测试的组成分布 Pearson 为 {cor[k]['pearson']}。")
        perf={r['group']:r for r in records('DETECTION_PERFORMANCE.csv') if int(r['length'])==k}
        f.write(f" Zero-shot 节点中 {perf['Zero-shot']['FP']}/{perf['Zero-shot']['nodes']} 误报，Seen 为 {perf['Seen']['FP']}/{perf['Seen']['nodes']}。")
    f.write('\n\n该拐点从这批测试数据回顾性选定。分组比较描述固定模型与阈值下的关联。需要干预训练覆盖的独立实验才能评估增强对误报的影响。\n')
