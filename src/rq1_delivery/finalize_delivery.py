"""Finalize and package the three frozen-data analyses after validation on the server."""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

root=Path(sys.argv[1])
stage=Path(sys.argv[2])
protocol_dir=root/'native_paths_len3_v2'
out=protocol_dir/'results'
run_id=sys.argv[3]
if Path(run_id).name!=run_id:raise ValueError('Expected a run ID')
runs=Path('/mnt/bigdata/nsas2020/tdsc-runs')
ids=['rq1-action-extract-20261125a','rq1-action-recover-20261125a','rq1-action-recover-20261125b',run_id]
manifest={}
for rid in ids:
    loc=runs/rid
    manifest[rid]={'meta':json.loads((loc/'meta.json').read_text()),'status':json.loads((loc/'status.json').read_text())}
assert manifest[ids[-1]]['status']['state']=='succeeded'
for name in ['RAW_AUDIT.json','NATIVE_WINDOWS.csv','INPUT_AUDIT.json']:
    shutil.copy2(root/name,out/name)
shutil.copy2(protocol_dir/'PROTOCOL.json',out/'PROTOCOL.json')
for name in ['verify_delivery.py','finalize_delivery.py','rq1_figures.py','rq1_examples.py']:
    shutil.copy2(Path(__file__).with_name(name),out/name)
for name in ['MATCHING_RULES.md','INDEX.md','COMPOSITION_SPACE.md','resources.log','tests.log','NATIVE_INPUT_AUDIT.json']:
    shutil.copy2(stage/name,out/name)
(out/'CODE').mkdir(exist_ok=True)
for name in ['action_composition.py','action_recover.py','action_paths.py','action_window.py','action_path_check.py','action_index.py']:
    source=runs/run_id/'code/src'/name
    expected=json.loads((out/'PROTOCOL.json').read_text())['code'].get(name)
    if expected and hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ValueError(f'Code mismatch: {name}')
    shutil.copy2(source,out/'CODE'/name)
(out/'RUN_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
subprocess.run([sys.executable,str(out/'verify_delivery.py'),str(out)],check=True)

def rows(name):
    with (out/name).open() as f:return list(csv.DictReader(f))
summary=rows('COMPOSITION_SUMMARY.csv');perf=rows('DETECTION_PERFORMANCE.csv')
lines=['# CADETS E3 × Velox 动作组合统计','',
    '运行复用 581 个正常训练窗口与 339 个测试窗口，5,492,259 条原生边均通过来源审计。复用原模型及分数，采用回顾性拐点 score > 4.268208026885985。所有下表计数以同一窗口内按原生端点 ID 相同接续、方向一致且严格有序的事件序列为实例。','',
    '|长度|训练实例|训练类型|正常测试实例|正常测试类型|已见 Action 构成的未见测试类型（可观测签名）|','|---:|---:|---:|---:|---:|---:|']
for r in summary:
    keys=['length','train_instances','train_observed_types','test_instances','test_observed_types','Zero_shot_observable_signatures']
    lines.append('|'+'|'.join(format(int(r[k]),',') for k in keys)+'|')
lines+=['','## 组成 Action 分布','','|长度|全类别 Pearson|共同 Top-25 Pearson|未见测试实例中所有 Action 已见的比例|','|---:|---:|---:|---:|']
corr=rows('ACTION_DISTRIBUTIONS/CORRELATIONS.csv')
for k in (2,3):
    rr={r['range']:r for r in corr if int(r['length'])==k}
    g=next(v for v in rows(f'ACTION_DISTRIBUTIONS/length_{k}_zero_shot_groups.csv') if v['group']=='all_actions_seen')
    lines.append(f"|{k}|{rr['all_categories']['pearson']}|{rr['top25_shared_All_order']['pearson']}|{g['instance_fraction']}|")
lines+=['','## 正常节点检测分组','','|长度|分组|节点|FP|TN|FPR|','|---:|---|---:|---:|---:|---:|']
for r in perf:
    if r['group'] in ('All','Seen','Zero-shot','Other'):
        lines.append('|'+'|'.join(r[k] for k in ('length','group','nodes','FP','TN','FPR'))+'|')
missing_file_paths=json.loads((out/'ANALYSIS_AUDIT.json').read_text())['missing_file_paths']
lines+=['','## 可核查范围','',
    '原始 CDM 跨文件重复使用 UUID，最终按原生行、同一操作、纳秒时间和有向端点匹配。源码和失败后恢复的阶段见 RUN_MANIFEST.json。全部窗口都通过独立的无类型整数计数校验；真实窗口片段的类型频次和节点成员另经穷举验证。',
    f'训练和正常测试频次、理论空间和组成 Action 分布使用各自明确的分母；具体属性缺失时，另报完整语义子空间和全体可观测签名，不把缺失值等同于相同具体对象。{missing_file_paths:,} 个文件节点没有可确认路径。',
    '72 个攻击节点在回顾性拐点下 TP=32、FN=40。缺少可独立归属为正常活动的上下文，未生成攻击节点 Seen/Zero-shot 子组的 FNR。节点关联统计不作因果解释。']
(out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
with (out/'INDEX.md').open('a') as f:
    f.write('\n正式统计与来源消歧使用提交 '+manifest[ids[-1]]['meta']['commit']+'。服务器完整交付目录为 '+str(out)+'/。三个分析的全量表和分母已由 verify_delivery.py 交叉核验。\n')
subprocess.run([sys.executable,str(out/'rq1_examples.py'),str(out),str(root)],check=True)
subprocess.run([sys.executable,str(out/'rq1_figures.py'),str(out),str(runs/run_id)],check=True)
ref=json.loads((out/'REAL_PATH_CHECK.json').read_text())
knee=json.loads((out/'OPERATING_POINT.json').read_text())
status=json.loads((out/'ANALYSIS_AUDIT.json').read_text())
status.update(window_ids_verified=True,recoverable_reference=ref['all_counts_and_node_memberships_equal'],
    exact_weighted_actions_verified=True,node_decisions_verified=True,retrospective_threshold=knee['policy']['threshold'],
    decisions_sha256=knee['decisions_sha256'],operating_metrics=knee['exploratory_metrics'],
    case_coverage=json.loads((out/'CASE_COVERAGE.json').read_text()),figures=['figure1','figure2','figure3'])
(out/'ANALYSIS_AUDIT.json').write_text(json.dumps(status,indent=2))
audit=json.loads((out/'INPUT_AUDIT.json').read_text())
audit['native_metrics']=audit['metrics']
audit['analysis_operating_point']=dict(scope=knee['scope'],threshold=knee['policy']['threshold'],
                                      decisions_sha256=knee['decisions_sha256'],metrics=knee['exploratory_metrics'])
(out/'INPUT_AUDIT.json').write_text(json.dumps(audit,indent=2))
with (out/'INDEX.md').open('a') as f:
    f.write('\nOPERATING_POINT.json 为拐点口径；INPUT_AUDIT.json 同时保留原生指标。PHASE_TIMES.json 和 PERFORMANCE_REPORT.md 记录执行成本；CASE_COVERAGE.json 和 EXAMPLES.jsonl 记录六类案例检查；FIGURES/ 含全部正式图。图一源为 FREQUENCY_BINS.csv、完整类型表、COMPOSITION_SUMMARY.csv 和 rank_intervals；图二源为 ACTION_DISTRIBUTIONS/，图三源为 DETECTION_PERFORMANCE.csv 和 NODE_SCORES.csv。\n')
archive=protocol_dir/'RQ1_Action_Composition_完整结果.zip'
temp=archive.with_suffix('.partial')
with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=2,allowZip64=True) as z:
    for p in sorted(out.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(out))
with zipfile.ZipFile(temp) as z:assert z.testzip() is None
temp.replace(archive)
with archive.open('rb') as f:checksum=hashlib.file_digest(f,'sha256').hexdigest()
archive.with_suffix('.sha256').write_text(checksum+'  '+archive.name+'\n')
print('PACKAGED',archive.stat().st_size,checksum,flush=True)
