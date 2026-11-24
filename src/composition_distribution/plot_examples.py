"""Render complete observed composition memberships and physical-event witnesses."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess


def quote(value):
    return json.dumps(str(value), ensure_ascii=False)


def node(key, label, **attrs):
    attributes = dict(label=label, **attrs)
    return f'{quote(key)} [' + ', '.join(f'{k}={quote(v)}' for k, v in attributes.items()) + '];'


def edge(a, b, **attrs):
    return f'{quote(a)} -> {quote(b)} [' + ', '.join(f'{k}={quote(v)}' for k, v in attrs.items()) + '];'


def render(directory, name, graph, formats):
    dot = directory / (name + '.dot')
    dot.write_text('\n'.join(graph + ['}']) + '\n')
    for fmt in formats:
        subprocess.run(['dot', '-T'+fmt, str(dot), '-o', str(directory/(name+'.'+fmt))], check=True)


def graphs(case, directory, index):
    members = {u['uid']: u for u in case['units']}
    resources = {r['nid']: r for r in case['resources']}
    events = case['events']
    title = (f"Example {index}   ·   {case['hop']}-hop composition {case['composition_id']}"
             f"   ·   {len(members)} units   ·   {len(events)} owned events"
             f"   ·   composition frequency {case['frequency']}")
    def start(subtitle):
        return ['digraph case_graph {', 'rankdir=LR;', 'graph [bgcolor=white, pad=0.3, nodesep=0.24, ranksep=0.6, labelloc=t, labeljust=l, fontname="Helvetica", fontsize=15];',
                'node [fontname="Helvetica", fontsize=10, style="filled,rounded", color="#6a7984", margin="0.10,0.07"];',
                'edge [fontname="Helvetica", fontsize=9, color="#637887"];',
                f'label={quote(title + chr(10) + subtitle)};']
    unit_nodes = [node(f'u{uid}', f"{u['program']}  ·  U{uid}\nunit type observed {u['unit_type_frequency']} times",
                       shape='box', fillcolor='#d8eeeb') for uid, u in sorted(members.items())]
    def object_node(nid):
        obj = resources.get(nid)
        label = f"R{nid}\n{obj['role']}" if obj else f"Subject / endpoint N{nid}"
        return node(f'n{nid}', label, shape='ellipse', fillcolor='#f7eedb' if obj else '#e8e9ee')
    shape = start('Lines join units to actual resource instances. Arrows show directed process dependencies.') + unit_nodes
    evidence = start('Each event has one owner. Arrows retain primary information flow. Dashed links identify secondary endpoints.') + unit_nodes
    owners = Counter()
    endpoint_ids = set()
    for event in events:
        eid, uid = event['eid'], event['owner']
        assert uid in members and not owners[eid]
        owners[eid] += 1
        key = f'e{eid}'
        label = f"E{eid}  ·  {event['op'].removeprefix('EVENT_')}"
        evidence.append(node(key, label, fillcolor='#edf0f3', shape='box'))
        target = event['object1']
        if target is None:
            evidence.append(edge(f'u{uid}',key,label='owner',arrowhead='none'))
        elif event['direction']=='object_to_actor':
            evidence.extend([edge(f'n{target}',key),edge(key,f'u{uid}')])
        elif event['direction']=='actor_to_object':
            evidence.extend([edge(f'u{uid}',key),edge(key,f'n{target}')])
        else:
            evidence.extend([edge(f'u{uid}',key,label='owner',arrowhead='none'),edge(key,f'n{target}',label='target 1',arrowhead='none')])
        for i in (target,event['object2']):
            if i is not None:endpoint_ids.add(i)
        if event['object2'] is not None:
            evidence.append(edge(key,f"n{event['object2']}",label='target 2',style='dashed',arrowhead='none'))
    subjects = defaultdict(list)
    for uid, u in members.items():
        if u['subject'] is not None:subjects[u['subject']].append(uid)
    for nid in sorted(endpoint_ids):
        evidence.append(object_node(nid))
        for uid in subjects[nid]:
            evidence.append(edge(f'u{uid}',f'n{nid}',style='dotted',arrowhead='none',label='subject ID'))
    for nid in sorted(resources):shape.append(object_node(nid))
    for incidence in case['resource_incidence']:
        uid, nid = incidence['uid'], incidence['resource']
        assert uid in members and nid in resources
        shape.append(edge(f'u{uid}',f'n{nid}',arrowhead='none'))
    links = Counter((r['src_uid'],r['dst_uid'],r['op']) for r in case['process_links'])
    for (src,dst,op), repeat in sorted(links.items()):
        label = op.removeprefix('EVENT_') + (f' × {repeat}' if repeat>1 else '')
        shape.append(edge(f'u{src}',f'u{dst}',label=label,color='#176661',fontcolor='#155550'))
    stem = f'Example_{index}_hop{case["hop"]}_c{case["composition_id"]}'
    render(directory,stem,shape,('svg','pdf','png'))
    render(directory,stem+'_events',evidence,('svg',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('examples',type=Path)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    for i,line in enumerate(args.examples.read_text().splitlines(),1):
        graphs(json.loads(line),args.out,i)
