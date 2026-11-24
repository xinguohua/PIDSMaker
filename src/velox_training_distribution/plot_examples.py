"""Draw each actual unit membership and each directed edge in its Velox window."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess


def quote(text):
    return json.dumps(str(text), ensure_ascii=False)


def node(key, label, **attrs):
    settings=dict(label=label, **attrs)
    return f'{quote(key)} ['+', '.join(f'{k}={quote(v)}' for k,v in settings.items())+'];'


def edge(a,b,**attrs):
    return f'{quote(a)} -> {quote(b)} ['+', '.join(f'{k}={quote(v)}' for k,v in attrs.items())+'];'


def header(title, caption):
    return ['digraph example {', 'rankdir=LR;',
            'graph [bgcolor=white,pad=0.25,nodesep=0.25,ranksep=0.7,labelloc=t,labeljust=l,fontname="Helvetica",fontsize=14];',
            'node [fontname="Helvetica",fontsize=10,style="rounded,filled",color="#6a7984"];',
            'edge [fontname="Helvetica",fontsize=9,color="#496874"];',
            f'label={quote(title+chr(10)+caption)};']


def render(path, lines, formats):
    source='\n'.join(lines+['}']).encode()
    for fmt in formats:
        subprocess.run(['dot','-T'+fmt,'-o',str(path.with_suffix('.'+fmt))],input=source,check=True)


def draw(case, out, i):
    units={u['uid']:u for u in case['units']}
    roles={n['nid']:n for n in case['nodes']}
    title=(f"Velox training window {case['units'][0]['segment']}   |   {case['hop']}-hop group {case['composition_id']}"
           f"   |   {len(units)} units, {len(case['events'])} edges   |   observed {case['frequency']} time(s)")
    groups=header(title,'Arrows show native process dependencies. Plain lines identify use of the same resource node.')
    for u in units.values():
        groups.append(node('u'+str(u['uid']),f"{u['program']}  ·  U{u['uid']}\nactivity type observed {u['unit_type_frequency']} times",shape='box',fillcolor='#d9efea'))
    for rid in sorted({x['resource'] for x in case['resource_incidence']}):
        r=roles[rid]
        groups.append(node('r'+str(rid),f"{r['kind']}  ·  #{r['native_nid']}",shape='ellipse',fillcolor='#f8edda'))
    for item in case['resource_incidence']:
        groups.append(edge('u'+str(item['uid']),'r'+str(item['resource']),arrowhead='none'))
    dependencies=Counter((e['src_uid'],e['dst_uid'],e['op']) for e in case['process_links'])
    for (a,b,op),n in sorted(dependencies.items()):
        groups.append(edge('u'+str(a),'u'+str(b),label=f'{op.removeprefix("EVENT_")} × {n}',color='#11675e'))
    stem=f'Example_{i}_h{case["hop"]}_c{case["composition_id"]}'
    render(out/stem,groups,('pdf','svg','png'))

    native=header(title,'Original directed endpoints. Each arrow is one training tensor row; b and r give its batch and row within batch.')
    endpoints={n['native_nid']:n for n in roles.values()}
    for key,n in sorted(endpoints.items()):
        label=str(n.get('label') or n['kind'])
        if len(label)>65:label=label[:62]+'…'
        native.append(node(key,f'{label}\n#{key}',shape='box' if n['kind']=='subject' else 'ellipse',
                           fillcolor='#d9efea' if n['kind']=='subject' else '#f8edda'))
    seen=set()
    for e in case['events']:
        assert e['native_src'] in endpoints and e['native_dst'] in endpoints and e['eid'] not in seen
        seen.add(e['eid'])
        native.append(edge(e['native_src'],e['native_dst'],
                           label=f"{e['op'].removeprefix('EVENT_')}  b{e['batch_id']}/r{e['batch_row']}"))
    render(out/(stem+'_native_edges'),native,('svg',))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('examples',type=Path)
    parser.add_argument('output',type=Path)
    a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    for i,line in enumerate(a.examples.read_text().splitlines(),1):
        draw(json.loads(line),a.output,i)
