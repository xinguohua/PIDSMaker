"""Report what the source graph labels actually disclose for this Velox configuration."""
import argparse
import csv
from collections import Counter
from pathlib import Path

import duckdb


def coverage(root):
    source=duckdb.connect(str(root/'native_units.duckdb'),read_only=True)
    fields={'subject':('path','cmd_line'),'file':('path',),'netflow':('remote_ip','remote_port')}
    records=[]
    for kind,names in fields.items():
        measured=Counter()
        key_ids=set()
        distinct_labels=set()
        for node_id,label,n in source.execute("""
                 select native_nid,label,count(*) from nodes where kind=? group by all""",[kind]).fetchall():
            key_ids.add(node_id)
            distinct_labels.add(label)
            pieces=(label or '').split(maxsplit=len(names))
            if pieces and pieces[0]==kind:
                values=pieces[1:]
            else:
                values=[]
            present=[i<len(values) and values[i].strip().lower() not in ('','na','none','null','0')
                     for i in range(len(names))]
            measured['total']+=n
            measured['any_value']+=n*any(present)
            for key,p in zip(names,present):
                measured[key]+=n*p
        for key in ['any_value',*names]:
            n=measured['total']
            records.append(dict(kind=kind,field=key,window_node_observations=n,distinct_native_ids=len(key_ids),
                distinct_raw_labels=len(distinct_labels),present=measured[key],missing=n-measured[key],
                coverage=measured[key]/n if n else None))
    with (root/'NATIVE_LABEL_COVERAGE.csv').open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(records[0]))
        out.writeheader()
        out.writerows(records)
    source.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    coverage(p.parse_args().root)
