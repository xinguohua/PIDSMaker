"""Count multi-unit types and observed marginal/joint frequency relationships exactly."""
import argparse
from collections import Counter
import csv
from fractions import Fraction
from pathlib import Path


def quantile(counter, q):
    total=sum(counter.values())
    if not total:return None
    rank=max(1,(q.numerator*total+q.denominator-1)//q.denominator)
    count=0
    for x in sorted(counter):
        count+=counter[x]
        if count>=rank:return x
    raise AssertionError('frequency cross-check failed')


def summarize(path):
    groups={}
    with path.open() as f:
        for row in csv.DictReader(f):
            minimum=int(row['min_component_frequency'])
            frequency=int(row['composition_frequency'])
            distinct=int(row['composition_types'])
            instances=int(row['composition_instances'])
            if min(minimum,frequency,distinct)<1 or instances!=distinct*frequency:
                raise ValueError('Invalid frequency cross-tab')
            key=(row['order'],int(row['hop']))
            cells=groups.setdefault(key,Counter())
            cells[minimum,frequency]+=distinct
    for order in ('forward','reverse'):
        for hop in range(1,6):
            cells=groups.get((order,hop),{})
            distinct=sum(cells.values()); instances=sum(f*n for (m,f),n in cells.items())
            comp=Counter();singleton_marginal=Counter()
            for (m,f),n in cells.items():
                if f>m:
                    raise ValueError('Repeated compositions exceed available constituents in disjoint partition')
                comp[f]+=n
                if f==1:singleton_marginal[m]+=n
            yield dict(order=order,hop=hop,multi_types=distinct,multi_instances=instances,
                       types_seen_once=comp[1],singleton_instance_share=comp[1]/instances if instances else None,
                       types_with_minimum_marginal_gt_joint=sum(n for (m,f),n in cells.items() if m>f),
                       types_with_minimum_marginal_eq_joint=sum(n for (m,f),n in cells.items() if m==f),
                       min_marginal_among_singletons_p25=quantile(singleton_marginal,Fraction(1,4)),
                       min_marginal_among_singletons_median=quantile(singleton_marginal,Fraction(1,2)),
                       min_marginal_among_singletons_p75=quantile(singleton_marginal,Fraction(3,4)),
                       min_marginal_among_singletons_max=max(singleton_marginal,default=None),
                       typical_type_frequency=quantile(comp,Fraction(1,2)),
                       most_frequent_type_frequency=max(comp,default=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    a=parser.parse_args(); rows=list(summarize(a.source))
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
