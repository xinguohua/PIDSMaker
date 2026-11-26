"""Descriptive FP/TN pairs with explicit 1.25-fold calipers."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

out=Path(os.environ['RUN_OUTPUT_DIR'])
data=pd.read_csv(Path(sys.argv[1])/'node_peaks.csv.gz',keep_default_na=False)
for side in ['src','dst']:
    data[side+'_type']=data[side].str.split().str[0]
    data[side+'_missing']=data[side].str.contains(r'(?:^|\s)(?:None|null|unknown)(?:\s|$)',case=False)
keys=['op','src_type','dst_type','src_missing','dst_missing']
rows=[]
for _,part in data.groupby(keys):
    controls=part[part.group=='TN']
    for fp in part[part.group=='FP'].itertuples():
        if fp.n<=0:
            continue
        subset=controls[controls.n.between(fp.n/1.25,fp.n*1.25)&controls.degree.between(fp.degree/1.25,fp.degree*1.25)].copy()
        if subset.empty:
            continue
        subset['opposite']=(subset.n<subset.maximum)!=(fp.n<fp.maximum)
        subset['distance']=np.abs(np.log(subset.n/fp.n))+np.abs(np.log(subset.degree/fp.degree))
        tn=subset.sort_values(['opposite','distance','node'],ascending=[False,True,True]).iloc[0]
        rows.append(dict(fp_uuid=fp.uuid,tn_uuid=tn.uuid,op=fp.op,fp_src=fp.src,fp_dst=fp.dst,tn_src=tn.src,tn_dst=tn.dst,
                         fp_n=fp.n,tn_n=tn.n,fp_max=fp.maximum,tn_max=tn.maximum,fp_degree=fp.degree,tn_degree=tn.degree,
                         fp_guess=fp.guess,tn_guess=tn.guess,fp_score=fp.score,tn_score=tn.score,opposite_support=bool(tn.opposite)))
pd.DataFrame(rows).to_csv(out/'case_pairs.csv',index=False)
(out/'summary.json').write_text(json.dumps(dict(eligible_fp=int(((data.group=='FP')&(data.n>0)).sum()),matched_fp=len(rows),
    opposite_support=sum(r['opposite_support'] for r in rows),distinct_tn=len({r['tn_uuid'] for r in rows})),indent=2))
