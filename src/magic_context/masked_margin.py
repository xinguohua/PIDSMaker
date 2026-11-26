"""Check substantive SCE margin thresholds on identical-input imbalanced groups."""
from pathlib import Path
import os,sys
import pandas as pd
d=Path(sys.argv[1])
g=pd.read_csv(d/'G3_groups.csv').dropna(subset=['minor_bias']).set_index('g0')
o=pd.read_csv(d/'DIRECTIONAL_RECONSTRUCTION_BIAS.csv',usecols=['g0','target','major','margin'])
x=o[o.g0.isin(g.index)&o.target.ne(o.major)].copy()
rows=[]
for min_gap in [1e-9,1e-4,.001,.01,.05]:
    for name,s in [('all',x),('positive_indegree',x[x.g0.map(g.indeg).gt(0)])]:
        rows.append(dict(group=name,threshold=min_gap,n=len(s),positive=int((s.margin>min_gap).sum()),proportion=(s.margin>min_gap).mean()))
pd.DataFrame(rows).to_csv(Path(os.environ['RUN_OUTPUT_DIR'])/'MARGIN_THRESHOLDS.csv',index=False)
