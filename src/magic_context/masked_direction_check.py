"""Re-evaluate detector association using the native SCE target directions."""
from pathlib import Path
import json,os,sys
import numpy as np
import pandas as pd
from masked_analysis import unit,dist
d,raw=map(Path,sys.argv[1:]);out=Path(os.environ['RUN_OUTPUT_DIR'])
vec=pd.read_csv(d/'TARGET_VECTORS.csv')
ax=np.unique(np.round(unit(vec.filter(regex='^y[0-9]').to_numpy(float)),10),axis=0)
def axes(x):
    direction=unit(x)
    code=np.argmax(direction@ax.T,axis=1)
    assert np.allclose(direction,ax[code])
    return code
f=pd.read_csv(d/'LOCAL_TARGET_DISTRIBUTION.csv')
f['axis']=f.target.map(vec.set_index('target').axis)
f=f.groupby(['g0','axis']).n.sum()
table=f.unstack(fill_value=0)
major=table.idxmax(axis=1)
unique=table.eq(table.max(axis=1),axis=0).sum(axis=1).eq(1)
normal=pd.read_csv(d/'NORMAL_POPULATION.csv.gz').set_index('id')
dirs=[]
for file in sorted((raw/'test0').glob('*.npz')):
    w=int(file.stem.split('-')[0])
    with np.load(file) as s:
        ids=s['id'];wins=normal.window.reindex(ids).values
        mask=wins==w
        dirs.append(pd.Series(axes(s['target'][mask]),index=ids[mask]))
code=pd.concat(dirs)
assert len(code)==len(normal) and code.index.is_unique
normal['axis']=code
normal['major_axis']=normal.g0.map(major)
normal['support_axis']=f.reindex(pd.MultiIndex.from_frame(normal[['g0','axis']])).fillna(0).values
normal['eligible_axis']=normal.support_axis.gt(0)&normal.g0.map(unique).fillna(False)&normal.axis.ne(normal.major_axis)
normal['margin_axis']=np.nan
row=normal.eligible_axis
normal.loc[row,'margin_axis']=normal.loc[row,'loss']-dist(unit(normal.loc[row,['p0','p1','p2']].to_numpy()),ax[normal.loc[row,'major_axis'].astype(int)])
normal['biased_axis']=row&normal.margin_axis.gt(1e-9)
normal.groupby(['prediction','eligible_axis','biased_axis']).size().to_csv(out/'DETECTION_BY_TARGET_DIRECTION.csv')
sel=pd.read_csv(d/'FP_TN_LEARNING_BIAS_ANALYSIS.csv').id
normal.loc[sel,['prediction','g0','axis','major_axis','support_axis','eligible_axis','margin_axis','biased_axis']].to_csv(out/'MATCHED_TARGET_DIRECTIONS.csv')
