"""Plot the two distributions from complete audited frequency source tables."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def panels(title):
    fig,axes=plt.subplots(1,5,figsize=(15.7,4.45))
    fig.subplots_adjust(left=.055,right=.926,bottom=.255,top=.79,wspace=.24)
    fig.suptitle(title,x=.055,y=.98,ha='left',fontsize=16,fontweight='bold')
    return fig,axes


def save(fig, directory, name):
    for fmt in ('pdf','png'):
        fig.savefig(directory/(name+'.'+fmt),dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def colorbar(fig,artist,label):
    ax=fig.add_axes([.94,.28,.009,.47])
    fig.colorbar(artist,cax=ax)
    ax.set_title(label,fontsize=8,loc='left',pad=10)


def plot(source, out):
    out.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'font.family':'DejaVu Sans'})
    data=[r for r in read(source/'FIG1_ROWS.csv') if r['order']=='forward']
    ranked=sorted(read(source/'FIG1_RANKS.csv'),key=lambda r:int(r['rank']))
    types=[r['unit_type'] for r in ranked]
    assert [int(r['rank']) for r in ranked]==list(range(1,len(types)+1))
    assert ranked==sorted(ranked,key=lambda r:(-int(r['training_frequency']),r['unit_type']))
    assert types and set(types)=={r['type_a'] for r in data}=={r['type_b'] for r in data}
    index={v:i for i,v in enumerate(types)}
    matrices=np.zeros((5,len(types),len(types)),dtype=np.int64)
    recorded=set()
    for r in data:
        k,i,j=int(r['hop'])-1,index[r['type_a']],index[r['type_b']]
        assert (k,i,j) not in recorded
        recorded.add((k,i,j))
        matrices[k,i,j]=int(r['composition_instances'])
    assert all(np.array_equal(m,m.T) for m in matrices) and len(recorded)==matrices.size
    cmap=plt.get_cmap('viridis').copy()
    cmap.set_bad('#eceeed')
    norm=LogNorm(vmin=1,vmax=max(2,int(matrices.max())))
    fig,axes=panels('Training compositions containing pairs of frequent execution-unit types')
    labels=[str(i+1) for i in range(len(types))]
    for k,(ax,m) in enumerate(zip(axes,matrices),1):
        im=ax.imshow(np.ma.masked_equal(m,0),cmap=cmap,norm=norm,interpolation='nearest')
        ax.set(title=f'{k} hop',xticks=range(len(types)),yticks=range(len(types)))
        ax.set_xticklabels(labels,fontsize=7);ax.set_yticklabels(labels if k==1 else [])
        ax.tick_params(length=0)
        if k==1:ax.set_ylabel('Execution-unit type rank')
    colorbar(fig,im,'Instances')
    fig.text(.055,.17,'Axes show the same types in training-frequency order. A diagonal cell requires two distinct units of that type.',fontsize=10)
    fig.text(.055,.105,'Grey = no observed co-occurrence in that partition. A composition contributes once per displayed type pair.',fontsize=9,color='#414951')
    save(fig,out,'Figure_1_Behavioral_Composition_Distribution')
    freq=[r for r in read(source/'FIG2_ROWS.csv') if r['order']=='forward']
    if not freq:
        raise ValueError('No multi-unit compositions in plot source')
    def bins(field):
        highest=max(int(r[field]) for r in freq)
        return np.geomspace(.9,max(1.1,highest*1.08),34)
    xb,yb=bins('min_component_frequency'),bins('composition_frequency')
    densities=[]
    for k in range(1,6):
        rr=[r for r in freq if int(r['hop'])==k]
        weights=[int(r['composition_types']) for r in rr]
        h,_,_=np.histogram2d([int(r['min_component_frequency']) for r in rr],
                             [int(r['composition_frequency']) for r in rr],bins=(xb,yb),weights=weights)
        assert int(h.sum())==sum(weights)
        densities.append(h)
    fig,axes=panels('Frequency of a complete composition and of its least frequent constituent')
    cm=plt.get_cmap('magma_r').copy();cm.set_bad('white')
    norm=LogNorm(vmin=1,vmax=max(2,max(int(h.max()) for h in densities)))
    for k,(ax,h) in enumerate(zip(axes,densities),1):
        im=ax.pcolormesh(xb,yb,np.ma.masked_equal(h.T,0),cmap=cm,norm=norm,shading='flat',rasterized=True)
        ax.set(xscale='log',yscale='log',title=f'{k} hop')
        ax.set_xlim(xb[0],xb[-1]);ax.set_ylim(yb[0],yb[-1])
        if k==1:ax.set_ylabel('Composition frequency')
        else:ax.tick_params(labelleft=False)
    colorbar(fig,im,'Types')
    fig.text(.055,.124,'x = minimum training frequency among the constituent execution-unit types. Both axes are logarithmic.',fontsize=10)
    fig.text(.055,.063,'Colour counts distinct multi-unit composition types in each bin. All panels share the axes and colour scale.',fontsize=9,color='#414951')
    save(fig,out,'Figure_2_Component_vs_Composition_Frequency')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path)
    ap.add_argument('output',type=Path)
    a=ap.parse_args()
    plot(a.source,a.output)
