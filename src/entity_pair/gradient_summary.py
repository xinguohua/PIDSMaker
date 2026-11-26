"""Summarize paired frozen-checkpoint gradients, weighting each matched FP equally."""
from pathlib import Path
import sys
import pandas as pd


def summarize(root, gradient_file, out):
    first=root/"pair-interference-s1-20261126a/output"
    nodes=pd.read_pickle(first/"queries.pkl").query("canonical").drop_duplicates("node")
    matches=pd.read_csv(first/"matched.csv")
    gradients=pd.read_csv(gradient_file,keep_default_na=False)
    evidence=nodes[["node","src","dst"]].merge(gradients,on=["src","dst"])
    evidence["conflict"]=evidence.cosine<0
    evidence["dominates"]=evidence.dominance>1
    evidence["batch_harm"]=evidence.batch_dot<0
    cols=["cosine","conflict","norm","projection","dominance","dominates","batch_harm"]
    rows=[]
    for group,g in evidence.groupby("group"):
        valid=matches[matches.fp_node.isin(g.node)&matches.tn_node.isin(g.node)]
        for side in ["fp","tn"]:
            d=valid.merge(g,left_on=side+"_node",right_on="node")
            rows.append(dict(group=group,side=side,nodes=d.node.nunique(),fp_sets=d.fp_node.nunique(),
                             contexts=len(d[["src","dst"]].drop_duplicates()),**d.groupby("fp_node")[cols].mean().mean().to_dict()))
    pd.DataFrame(rows).to_csv(out,index=False)


if __name__=="__main__":
    summarize(*map(Path,sys.argv[1:]))
