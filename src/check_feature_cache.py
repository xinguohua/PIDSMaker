"""One real, already-generated window: byte-exact cache round trip, no inference."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,yaml
from baseline_matrix import digest,save,git
from pidsmaker.utils.feature_cache import save_feature_data,load_feature_data

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-cell',type=Path,required=True);a=p.parse_args()
    files=list((a.source_cell/'artifacts/feat_inference').rglob('*.TemporalData.simple'))
    source=min(files,key=lambda p:p.stat().st_size);before=digest(source)
    cfg=yaml.safe_load((a.source_cell/'resolved_config.yml').read_text())
    assert cfg['featurization']['used_method']=='word2vec'
    original=load_feature_data(source)
    width=cfg['dataset']['num_node_types']+cfg['featurization']['emb_dim']
    edge_width=original.msg.shape[1]-2*width
    out=Path(os.environ['RUN_OUTPUT_DIR']);target=out/'roundtrip.TemporalData.simple'
    save_feature_data(original,target,'node_table_v1',edge_width)
    restored=load_feature_data(target)
    checks={}
    for key in ('src','dst','t','y','msg'):
        left,right=getattr(original,key),getattr(restored,key)
        checks[key]=left.dtype==right.dtype and left.shape==right.shape and torch.equal(left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8))
    assert all(checks.values()) and digest(source)==before
    save(out/'summary.json',{'state':'real_window_lossless_verified','commit':git('rev-parse','HEAD'),
         'source':str(source),'source_sha256':before,'source_unchanged':True,
         'edges':original.src.numel(),'bitwise_equal':checks,'format':'node_table_v1',
         'dense_bytes':source.stat().st_size,'compact_bytes':target.stat().st_size,
         'training_or_inference_repeated':False})

if __name__=='__main__':main()
