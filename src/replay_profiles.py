"""Explicit native replay contracts. Implemented is distinct from empirically qualified."""
PROFILES={
 'velox':{'epochs':12,'encoder':'none','scoring':'edge','state':'linear weights; native fitted word2vec and unchanged feature tensors'},
 'ocrapt':{'epochs':100,'encoder':'rgcn_per_type','scoring':'node','state':'per-type c/r buffers plus _frozen set; native feature scaling; validation contamination calibration'},
 'orthrus':{'epochs':12,'encoder':'tgn,graph_attention','scoring':'edge','state':'TGN buffers and pending messages; native sampled neighbor graphs; fitted word2vec; native clustering'},
 'magic':{'epochs':12,'encoder':'magic_gat','scoring':'node','state':'seeded native masking; kNN fitted on benign training embeddings with training standardization'},
 'kairos':{'epochs':12,'encoder':'graph_attention,tgn','scoring':'edge','state':'TGN memory/last_update buffers and pending messages; native daily state resets and original node mapping'}
}
def validate_profile(name,cfg):
 p=PROFILES[name]
 if cfg.training.encoder.used_methods!=p['encoder']:raise ValueError('Unqualified encoder variant for '+name)
 if cfg.evaluation.used_method!='node_evaluation':raise ValueError('Only current node-level reported metrics are supported')
 if cfg.featurization.training_split!='train':raise ValueError('Feature fit partition is not train')
 return dict(model=name,**p)
