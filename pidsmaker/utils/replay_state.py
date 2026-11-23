"""Persist inference-relevant non-state_dict state without changing native execution."""
from pathlib import Path
import random
import numpy as np
import torch

def save_replay_state(model,directory):
    modules={}
    for name,module in model.named_modules():
        state={}
        if hasattr(module,'_frozen'):state['_frozen']=sorted(module._frozen)
        for attr in ('msg_s_store','msg_d_store'):
            if hasattr(module,attr):
                # Constructor supplies empty entries. Preserve every nonempty pending message.
                state[attr]={k:tuple(t.detach().cpu().clone() for t in values) for k,values in getattr(module,attr).items() if any(t.numel() for t in values)}
        if state:modules[name]=state
    torch.save({'schema':1,'modules':modules,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'semantics':'checkpoint after native eval transition; inference_loop intentionally resets RNG with native set_seed'},Path(directory)/'replay_state.pt')

def restore_replay_state(model,path,required=False):
    path=Path(path)
    if not path.exists():
        if required:raise ValueError('Missing inference-relevant state artifact: '+str(path))
        return {'legacy_state':True,'reason':'Velox has no pending online or frozen hypersphere state'}
    payload=torch.load(path,map_location='cpu',weights_only=False)
    modules=dict(model.named_modules())
    for name,state in payload['modules'].items():
        module=modules[name]
        for attr,value in state.items():
            if attr=='_frozen':module._frozen=set(value)
            else:
                target=getattr(module,attr);device=next(module.buffers()).device
                target.update({k:tuple(t.to(device) for t in values) for k,values in value.items()})
    # Do not override the native inference_loop RNG reset or online val->test transitions.
    return {'restored_modules':list(payload['modules']),'rng_policy':payload['semantics']}
