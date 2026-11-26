"""Lossless window-local storage for repeated static node feature vectors."""
import torch


def save_feature_data(data, path, cache_format='dense', edge_width=None):
    if cache_format == 'dense' or (cache_format == 'node_table_v1' and 'msg_node_feats' in data):
        torch.save(data, path)
        return
    if cache_format != 'node_table_v1':
        raise ValueError('Unknown feature cache format: ' + cache_format)
    msg = data.msg
    assert msg.dtype == torch.float32 and msg.device.type == 'cpu'
    width = (msg.shape[1] - edge_width) // 2
    assert width > 0 and 2 * width + edge_width == msg.shape[1]
    ids, inverse = torch.unique(torch.cat((data.src, data.dst)), return_inverse=True)
    n = data.src.numel()
    nodes = torch.empty((ids.numel(), width), dtype=msg.dtype)
    nodes[inverse[:n]] = msg[:, :width]
    nodes[inverse[n:]] = msg[:, -width:]
    # Compare bits, including signed zero, before choosing this representation.
    for original, indices in [(msg[:, :width], inverse[:n]), (msg[:, -width:], inverse[n:])]:
        assert torch.equal(nodes[indices].contiguous().view(torch.int32), original.contiguous().view(torch.int32)), 'Node features vary within window'
    edges, edge_index = torch.unique(msg[:, width:width+edge_width].contiguous().view(torch.int32), dim=0, return_inverse=True)
    assert edges.shape[0] <= edge_width + 1, 'Expected native one-hot edge types'
    payload = {'format':cache_format, 'data':data, 'nodes':nodes,
               'src_index':inverse[:n].clone(), 'dst_index':inverse[n:].clone(),
               'edge_types':edges.view(torch.float32), 'edge_index':edge_index}
    try:
        data.msg = None
        torch.save(payload, path)
    finally:
        data.msg = msg


def load_feature_data(path, *, expand=True):
    payload = torch.load(path)
    if not isinstance(payload, dict) or payload.get('format') != 'node_table_v1':
        if expand and hasattr(payload, 'msg_node_feats'):
            from pidsmaker.utils.data_utils import expand_compact_msg
            return expand_compact_msg(payload)
        return payload
    data = payload['data']
    data.msg = torch.cat((payload['nodes'][payload['src_index']],
                          payload['edge_types'][payload['edge_index']],
                          payload['nodes'][payload['dst_index']]), dim=1)
    return data
