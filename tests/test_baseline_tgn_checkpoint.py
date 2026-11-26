"""Real PIDSMaker TGN state regression, synthetic messages only; no training.
Run in the verified detector environment: python -m unittest discover -s tests -p test_baseline_tgn_checkpoint.py -v
Missing dependencies are explicitly skipped, never substituted with mocks.
"""
import ast
import os
from types import SimpleNamespace as NS
import importlib.util
from pathlib import Path
import tempfile
import unittest
from test_baseline_protocol import b, ROOT

AVAILABLE=all(importlib.util.find_spec(m) is not None for m in ('torch','torch_geometric'))

@unittest.skipUnless(AVAILABLE,'requires real torch and torch_geometric in detector environment')
class TGNCheckpointTests(unittest.TestCase):
    def test_chunked_flush_matches_simultaneous_update(self):
        import torch
        spec = importlib.util.spec_from_file_location('real_tgn', ROOT/'pidsmaker/tgn.py')
        tgn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tgn)
        torch.manual_seed(3)
        memory = tgn.TGNMemory(7, 2, 4, 2, tgn.IdentityMessage(2, 4, 2),
                               tgn.LastAggregator(), torch.device('cpu'))
        # Cross-chunk dependencies in both directions, repeated nodes, and an
        # untouched last chunk all have to match the original all-node GRU step.
        with torch.no_grad():
            memory.memory.normal_()
            memory.update_state(torch.tensor([0, 3, 1, 4]), torch.tensor([3, 0, 4, 1]),
                                torch.tensor([1, 2, 3, 4]), torch.randn(4, 2))
            nodes = torch.arange(memory.num_nodes)
            expected_memory, expected_update = memory._get_updated_memory(nodes)
            memory.flush_chunk_size = 2
            memory._flush_all_memory()
        torch.testing.assert_close(memory.memory, expected_memory)
        torch.testing.assert_close(memory.last_update, expected_update)


    def test_pending_messages_and_continuous_eval_roundtrip(self):
        import torch
        spec=importlib.util.spec_from_file_location('real_tgn',ROOT/'pidsmaker/tgn.py')
        tgn=importlib.util.module_from_spec(spec);spec.loader.exec_module(tgn)
        def build():
            return tgn.TGNMemory(4,2,4,2,tgn.IdentityMessage(2,4,2),tgn.LastAggregator(),torch.device('cpu'))
        torch.manual_seed(0)
        model=build()
        model.update_state(torch.tensor([0]),torch.tensor([1]),torch.tensor([1]),torch.tensor([[.3,.8]]))
        before=model.last_update.clone()
        with tempfile.TemporaryDirectory() as d:
            # Execute the actual final eval/save blocks, not a mocked training loop.
            tree=ast.parse((ROOT/'pidsmaker/detection/training_methods/training_loop.py').read_text())
            blocks=[n for n in ast.walk(tree) if isinstance(n,ast.If) and
                    (ast.unparse(n.test) == 'fixed_budget and epoch == num_epochs - 1' or
                     ast.unparse(n.test) == 'cfg._save_for_viz or (fixed_budget and epoch == num_epochs - 1)')]
            self.assertEqual(len(blocks),2)
            blocks.sort(key=lambda n:n.lineno)
            def save_model(m,path):
                Path(path).mkdir(parents=True)
                torch.save(m.state_dict(),Path(path)/'state_dict.pkl')
            scope=dict(model=model,fixed_budget=True,epoch=11,num_epochs=12,os=os,
                       cfg=NS(_save_for_viz=False,training=NS(_trained_models_dir=d)),save_model=save_model)
            exec(compile(ast.Module(body=blocks,type_ignores=[]),'<actual checkpoint blocks>','exec'),scope)
            self.assertFalse(torch.equal(before,model.last_update))
            checkpoint=Path(d)/'model_epoch_11/state_dict.pkl' 
            restored=b.load_evaluation_checkpoint(build(),checkpoint)
            for k,v in model.state_dict().items(): torch.testing.assert_close(v,restored.state_dict()[k])
            # Validation followed by test, with no intervening reset.
            for stamp in (2,3):
                for m in (model,restored):
                    m.eval()
                    m.update_state(torch.tensor([1]),torch.tensor([2]),torch.tensor([stamp]),torch.tensor([[.1,.7]]))
                for x,y in zip(model(torch.arange(4)),restored(torch.arange(4))):
                    torch.testing.assert_close(x,y)

if __name__=='__main__':unittest.main()
