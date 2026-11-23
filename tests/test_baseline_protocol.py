"""CPU-only protocol tests; deliberately do not import training dependencies."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('baseline_matrix', ROOT/'src/baseline_matrix.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class ProtocolTests(unittest.TestCase):
    def test_matrix_and_disjoint_dates(self):
        rows=b.matrix(); self.assertEqual(len({(r['dataset'],r['model']) for r in rows}),15)
        self.assertEqual((rows[0]['dataset'],rows[0]['model']),('CADETS_E3','velox'))
        for p in b.protocol().values():
            days=[set(p[x+'_dates']) for x in ('train','val','test','unused')]
            for i,s in enumerate(days):
                for t in days[i+1:]: self.assertFalse(s&t)
    def test_metrics_and_na(self):
        rows=[dict(node_id=i,gt=y,prediction=p) for i,(y,p) in enumerate([(1,1),(0,1),(0,0),(1,0)])]
        m=b.metrics(rows)
        for k in ('precision','recall','f1','fpr','fnr'): self.assertEqual(m[k],.5)
        self.assertIsNone(b.metrics([])['f1'])
        with self.assertRaises(ValueError): b.metrics(rows+rows[:1])
    def test_test_metrics_cannot_select_fixed_budget_epoch(self):
        path=ROOT/'pidsmaker/tasks/evaluation.py'
        tree=ast.parse(path.read_text()); f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='best_metric_pick_best_epoch')
        scope={};exec(compile(ast.Module(body=[f],type_ignores=[]),str(path),'exec'),scope)
        cfg=SimpleNamespace(evaluation=SimpleNamespace(best_model_selection='fixed_budget'),training=SimpleNamespace(num_epochs=12))
        pick=scope[f.name]
        with self.assertRaises(ValueError):pick({'epoch':9,'ap':1.0},{'stats':None},cfg)
        s={'epoch':11,'ap':0.0}; self.assertIs(pick(s,{'stats':None},cfg)['stats'],s)
    def test_final_epoch_condition_handles_odd_budgets(self):
        path=ROOT/'pidsmaker/detection/training_methods/training_loop.py'
        tree=ast.parse(path.read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.IfExp))
        code=compile(ast.Expression(node.test),str(path),'eval')
        for budget in (1,3,12,100):
            selected=[e for e in range(budget) if eval(code,dict(epoch=e,num_epochs=budget,fixed_budget=True))]
            self.assertEqual(selected,[budget-1])

if __name__=='__main__': unittest.main()
