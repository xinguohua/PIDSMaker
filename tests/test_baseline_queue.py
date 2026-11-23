import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from baseline_queue import initialize,register,accept,REQUIRED
from baseline_matrix import prepare
class QueueTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.path=self.root/'matrix.json';self.doc=initialize(self.path);self.row=self.doc['rows'][0]
 def tearDown(self):self.tmp.cleanup()
 def test_fifteen_and_order(self):
  self.assertEqual(len(self.doc['rows']),15);self.assertEqual(self.row['model'],'velox');self.assertTrue(all(r['seed']==0 for r in self.doc['rows']))
 def test_no_duplicate_active(self):
  register(self.row,self.root/'a');register(self.row,self.root/'a');self.assertEqual(self.row['attempt_ids'],['a'])
  with self.assertRaises(ValueError):register(self.row,self.root/'b')
 def test_smoke_and_incomplete_rejected(self):
  register(self.row,self.root/'a')
  for ev in [{'full_accepted':False},{'full_accepted':True,'checks':{}}]:
   with self.assertRaises(ValueError):accept(self.row,self.row['attempts'][0],ev)
  self.assertIsNone(self.row['accepted_run_id'])
 def test_failure_history_preserved(self):
  register(self.row,self.root/'a');self.row['attempts'][0]['failure']='interrupted';self.row['active_run_id']=None;register(self.row,self.root/'b')
  self.assertEqual(self.row['attempt_ids'],['a','b']);self.assertEqual(self.row['attempts'][0]['failure'],'interrupted')
 def test_existing_prepare_not_reset(self):
  before=self.path.read_bytes();prepare(self.root);self.assertEqual(before,self.path.read_bytes());self.assertEqual(initialize(self.path),self.doc)
 def test_accept_first_only(self):
  register(self.row,self.root/'a');accept(self.row,self.row['attempts'][0],{'full_accepted':True,'checks':dict.fromkeys(REQUIRED,True)})
  self.assertEqual(self.row['state'],'completed')
  with self.assertRaises(ValueError):register(self.row,self.root/'b')
if __name__=='__main__':unittest.main()
