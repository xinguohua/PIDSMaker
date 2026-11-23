import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from baseline_queue import initialize,register,accept,REQUIRED,available_devices,worker_environment,reclaim_accepted_caches
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
 def test_two_slots_remain_reserved_through_verification(self):
  devices=[{'physical_index':0},{'physical_index':1},{'physical_index':2}]
  rows=[{'active_run_id':'training','device_assignment':devices[1]}, {'active_run_id':'checking','state':'pipeline_finished_pending_acceptance','device_assignment':devices[0]}]
  self.assertEqual(available_devices(rows,devices),[])
  rows[1]['active_run_id']=None
  self.assertEqual(available_devices(rows,devices),[devices[0]])
  self.assertEqual(worker_environment(devices[1])['CUDA_VISIBLE_DEVICES'],'1')
  self.assertEqual(available_devices(rows,[devices[0]]),[])
if __name__=='__main__':unittest.main()

class RepairAcceptanceTests(unittest.TestCase):
 def test_missing_coverage_is_rejected_even_after_repair(self):
  row={'accepted_run_id':None};attempt={'run_id':'derived'}
  evidence={'full_accepted':True,'checks':dict.fromkeys(REQUIRED,True)}
  del evidence['checks']['independent_coverage']
  with self.assertRaises(ValueError):accept(row,attempt,evidence)
 def test_failed_original_history_survives_derived_acceptance(self):
  row={'accepted_run_id':None,'attempt_ids':['original','derived'],'attempts':[{'run_id':'original','failure':'1024 overwritten scores'},{'run_id':'derived'}]}
  accept(row,row['attempts'][1],{'full_accepted':True,'checks':dict.fromkeys(REQUIRED,True)})
  self.assertEqual(row['attempts'][0]['failure'],'1024 overwritten scores');self.assertEqual(row['accepted_run_id'],'derived')

class CacheReclamationTests(unittest.TestCase):
 def test_pending_reference_and_active_worker_protect_cache(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);path=root/'baseline-matrix/matrix.json';doc=initialize(path);row=doc['rows'][0]
   run=root/'accepted';cell=run/'output/CADETS_E3-velox-full'
   nx=cell/'artifacts/construction/CADETS_E3/construction/hash/nx';nx.mkdir(parents=True);(nx/'graph.pt').write_bytes(b'graph')
   feature=cell/'artifacts/feat_inference';feature.mkdir();(feature/'a.TemporalData.simple').write_bytes(b'tensor')
   retained=[cell/'objects.jsonl',cell/'uuid_mapping.json',cell/'resolved_config.yml',cell/'artifacts/featurization/model.pt',cell/'artifacts/training/model.pt',nx.parent/'indexid2msg/mapping.json']
   for p in retained:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'keep')
   register(row,run);accept(row,row['attempts'][0],{'full_accepted':True,'checks':dict.fromkeys(REQUIRED,True),'source_cell':str(cell)})
   pending=doc['rows'][1];pending_run=root/'pending';register(pending,pending_run)
   link=pending_run/'output/CADETS_E3-ocrapt-full/artifacts/construction/CADETS_E3/construction/hash/nx';link.parent.mkdir(parents=True);link.symlink_to(nx)
   reclaim_accepted_caches(path,doc);self.assertFalse(feature.exists());self.assertTrue(nx.exists())
   pending['active_run_id']=None;pending['state']='prepared'
   reclaim_accepted_caches(path,doc);self.assertTrue(nx.exists());self.assertFalse(feature.exists())
   link.unlink();reclaim_accepted_caches(path,doc);self.assertFalse(nx.exists())
   self.assertTrue(all(p.read_bytes()==b'keep' for p in retained))
   self.assertEqual(len(row['cache_reclamation_receipts']),2)
   for p in row['cache_reclamation_receipts']:self.assertTrue(json.loads(Path(p).read_text())['deleted'])
