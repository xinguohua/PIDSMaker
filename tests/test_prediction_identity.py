import importlib.util,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'pidsmaker/utils/prediction_identity.py'
s=importlib.util.spec_from_file_location('prediction_identity',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class IdentityTests(unittest.TestCase):
 def test_actual_nanosecond_collision(self):
  interval='2018-04-12 11:52:09.266362231~2018-04-12 11:54:11.446358030'
  a=m.prediction_name(interval,0,1408);b=m.prediction_name(interval,0,1409)
  self.assertNotEqual(a,b);self.assertEqual(m.timestamp_part(a),interval);self.assertEqual(m.timestamp_part(b),interval)
 def test_legacy_unchanged(self):self.assertEqual(m.timestamp_part('2018-04-02 00:00:00.123456789'),'2018-04-02 00:00:00.123456789')
 def test_stable_order(self):self.assertLess(m.prediction_name('t',0,9),m.prediction_name('t',0,10))

 def test_evaluation_reads_batch_filenames_with_extension(self):
  end = '2018-04-12 11:54:11.446358030'
  interval = '2018-04-12 11:52:09.266362231~' + end
  for stem in (interval, m.prediction_name(interval, 0, 1408), m.prediction_name(interval, 0, 1409)):
   with self.subTest(stem=stem):
    filename = stem + '.csv'
    self.assertEqual(m.timestamp_part(filename), interval)
    self.assertEqual(m.timestamp_part(filename.split('~')[1]), end)
