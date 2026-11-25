"""Regression check through the window reader, typed counts and node grouping."""
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from action_paths import pack, paths
from action_path_check import check_real_window
import action_window as window


class ContentJoinTest(unittest.TestCase):
    def test_cross_instance_join_and_membership(self):
        attrs = {17: ('A', 1), 92: ('A', 1), 31: ('B', 1), 84: ('B', 1),
                 7: ('C', 1), 8: ('unknown', 0), 9: ('unknown', 0)}
        events = [(17, 31, 'write', 1), (92, 84, 'write', 2),
                  (84, 7, 'read', 2), (84, 7, 'read', 3),
                  (17, 8, 'write', 4), (9, 7, 'read', 5)]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with sqlite3.connect(root/'native.sqlite') as db, sqlite3.connect(root/'semantic.sqlite') as sem:
                db.execute('create table edges(w,row,u,v,ns,op,uuid,y)')
                sem.execute('create table source(w,row,seq)')
                for i, (u,v,op,ns) in enumerate(events):
                    db.execute('insert into edges values(1,?,?,?,?,?,?,0)',(i,u,v,ns,op,i))
                    sem.execute('insert into source values(1,?,0)',(i,))
            keys = {('A','write','B'):1, ('B','read','C'):2,
                    ('A','write','unknown'):3, ('unknown','read','C'):4}
            window.configure(root,attrs,keys,dict.fromkeys(attrs),set(),{3,4},
                             {1,2},{2:set(),3:set(),5:set()},(paths,check_real_window,pack))
            window.initialize()
            try:
                audit, rows, flags, _ = window.compute_one((1,'test'),tmp)
            finally:
                window._connection.close()
            pairs = [(b,n) for k,b,_,n,_,_ in rows if k==2]
            self.assertEqual(pairs, [(pack((1,2)), '3')])
            self.assertEqual(audit['instances_2'],3)
            self.assertEqual(set(flags),{(2,n) for n in (17,92,31,84,7)})


if __name__=='__main__':
    unittest.main()
