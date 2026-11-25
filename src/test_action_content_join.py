"""Regression check through the window reader, typed counts and node grouping."""
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from action_paths import candidate_space, pack, paths
from action_path_check import check_real_window
import action_window as window
from action_index import ActionIndex
from unittest.mock import patch


class ContentJoinTest(unittest.TestCase):
    def test_reference_search_checks_every_event_in_a_tied_group(self):
        groups = [[(0,9,10,1), (1,0,1,2)], [(2,1,2,3)], [(3,2,3,4)]]
        audit = check_real_window(groups, (1,2,3), paths)
        self.assertEqual(audit['longest_path_event_rows'], [1,2,3])
        self.assertEqual(audit['exact_instances'], {1:3,2:2,3:1})
        groups = [[(0,0,1,2), (1,1,2,3)], [(2,2,3,4)]]
        audit = check_real_window(groups, (1,2,3), paths)
        self.assertFalse(audit['longest_requested_length_found'])
        self.assertEqual(audit['exact_instances'], {1:3,2:1,3:0})

    def test_sparse_output_lengths_advance_through_intermediate_steps(self):
        actions = {1:('a','x','a'), 2:('a','y','a')}
        self.assertEqual(candidate_space(actions, actions, {'a'}, (1,2,3,5)),
                         {1:2,2:4,3:8,5:32})

    def test_equal_content_across_native_ids_joins(self):
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
            with sqlite3.connect(root/'counts.sqlite') as counts:
                counts.execute('create table counts(k,seq,train,primary key(k,seq)) without rowid')
            window.configure(root,attrs,keys,dict.fromkeys(attrs),set(),{3,4},
                             {1,2},root/'counts.sqlite',(paths,check_real_window,pack))
            window.initialize()
            try:
                with patch.object(ActionIndex, 'witness', side_effect=AssertionError('eager witness')):
                    audit, rows, flags, _ = window.compute_one((1,'test'),tmp)
                groups, _, _, _ = window.read_window(
                    window._connection, 1, True, attrs, keys, dict.fromkeys(attrs), set())
                self.assertEqual(ActionIndex(groups).witness((1,2)), (0,2))
            finally:
                window._connection.close()
                window._support.close()
            pairs = [(b,n) for k,b,_,n,_ in window.result_rows(rows) if k==2]
            self.assertEqual(pairs, [(pack((1,2)), '3')])
            self.assertEqual(audit['instances_2'],3)
            self.assertEqual(set(flags),{(2,n) for n in (17,31,92,84,7)})

    def test_window_order_support_missing_and_exclusions(self):
        # Same ns with increasing sequence joins; same pair does not. ID 99 is
        # deliberately excluded; the identical missing signatures never join.
        attrs={1:('A',1),2:('B',1),3:('B',1),4:('C',1),5:('?',0),6:('?',0),99:('A',1)}
        keys={('A','x','B'):1,('B','y','C'):2,('A','z','?'):3,('?','x','C'):4}
        events=[(1,1,2,10,'x',0,0),(1,2,4,10,'y',1,0),
                (1,3,4,10,'y',0,0),(1,1,5,11,'z',0,0),
                (1,6,4,12,'x',0,0),(1,99,2,9,'x',0,0),
                (2,3,4,20,'y',0,0)]
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            with sqlite3.connect(root/'native.sqlite') as db, sqlite3.connect(root/'semantic.sqlite') as sem:
                db.execute('create table edges(w,row,u,v,ns,op,uuid,y)')
                sem.execute('create table source(w,row,seq)')
                for row,(w,u,v,ns,op,seq,y) in enumerate(events):
                    db.execute('insert into edges values(?,?,?,?,?,?,?,?)',(w,row,u,v,ns,op,row,y))
                    sem.execute('insert into source values(?,?,?)',(w,row,seq))
            with sqlite3.connect(root/'counts.sqlite') as db:
                db.execute('create table counts(k,seq,train text,primary key(k,seq)) without rowid')
                db.execute('insert into counts values(?,?,?)',(2,pack((1,2)),'9999999999999999999999999'))
            window.configure(root,attrs,keys,dict.fromkeys(set(attrs)-{99}),set(),{3,4},
                             {1,2},root/'counts.sqlite',(paths,check_real_window,pack))
            window.initialize()
            try:
                audit,rows,flags,_=window.compute_one((1,'test'),tmp)
                values=list(window.result_rows(rows))
                self.assertEqual(audit['unknown_label_edges'],1)
                self.assertEqual(audit['instances_2'],1)
                self.assertEqual(audit['semantic_unknown_instances_1'],2)
                self.assertEqual(flags,{(2,n):1 for n in (1,2,4)})
                self.assertEqual([n for k,_,_,n,_ in values if k==2],['1'])
                audit,rows,flags,_=window.compute_one((2,'test'),tmp)
                list(window.result_rows(rows))
                self.assertEqual(audit['instances_2'],0)
                self.assertFalse(flags)
            finally:
                window._connection.close()
                window._support.close()


if __name__=='__main__':
    unittest.main()
