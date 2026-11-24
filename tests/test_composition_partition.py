"""Exercise complete partitions and the factored event/relationship reconstruction."""
import csv
from collections import deque
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/composition_distribution'))
import duckdb
import numpy as np
from partition import audit, breadth_first, csr, partition


class PartitionTests(unittest.TestCase):
    def test_lazy_bfs_matches_projected_graph(self):
        rng=np.random.default_rng(7)
        for _ in range(12):
            n, nr = 23, 7
            arcs=rng.integers(n,size=(2,28))
            arcs=np.concatenate((arcs,arcs[::-1]),axis=1)
            u,r=np.nonzero(rng.random((n,nr))<.22)
            proc=csr(*arcs,n)
            resources=csr(u,r,n)
            members=csr(r,u,nr)
            neighbors=[set(arcs[1,arcs[0]==i]) for i in range(n)]
            for i,j in zip(u,r):
                neighbors[i].update(u[r==j])
            for direction in (1,-1):
                order=np.arange(n)[::direction]
                for h in range(1,6):
                    wanted={}
                    for seed in order:
                        if seed in wanted:continue
                        wanted[seed]=(seed,0);q=deque([seed])
                        while q:
                            k=q.popleft(); _,d=wanted[k]
                            if d==h:continue
                            for v in neighbors[k]-wanted.keys():
                                wanted[v]=(seed,d+1);q.append(v)
                    own,depth=breadth_first(order,proc,resources,members,h)
                    self.assertEqual(list(zip(own,depth)),[wanted[i] for i in range(n)])

    def test_archive_and_boundaries(self):
        with tempfile.TemporaryDirectory() as dirname:
            root=Path(dirname);db=duckdb.connect(str(root/'raw.duckdb'))
            db.execute('create table units(uid BIGINT,subject BIGINT,start_ns BIGINT,segment int)')
            db.execute('insert into units values (0,10,100,0),(1,11,110,0),(2,12,120,0),(3,13,130,0),(4,14,140,1)')
            db.execute('create table nodes(nid bigint,role varchar)')
            db.execute("insert into nodes values (10,'a'),(11,'a'),(12,'b'),(13,'c'),(14,'d'),(20,'f'),(21,'f'),(22,'f'),(23,'standalone')")
            db.execute('create table events(eid bigint,owner bigint,subject bigint,object1 bigint,object2 bigint,object1_uid bigint,object2_uid bigint,ts bigint,op varchar,uuid varchar,direction varchar)')
            db.execute("""insert into events values
               (1,0,10,11,null,1,null,104,'fork','e1','out'),
               (2,1,11,20,null,null,null,112,'write','e2','out'),
               (3,2,12,20,21,null,null,124,'read','e3','in'),
               (4,2,12,20,21,null,null,124,'read','e3','in'),
               (5,3,13,21,null,null,null,134,'read','e4','in'),
               (6,3,13,10,null,0,null,144,'signal','e5','out'),
               (7,4,14,20,null,null,null,145,'read','e6','in')""")
            db.execute('create table incidence(uid bigint,resource bigint,segment int)')
            db.execute('insert into incidence values (1,20,0),(2,20,0),(2,21,0),(3,21,0),(4,20,1)')
            db.execute('create table proc_edges(eid bigint,src_uid bigint,dst_uid bigint,ts bigint,op varchar)')
            db.execute("insert into proc_edges values (1,0,1,104,'fork')")
            db.close()
            partition(root);audit(root)
            with (root/'INTEGRITY_AUDIT.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),10)
            for row in rows:
                self.assertEqual(int(row['recovered_unique_events']),7)
                self.assertEqual(int(row['recovered_unique_nodes']),9)
                self.assertEqual(int(row['duplicate_event_owners']),0)
            self.assertEqual(int(rows[0]['cross_composition_events']),4)


if __name__=='__main__':
    unittest.main()
