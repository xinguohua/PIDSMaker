"""Independent exhaustive event-instance oracle for the content-group engine."""
from collections import Counter, defaultdict
from itertools import combinations
import math
import random
import unittest
from action_index import ActionIndex, Participation, LENGTHS, signature


def oracle(groups, endpoints):
    events = [(g, *e) for g, batch in enumerate(groups) for e in batch]
    counts = {k: Counter() for k in LENGTHS}
    nodes = {k: defaultdict(set) for k in LENGTHS}
    for k in LENGTHS:
        for chain in combinations(events, k):
            if all(x[0] < y[0] and x[3] == y[2] for x, y in zip(chain, chain[1:])):
                seq = tuple(x[4] for x in chain)
                counts[k][seq] += 1
                nodes[k][seq].update(n for x in chain for n in endpoints[x[1]])
    return counts, nodes


class IndexTest(unittest.TestCase):
    def test_random_oracle_and_witness(self):
        rng = random.Random(73)
        for _ in range(80):
            # Includes self-loops, repeated Actions, disconnected missing endpoints.
            types = [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -2)]
            groups = [[] for _ in range(5)]
            endpoints = {}
            for eid in range(10):
                a = rng.randrange(len(types))
                u, v = types[a]
                # Multiple native instances may share one semantic type.
                u = u*2 + rng.randrange(2)
                v = v*2 + rng.randrange(2)
                groups[rng.randrange(5)].append((eid, u, v, a))
                endpoints[eid] = (2*eid, 2*eid+1)  # Distinct actual IDs throughout.
            expected, members = oracle(groups, endpoints)
            index = ActionIndex(groups)
            event_map = {e[0]: (g, e) for g, batch in enumerate(groups) for e in batch}
            for k in LENGTHS:
                self.assertEqual(dict(index.sequences(k)), expected[k])
                participation = Participation()
                for seq in expected[k]:
                    self.assertEqual(index.nodes(seq, endpoints), members[k][seq])
                    witness = index.witness(seq)
                    self.assertEqual(len(set(witness)), k)
                    chain = [event_map[eid] for eid in witness]
                    self.assertEqual(tuple(e[3] for _, e in chain), seq)
                    self.assertTrue(all(g < h and x[2] == y[1] for (g,x),(h,y) in zip(chain,chain[1:])))
                    participation.add(index, seq, 2)
                flags = participation.flags(index, endpoints)
                self.assertEqual(set(flags), set().union(*members[k].values()) if members[k] else set())

    def test_parallel_processes_never_cross(self):
        groups = [[(1,10,17,1),(2,11,92,2)],[(3,17,20,3),(4,92,21,4)]]
        self.assertEqual(dict(ActionIndex(groups).sequences(2)),{(1,3):1,(2,4):1})
        groups = [[(1,10,17,1),(2,11,92,1)],[(3,17,20,3),(4,92,21,3)]]
        self.assertEqual(dict(ActionIndex(groups).sequences(2)),{(1,3):2})

    def test_large_integers(self):
        index = ActionIndex([[(g, 0, 0, 1)] for g in range(6)])
        # Compressed multiplicities represent 10^6 actual rows per group.
        index.counts[1] = [10**6]*6
        index.suffix[1] = [(6-g)*10**6 for g in range(7)]
        result = dict(index.sequences(3))[(1,)*3]
        self.assertEqual(result, math.comb(6,3)*10**18)
        self.assertGreater(result, 2**64)

    def test_actual_event_count_over_32_bits(self):
        n=4000
        index=ActionIndex([[(g,0,0,1)] for g in range(n)])
        result=dict(index.sequences(3))[(1,)*3]
        self.assertEqual(result,math.comb(n,3))
        self.assertGreater(result,2**32)

    def test_file_path_completeness(self):
        from action_composition import semantic_complete
        for paths,expected in (([],False),([' <UNKNOWN> '],False),(['<unknown>','/tmp/report'],True),
                               (['/tmp/<unknown>'],True)):
            self.assertEqual(semantic_complete(dict(kind='file',paths=paths,commands=[])),expected)

    def test_signature_lossless(self):
        self.assertEqual(signature('{"b": ["雪", "a"], "a": null}'),
                         signature('{"a":null,"b":["雪","a"]}'))
        self.assertNotEqual(signature('{"a":null}'), signature('{"a":""}'))
        self.assertNotEqual(signature('{"a":["x","y"]}'), signature('{"a":["x,y"]}'))

    def test_window_isolation(self):
        left = ActionIndex([[(0,0,1,1)]])
        right = ActionIndex([[(1,1,2,2)]])
        self.assertFalse(dict(left.sequences(2)))
        self.assertFalse(dict(right.sequences(2)))

class PersistenceTest(unittest.TestCase):
    def test_atomic_recovery_and_big_counts(self):
        import sqlite3
        from action_paths import commit_window, pack
        db=sqlite3.connect(':memory:')
        db.create_function('bigadd',2,lambda a,b:str(int(a)+int(b)))
        db.executescript('''create table counts(k,seq,train text,test text,w,primary key(k,seq));
            create table node_flags(k,id,bits,primary key(k,id));
            create table window_done(w primary key,audit);''')
        row=(3,pack((1,)*3),str(10**40),'0',1)
        def interrupted():
            yield row
            raise RuntimeError('simulated interrupted stream')
        with self.assertRaises(RuntimeError):
            commit_window(db,dict(window=1,split='train'),interrupted(),{(3,1):1})
        self.assertEqual(db.execute('select count(*) from counts').fetchone()[0],0)
        self.assertTrue(commit_window(db,dict(window=1,split='train'),[row],{(3,1):1}))
        self.assertFalse(commit_window(db,dict(window=1,split='train'),[row],{(3,1):2}))
        self.assertTrue(commit_window(db,dict(window=2,split='train'),[row],{(3,1):2}))
        self.assertEqual(db.execute('select train from counts').fetchone()[0],str(2*10**40))
        self.assertEqual(db.execute('select bits from node_flags').fetchone()[0],3)
        db.close()

    def test_protocol_rejects_legacy_and_changed_input(self):
        import sqlite3
        from action_paths import bind_protocol
        with sqlite3.connect(':memory:') as db:
            bind_protocol(db,{'protocol':'v2','input':'a'})
            bind_protocol(db,{'protocol':'v2','input':'a'})
            with self.assertRaises(ValueError):bind_protocol(db,{'protocol':'v2','input':'b'})
        with sqlite3.connect(':memory:') as db:
            db.execute('create table counts(k)')
            with self.assertRaises(ValueError):bind_protocol(db,{'protocol':'v2'})

    def test_candidates_and_category_precedence(self):
        from action_paths import candidate_space,node_category
        from itertools import product
        actions={1:('a','x','b'),2:('b','y','a'),3:('b','z','b'),4:('b','x','?'),5:('?','x','a')}
        actual=candidate_space(actions,set(actions),{'a','b'})
        for k in LENGTHS:
            expected=sum(all(actions[x][2] in {'a','b'} and actions[x][2]==actions[y][0]
                             for x,y in zip(seq,seq[1:])) for seq in product(actions,repeat=k))
            self.assertEqual(actual[k],expected)
        self.assertEqual([node_category(b) for b in (0,1,3,5,9,10,14)],
                         ['Other','Seen','Zero-shot','Other','Other','Zero-shot','Zero-shot'])

    def test_duplicate_event_rejected(self):
        with self.assertRaises(ValueError):ActionIndex([[(1,0,0,1)],[(1,0,0,1)]])


if __name__ == '__main__':
    unittest.main()
