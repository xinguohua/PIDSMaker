"""Select a repeated, identified unit type and a different type of the same program."""
import argparse
import json
from pathlib import Path

import duckdb


def produce(root):
    db=duckdb.connect(str(root/'raw.duckdb'), read_only=True)
    db.execute('SET threads=4')
    db.execute('CREATE TEMP TABLE types AS SELECT * FROM read_csv_auto(?)',[str(root/'UNIT_TYPES_BY_INSTANCE.csv')])
    db.execute('CREATE TEMP TABLE inventory AS SELECT * FROM read_csv_auto(?)',[str(root/'EXECUTION_UNIT_TYPES.csv')])
    db.execute("""CREATE TEMP TABLE options AS
      SELECT t.unit_type,u.uid,u.program,i.frequency,count(e.eid) n
      FROM units u JOIN types t USING(uid) JOIN inventory i USING(unit_type)
      JOIN events e ON e.owner=u.uid WHERE u.subject IS NOT NULL AND u.program<>'UNKNOWN_PROGRAM'
      GROUP BY ALL HAVING count(e.eid) BETWEEN 1 AND 24""")
    # Prefer examples where the raw operation count itself witnesses the difference.
    repeated=db.execute("""SELECT unit_type,program,frequency,n FROM options x
      WHERE frequency>1 AND EXISTS (SELECT 1 FROM options z WHERE z.program=x.program AND z.n<>x.n)
      GROUP BY ALL HAVING count(*)>=2 ORDER BY frequency DESC,n,unit_type LIMIT 1""").fetchone()
    if not repeated:
        raise ValueError('No compact identified repeated type with a same-program count difference')
    typ,program,freq,n=repeated
    same=db.execute('SELECT uid FROM options WHERE unit_type=? ORDER BY uid LIMIT 2',[typ]).fetchall()
    other=db.execute("""SELECT uid FROM options WHERE program=? AND unit_type<>? AND n<>?
        ORDER BY abs(n-?),frequency DESC,uid LIMIT 1""",[program,typ,n,n]).fetchone()
    def records(query,args):
        cursor=db.execute(query,args)
        names=[x[0] for x in cursor.description]
        return [dict(zip(names,r)) for r in cursor.fetchall()]
    shards={r['shard']:r['path'] for r in json.loads((root/'stage/source.json').read_text())}
    examples=[]
    for (uid,) in same+[other]:
        u=records("""SELECT u.uid,u.program,u.subject,u.start_ns,u.end_ns,u.start_status,t.unit_type,i.frequency
               FROM units u JOIN types t USING(uid) JOIN inventory i USING(unit_type) WHERE uid=?""",[uid])[0]
        u['events']=records("""SELECT e.eid,e.ts,e.op,e.direction,e.object1,a.kind AS kind1,a.role AS role1,
             e.object2,b.kind AS kind2,b.role AS role2,e.byte_offset,e.byte_length
             FROM events e LEFT JOIN nodes a ON e.object1=a.nid LEFT JOIN nodes b ON e.object2=b.nid
             WHERE owner=? ORDER BY ts,eid""",[uid])
        for e in u['events']:
            shard,line=divmod(e['eid'],10_000_000)
            e.update(source_file=shards[shard],source_line=line)
        examples.append(u)
    assert len(examples)==3 and examples[0]['unit_type']==examples[1]['unit_type']!=examples[2]['unit_type']
    assert len(examples[0]['events'])==len(examples[1]['events'])!=len(examples[2]['events'])
    (root/'EXECUTION_UNIT_EXAMPLES.json').write_text(json.dumps(examples,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    produce(parser.parse_args().root.resolve())
