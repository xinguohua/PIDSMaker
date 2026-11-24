"""Check unique edge ownership, directed native reconstruction and window separation."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from partition_native import audit, copy, native_units, partitions
from verify_native_depths import verify


def check(root):
    db=duckdb.connect()
    db.execute("""create temp table fixture(window_id int,nid bigint,kind varchar,label varchar,uuid varchar)""")
    db.executemany('insert into fixture values (?,?,?,?,?)',[
        (0,1,'subject','subject None master','a'),(0,2,'subject','subject None child','b'),
        (0,3,'file','file /shared','c'),(0,4,'netflow','netflow None None','d'),
        (0,5,'subject','subject None None','e'),(0,6,'file','file /isolated','f'),
        (1,3,'file','file /shared','c'),(1,2,'subject','subject None child','b')])
    copy(db,'select * from fixture',root/'NATIVE_NODES'/'a.parquet')
    copy(db,"""select eid,window_id,tensor_row,src,dst,ts,op,'same' edge_uuid,
       tensor_row nx_key,batch_id,tensor_row batch_row from (values
       (0,0,0,1,2,1,'EVENT_FORK',0),(1,0,1,3,2,2,'EVENT_READ',0),
       (2,0,2,3,2,2,'EVENT_READ',1),(3,0,3,1,3,3,'EVENT_WRITE',1),
       (4,0,4,3,4,4,'EVENT_WRITE',1),(4294967296,1,0,3,2,5,'EVENT_READ',2))
       v(eid,window_id,tensor_row,src,dst,ts,op,batch_id)""",root/'NATIVE_EDGES'/'a.parquet')
    db.close()
    native_units(root)
    with duckdb.connect(str(root/'native_units.duckdb'),read_only=True) as db:
        assert db.execute('select count(*) from units').fetchone()==(5,)
        assert db.execute('select count(*) from units where subject is null').fetchone()==(1,)
        assert db.execute("select count(*) from events where op='EVENT_READ' and direction='object_to_actor'").fetchone()==(3,)
        partitions(db,root)
        audit(db,root)
    verify(root)
    print('passed fixture',flush=True)


if __name__=='__main__':
    with TemporaryDirectory(dir=sys.argv[1] if len(sys.argv)>1 else None) as folder:
        check(Path(folder))
