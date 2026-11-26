import json
import sqlite3
from types import SimpleNamespace

from first_file_paths import build
from pidsmaker.preprocessing.file_paths import effective_file_path, read_evidence


def test_missing_file_path_requires_evidence_on_first_training_event(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path/"data"))
    def db(name, script):
        con = sqlite3.connect(tmp_path/name)
        con.executescript(script)
        return con

    db('events', """
        create table nodes(id,uuid,kind,path);
        insert into nodes values(1,'a','file',null),(2,'b','file','None'),(3,'c','file',null),(4,'d','file',null);
        create table events(src,dst,ns,uuid);
        insert into events values(1,0,1,'A'),(2,0,1,'before'),(2,0,2,'B'),
          (3,0,1,'C'),(3,0,1,'C2'),(4,0,1,'D');
    """).close()
    db('native', """
        create table windows(w,split); insert into windows values(1,'train'),(2,'test');
        create table edges(w,row,u,v,ns,uuid,y);
        insert into edges values(1,0,1,0,1,'A',0),(1,1,2,0,2,'B',0),(1,2,3,0,1,'C',0),(2,3,4,0,1,'D',0);
    """).close()
    sem = db('sem', 'create table source(w,row,file,offset);')
    path = tmp_path/'event'
    event = {'uuid':'A','timestampNanos':1,'predicateObject':{'UUID':'a'},'predicateObjectPath':{'string':'/first'}}
    path.write_text(json.dumps({'datum':{'com.bbn.tc.schema.avro.cdm18.Event':event}}))
    sem.executemany('insert into source values(1,?,?,0)',[(i,str(path)) for i in range(3)])
    sem.commit()
    sem.close()
    dates = ['1970-01-01']
    build(tmp_path/'events',tmp_path/'native',tmp_path/'sem',
          dict(dataset='test',task_id='t',commit='c',protocol=dict(timezone='UTC',train_dates=dates)),tmp_path/'out')
    cfg=SimpleNamespace(dataset=SimpleNamespace(name='test',train_dates=dates))
    mapping=read_evidence(cfg)
    assert mapping=={'a':'/first'}
    assert effective_file_path('a','/known',mapping)=='/known'
    assert effective_file_path('a',None,mapping)=='/first'
