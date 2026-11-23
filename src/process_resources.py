"""Linux process-tree RSS and NVIDIA device/process memory, independent of tracemalloc."""
import json,os,shutil,signal,subprocess,sys,time
from pathlib import Path

def sample(root_pid,output,initial_free,started):
    processes={}
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:
            fields=dict(line.split(':',1) for line in (d/'status').read_text().splitlines() if ':' in line)
            processes[int(d.name)]={'ppid':int(fields['PPid']),'rss_bytes':int(fields.get('VmRSS','0 kB').split()[0])*1024}
        except (OSError,ValueError,KeyError):continue
    selected={root_pid};previous=set()
    while selected!=previous:
        previous=set(selected);selected.update(pid for pid,s in processes.items() if s['ppid'] in selected)
    tree={str(pid):processes[pid] for pid in selected if pid in processes}
    gpu={}
    for label,query in [('device','--query-gpu=index,uuid,memory.used,memory.total'),('process','--query-compute-apps=pid,gpu_uuid,used_gpu_memory')]:
        try:
            p=subprocess.run(['nvidia-smi',query,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
            gpu[label]={'exit_code':p.returncode,'csv':p.stdout.strip(),'units':'MiB; device includes other jobs, process rows identify attribution'}
        except (OSError,subprocess.TimeoutExpired) as e:gpu[label]={'unavailable':type(e).__name__}
    free=shutil.disk_usage(output).free
    return {'unix_time':time.time(),'elapsed_seconds':time.monotonic()-started,'processes':tree,'sum_tree_rss_bytes':sum(p['rss_bytes'] for p in tree.values()),'rss_note':'Sum RSS includes shared pages multiple times; not PSS or Python allocator peak','nvidia':gpu,'filesystem_free_bytes':free,'filesystem_net_growth_bytes':initial_free-free,'disk_note':'Filesystem delta includes concurrent jobs; not attributed run size','reserve_bytes':4*1024**3}

def run(command,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    initial=shutil.disk_usage(output).free;start=time.monotonic()
    if initial<4*1024**3:raise RuntimeError('Below 4GiB disk reserve')
    child=subprocess.Popen(command,start_new_session=True)
    peak=0;stopped=False
    with (output/'process_resources.jsonl').open('x') as f:
        while True:
            record=sample(child.pid,output,initial,start);peak=max(peak,record['sum_tree_rss_bytes']);f.write(json.dumps(record)+'\n');f.flush()
            if record['filesystem_free_bytes']<4*1024**3 and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM);stopped=True
            if child.poll() is not None:break
            time.sleep(2)
    result={'exit_code':child.returncode,'disk_guard_triggered':stopped,'sampled_peak_tree_rss_bytes':peak,'elapsed_seconds':time.monotonic()-start,'sampling_interval_seconds':2,'not_continuous_peak':True}
    (output/'process_resources_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return child.returncode
if __name__=='__main__':
    # Output directory precedes --, the rest is an exact argument vector.
    raise SystemExit(run(sys.argv[3:],sys.argv[1]))
