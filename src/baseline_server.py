"""Launch the baseline CLI with existing private server credentials in memory."""
import argparse,json,os,sys
from pathlib import Path
from server_credentials import database_options
p=argparse.ArgumentParser(add_help=False);p.add_argument('--server-config',type=Path,required=True)
a,_=p.parse_known_args()
server=json.loads(a.server_config.read_text())
os.environ[server['database']['password_env']]=database_options(server)['password']
os.environ.setdefault('NLTK_DATA',str(Path(server['data_disk_root'])/'nsas2020/tdsc-envs/nltk_data'))
from baseline_matrix import cli
cli()
