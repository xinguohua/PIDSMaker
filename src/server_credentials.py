"""Read an existing project's DB credential into memory; never log its value."""
import json
import subprocess

def database_options(server):
    db = server['database']
    raw = subprocess.check_output(['docker', 'inspect', db['credential_container'], '--format', '{{json .Config.Env}}'], text=True)
    values = dict(item.split('=', 1) for item in json.loads(raw) if '=' in item)
    return dict(host=db['host'], port=db['port'], user=db['user'],
                password=values[db['credential_variable']], connect_timeout=10)
