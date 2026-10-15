"""Start or reopen the local studio without leaving a console window open."""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = 'http://127.0.0.1:8770/'


def ready():
    try:
        with urllib.request.urlopen(URL+'api/bootstrap', timeout=1) as response:
            value = json.load(response)
            return isinstance(value.get('status'), dict) and 'token' in value
    except (OSError, ValueError, urllib.error.URLError):
        return False


def main():
    if not ready():
        logs = ROOT/'data/studio_runtime'
        logs.mkdir(parents=True, exist_ok=True)
        python = Path(sys.executable).with_name('pythonw.exe')
        if not python.exists():
            python = Path(sys.executable)
        with (logs/'server.log').open('ab') as output:
            subprocess.Popen([str(python), '-u', str(ROOT/'tools/crop_studio.py')], cwd=ROOT,
                             stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        for _ in range(60):
            if ready():
                break
            time.sleep(.25)
        else:
            raise RuntimeError('工作台未启动，请检查8770端口和 data/studio_runtime/server.log')
    url = URL
    if len(sys.argv) > 1:
        url += '?source='+urllib.parse.quote(str(Path(sys.argv[1]).resolve()))
    webbrowser.open(url)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
