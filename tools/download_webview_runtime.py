"""Cache the x64 fixed WebView2 runtime linked by Microsoft's official download page."""
import hashlib
import json
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'data/portable_build/webview'
PAGE = 'https://developer.microsoft.com/en-us/microsoft-edge/webview2/'


def fetch():
    CACHE.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(PAGE, timeout=30) as response:
        page = response.read().decode('utf-8').replace('\\u002F', '/')
    urls = re.findall(r'https://msedge\.sf\.dl\.delivery\.mp\.microsoft\.com/[^"<>\s]+/Microsoft\.WebView2\.FixedVersionRuntime\.[\d.]+\.x64\.cab', page)
    if not urls:
        raise ValueError('Official x64 runtime link not found')
    url = urls[0]
    cab = CACHE/url.rsplit('/', 1)[1]
    if not cab.exists():
        temporary = cab.with_suffix('.cab.tmp')
        print('Downloading official WebView2: '+cab.name, flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream, 1024*1024)
        temporary.replace(cab)
    folder = CACHE/cab.stem
    if not list(folder.rglob('msedgewebview2.exe')):
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.run(['expand.exe', str(cab), '-F:*', str(folder)], check=True, stdout=subprocess.DEVNULL)
    executables = list(folder.rglob('msedgewebview2.exe'))
    if len(executables) != 1:
        raise ValueError('Unexpected fixed runtime directory structure')
    command = "$s=Get-AuthenticodeSignature -LiteralPath $env:DDH_RUNTIME_EXE; [pscustomobject]@{status=[string]$s.Status;subject=$s.SignerCertificate.Subject}|ConvertTo-Json -Compress"
    import os
    signature = subprocess.check_output([shutil.which('pwsh') or 'powershell.exe', '-NoProfile', '-Command', command],
                                        env={**os.environ, 'DDH_RUNTIME_EXE': str(executables[0])},
                                        encoding='utf-8', creationflags=subprocess.CREATE_NO_WINDOW)
    signed = json.loads(signature)
    if signed['status'] != 'Valid' or 'Microsoft Corporation' not in signed['subject']:
        raise ValueError('WebView2 executable publisher signature invalid')
    provenance = {'official_page': PAGE, 'url': url, 'cab_sha256': hashlib.sha256(cab.read_bytes()).hexdigest(),
                  'signature': signed, 'runtime_folder': str(executables[0].parent)}
    (CACHE/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance), flush=True)
    return executables[0].parent


if __name__ == '__main__':
    fetch()
