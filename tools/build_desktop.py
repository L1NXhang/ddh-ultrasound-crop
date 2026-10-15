"""Build the native Windows WebView2 host using the installed .NET compiler."""
import hashlib
import json
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.0.4258.31'
PACKAGE_SHA256 = '56f7f4b8bf9aee4b8efefbbdd4f67d5f74ebd1b100ed0806da71bf76af481aa9'
RUNTIME_FILES = ['lib/net462/Microsoft.Web.WebView2.Core.dll',
                 'lib/net462/Microsoft.Web.WebView2.WinForms.dll',
                 'runtimes/win-x64/native/WebView2Loader.dll', 'LICENSE.txt']


def dependencies():
    cache = ROOT/'data/desktop_build'
    cache.mkdir(parents=True, exist_ok=True)
    package = cache/f'webview2.{VERSION}.nupkg'
    url = f'https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/{VERSION}/microsoft.web.webview2.{VERSION}.nupkg'
    if not package.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            package.write_bytes(response.read())
    if hashlib.sha256(package.read_bytes()).hexdigest() != PACKAGE_SHA256:
        raise ValueError('WebView2 SDK 校验失败，请保留缓存并检查下载来源')
    runtime = ROOT/'desktop/runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package) as archive:
        for name in RUNTIME_FILES:
            path, content = runtime/Path(name).name, archive.read(name)
            if not path.exists() or path.read_bytes() != content:
                path.write_bytes(content)
    (runtime/'provenance.json').write_text(json.dumps({'version': VERSION, 'url': url,
                                                    'sha256': PACKAGE_SHA256}, indent=2), encoding='utf-8')
    return runtime


def icon():
    image = Image.new('RGBA', (256, 256))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, 248, 248), radius=48, fill='#b76143')
    draw.line((84, 61, 84, 175, 195, 175), fill='#fffefa', width=14)
    draw.line((61, 84, 175, 84, 175, 195), fill='#fffefa', width=14)
    path = ROOT/'desktop/app.ico'
    image.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return path


def build(destination=None):
    runtime = dependencies()
    destination = Path(destination) if destination else ROOT.parent/'DDH裁剪工具.exe'
    destination.parent.mkdir(parents=True, exist_ok=True)
    compiler = Path(os.environ['WINDIR'])/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    references = ['System.dll', 'System.Core.dll', 'System.Windows.Forms.dll', 'System.Drawing.dll',
                  'System.Net.Http.dll', 'System.Web.Extensions.dll']
    references += [str(runtime/name) for name in ('Microsoft.Web.WebView2.Core.dll', 'Microsoft.Web.WebView2.WinForms.dll')]
    command = [str(compiler), '/nologo', '/target:winexe', '/platform:x64', '/optimize+', '/utf8output',
               '/out:'+str(destination), '/win32manifest:'+str(ROOT/'desktop/app.manifest'),
               '/win32icon:'+str(icon())]
    command += ['/reference:'+r for r in references]
    command += [str(p) for p in sorted((ROOT/'desktop').glob('*.cs'))]
    subprocess.run(command, check=True, cwd=ROOT)
    destination.with_suffix('.exe.config').write_text(
        '<configuration><startup><supportedRuntime version="v4.0" sku=".NETFramework,Version=v4.8"/></startup></configuration>', encoding='utf-8')
    print('Desktop executable:', destination)


if __name__ == '__main__':
    build()
