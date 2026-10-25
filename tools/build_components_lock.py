"""生成 components.lock.json：下载锁定的 wheel，计算 sha256，记录镜像 URL。

只允许改动下面 PINNED 里的版本号来升级组件；wheel 文件名必须与 PINNED 完全一致。
产物：
  data/components_cache/<wheel 文件>          原始 wheel（构建缓存，可删）
  tools/webapp/../components.lock.json?      不——写 data/components.lock.json? 也不。
  实际写到 tools/components.lock.json 随包分发（打包脚本从 ROOT 引用）。

用法：python tools/build_components_lock.py [--skip-existing]
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'data/components_cache'
LOCK = ROOT/'components.lock.json'
CHUNK = 1024*1024

PYTORCH_TAG = {'cpu': 'cpu', 'cuda': 'cu121'}
PYTORCH_MIRRORS = ['https://mirrors.aliyun.com/pytorch-wheels/{tag}/',
                   'https://download.pytorch.org/whl/{tag}/']
PYPI_INDEX = ['https://mirrors.aliyun.com/pypi/simple/{package}/',
              'https://pypi.org/simple/{package}/']

# 版本与开发环境（.venv，torch 2.5.1+cu121）完全一致；升级时成对修改。
TORCH = 'torch-2.5.1+{tag}-cp311-cp311-win_amd64.whl'
TORCHVISION = 'torchvision-0.20.1+{tag}-cp311-cp311-win_amd64.whl'
DEPS = ['sympy-1.13.1-py3-none-any.whl',
        'mpmath-1.3.0-py3-none-any.whl',
        'networkx-3.6.1-py3-none-any.whl',
        'jinja2-3.1.6-py3-none-any.whl',
        'MarkupSafe-3.0.3-cp311-cp311-win_amd64.whl',
        'fsspec-2026.9.0-py3-none-any.whl',
        'filelock-4.0.1-py3-none-any.whl',
        'typing_extensions-4.16.0-py3-none-any.whl']


def wheel_urls_pytorch(filename, tag):
    quoted = quote_plus_name(filename)
    return [mirror.format(tag=tag)+quoted for mirror in PYTORCH_MIRRORS]


def wheel_urls_pypi(filename, package):
    urls = []
    for index in PYPI_INDEX:
        try:
            with urlopen(Request(index.format(package=package), headers={'User-Agent': 'DDH-Crop-Build'}), timeout=60) as response:
                page = response.read().decode('utf-8', 'replace')
            match = re.search(rf'href="([^"]+/{re.escape(filename)})#sha256=([0-9a-f]{{64}})"', page, re.I)
            if match:
                urls.append(urljoin(index.format(package=package), match.group(1)))
                break
        except Exception as error:
            print(f'  索引不可用 {index.format(package=package)}：{error}')
    return urls


def quote_plus_name(filename):
    """wheel 文件名里的 + 在 URL 里必须编码为 %2B。"""
    return filename.replace('+', '%2B')


def pypi_package_of(filename):
    return re.split(r'-\d+\.', filename, 1)[0].lower().replace('_', '-')


def fetch(filename, urls, skip_existing):
    target = CACHE/filename
    if skip_existing and target.is_file():
        digest = hashlib.sha256()
        with target.open('rb') as stream:
            for chunk in iter(lambda: stream.read(CHUNK), b''):
                digest.update(chunk)
        return {'sha256': digest.hexdigest(), 'bytes': target.stat().st_size, 'urls': urls, 'cached': True}
    CACHE.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix('.part')
    errors = []
    for url in urls:
        try:
            print(f'  ↓ {url.split("/")[2]} …', flush=True)
            started = time.time()
            digest = hashlib.sha256()
            size = 0
            with urlopen(Request(url, headers={'User-Agent': 'DDH-Crop-Build'}), timeout=120) as response, part.open('wb') as stream:
                for chunk in iter(lambda: response.read(CHUNK), b''):
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            part.replace(target)
            speed = size/max(time.time()-started, .1)/1e6
            print(f'    {size/1e6:.0f} MB，{speed:.0f} MB/s，sha256 {digest.hexdigest()[:16]}…')
            return {'sha256': digest.hexdigest(), 'bytes': size, 'urls': urls}
        except Exception as error:
            errors.append(f'{url.split("/")[2]}: {error}')
    raise SystemExit(f'下载失败 {filename}：{errors}')


def main():
    skip = '--skip-existing' in sys.argv
    entries = {}
    print('=== PyPI 依赖 wheel ===')
    for filename in DEPS:
        package = pypi_package_of(filename)
        urls = wheel_urls_pypi(filename, package)
        if not urls:
            raise SystemExit(f'在镜像索引里找不到 {filename}')
        entries[filename] = fetch(filename, urls, skip)
    print('=== torch / torchvision（按变体）===')
    wheels = {}
    for variant in ('cpu', 'cuda'):
        tag = PYTORCH_TAG[variant]
        wheels[variant] = {}
        for template in (TORCH, TORCHVISION):
            filename = template.format(tag=tag)
            print(f'[{variant}] {filename}')
            wheels[variant][filename] = fetch(filename, wheel_urls_pytorch(filename, tag), skip)
    lock = {
        'schema': 'ddh_components/v1',
        'python': 'cp311', 'platform': 'win_amd64',
        'torch': '2.5.1', 'torchvision': '0.20.1',
        'wheels': {
            'cpu': {'label': 'CPU 推理组件',
                    'bytes': sum(e['bytes'] for e in wheels['cpu'].values()),
                    'wheels': {name: {**e, 'urls': e['urls']} for name, e in wheels['cpu'].items()}},
            'cuda': {'label': 'CUDA 推理/训练组件（需要 NVIDIA 显卡）',
                     'bytes': sum(e['bytes'] for e in wheels['cuda'].values()),
                     'wheels': {name: {**e, 'urls': e['urls']} for name, e in wheels['cuda'].items()}}},
        'deps': {name: entries[name] for name in DEPS},
    }
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=1), encoding='utf-8')
    total = sum(e['bytes'] for e in entries.values())+sum(e['bytes'] for w in wheels.values() for e in w.values())
    print(f'→ {LOCK}')
    print(f'  变体合计：cpu {sum(e["bytes"] for e in wheels["cpu"].values())/1e6:.0f} MB + deps '
          f'{sum(e["bytes"] for e in entries.values())/1e6:.0f} MB；cuda {sum(e["bytes"] for e in wheels["cuda"].values())/1e6:.0f} MB + deps')


if __name__ == '__main__':
    main()
