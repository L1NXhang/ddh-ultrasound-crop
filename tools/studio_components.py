"""推理组件（torch/torchvision）的按需下载、校验与加载。仅依赖标准库。

设计约定（v0.7.0 引导式安装）：
  * 安装包不含 torch，只带 components.lock.json（每个 wheel 的文件名、字节数、
    sha256、镜像 URL 列表，全部在打包时由 tools/build_components_lock.py 固化）。
  * 用户机器上首次启动时，前端提示选择 CPU / CUDA 组件；下载走镜像列表，
    断点续传（HTTP Range），流式计算 sha256，与锁文件不符立即删除换镜像。
  * wheel 即 zip，直接解包到 <app>/components/，运行时把该目录插入 sys.path。
  * 本模块不 import torch——加载动作由调用方在 ensure_components() 之后进行。
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

CHUNK = 1024 * 1024
TIMEOUT = 60
LINE_LIMIT = 200


def ensure_components(root):
    """把 <root>/components 目录插入 sys.path（幂等），并注册 DLL 搜索目录。

    必须在任何 torch/torchvision import 之前调用（ModelRunner.load 与
    run_train_job 的入口处）。
    """
    folder = Path(root)/'components'
    folder.mkdir(parents=True, exist_ok=True)
    text = str(folder)
    if text not in sys.path:
        sys.path.insert(0, text)
    if os.name == 'nt':
        try:
            os.add_dll_directory(text)
        except (AttributeError, OSError):
            pass
    return folder


def torch_available(root=None):
    if root is not None:
        ensure_components(root)
    if 'torch' in sys.modules:
        return True                                # 已加载（或测试桩）
    try:
        return importlib.util.find_spec('torch') is not None
    except ValueError:
        return True                                # 已 import 但无 __spec__（特殊加载器）


def suggest_variant():
    """探测 NVIDIA 显卡：有 → 建议 cuda，否则 cpu。只在 Windows 上探测。"""
    if os.name != 'nt':
        return 'cpu'
    for name in ('nvidia-smi.exe',):
        system = Path(os.environ.get('WINDIR', r'C:\Windows'))/'System32'/name
        if system.is_file():
            return 'cuda'
    try:
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=10, creationflags=flags)
        if result.returncode == 0 and result.stdout.strip():
            return 'cuda'
    except (OSError, subprocess.SubprocessError):
        pass
    return 'cpu'


def load_lock(path):
    lock = json.loads(Path(path).read_text(encoding='utf-8'))
    if lock.get('schema') != 'ddh_components/v1':
        raise ValueError('组件清单格式不认识，请更新软件')
    return lock


def _hash_file(path, cancelled=None, progress=None, total=None):
    digest = hashlib.sha256()
    size = Path(path).stat().st_size
    done = 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b''):
            if cancelled and cancelled():
                raise InterruptedError('已取消')
            digest.update(chunk)
            done += len(chunk)
            if progress:
                progress(done, total if total is not None else size, '校验中')
    return digest.hexdigest()


def download_file(urls, destination, sha256=None, progress=None, cancelled=None,
                  segments=1, local_sources=()):
    """获得 destination 文件。顺序：本机已有源 → 网络（多段并行）→ 单流兜底。

    local_sources：本机已有的同内容文件，先 sha256 校验，通过直接复制（零下载）。
    网络：支持 Range 且文件较大时按 segments 段并行（各段 .partN 可独立续传），
    完成后按序拼接，整体 sha256 由本函数校验，不符即弃并尝试下一个镜像。
    progress(done, total, stage)；cancelled() 返回 True 时中止（保留断点）。
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for source in local_sources:                     # 本机源：校验通过即复制
        source = Path(source)
        if not source.is_file() or source.resolve() == destination.resolve():
            continue
        try:
            if progress:
                progress(0, source.stat().st_size, '发现本机已有组件，导入中')
            if sha256 is None or _hash_file(source, cancelled, progress) == sha256:
                if progress:
                    progress(1, 1, '从本机导入')
                shutil.copyfile(source, destination)
                return destination
        except InterruptedError:
            raise
        except Exception:
            continue                                 # 本机源坏了就走网络
    errors = []
    for url in urls:
        try:
            if segments > 1:
                try:
                    _download_segmented(url, destination, segments, progress, cancelled)
                except _NoRangeSupport:
                    _download_single(url, destination, progress, cancelled)
            else:
                _download_single(url, destination, progress, cancelled)
            if sha256 and _hash_file(destination, cancelled, progress) != sha256:
                raise ValueError(f'校验失败（期望 {sha256[:12]}…）')
            return destination
        except InterruptedError:
            raise
        except Exception as error:                   # 镜像失败：换下一个
            host = url.split('/')[2] if '/' in url else url
            errors.append(f'{host}: {error}')
            if progress:
                progress(0, None, f'镜像不可用，尝试下一个（{len(errors)}）')
    raise ConnectionError('所有下载源都失败了：'+'；'.join(errors))


class _NoRangeSupport(Exception):
    pass


def _probe_total(url):
    """Range: 0-0 探测总长；服务器不支持 Range 时抛 _NoRangeSupport。"""
    request = Request(url, headers={'Range': 'bytes=0-0', 'User-Agent': 'DDH-Crop-Component-Installer'})
    with urlopen(request, timeout=TIMEOUT) as response:
        if response.status != 206:
            raise _NoRangeSupport()
        total = response.headers.get('Content-Range', '').rsplit('/', 1)[-1]
        return int(total) if total.isdigit() else None


def _download_segmented(url, destination, segments, progress, cancelled):
    total = _probe_total(url)
    if not total or total < CHUNK*16:                # 小文件单流即可
        raise _NoRangeSupport()
    bounds = [total*i//segments for i in range(segments+1)]
    parts = [destination.with_suffix(destination.suffix+f'.part{i}') for i in range(segments)]
    state = {'done': 0}
    state_lock = threading.Lock()

    def worker(index):
        start, end = bounds[index], bounds[index+1]-1
        part = parts[index]
        have = part.stat().st_size if part.exists() else 0
        request = Request(url, headers={'Range': f'bytes={start+have}-{end}',
                                        'User-Agent': 'DDH-Crop-Component-Installer'})
        with urlopen(request, timeout=TIMEOUT) as response:
            with part.open('ab' if have else 'wb') as stream:
                while True:
                    if cancelled and cancelled():
                        raise InterruptedError('下载已取消')
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    stream.write(chunk)
                    with state_lock:
                        state['done'] += len(chunk)
                    if progress:
                        progress(state['done'], total, f'下载中（{segments} 线程）')

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(segments)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if cancelled and cancelled():
        raise InterruptedError('下载已取消')
    if any(part.stat().st_size != bounds[i+1]-bounds[i] for i, part in enumerate(parts)):
        raise ConnectionError('分段下载不完整')
    with destination.open('wb') as output:           # 按序拼接，整体哈希由调用方校验
        for part in parts:
            with part.open('rb') as stream:
                shutil.copyfileobj(stream, output, CHUNK)
    for part in parts:
        part.unlink()


def _download_single(url, destination, progress, cancelled):
    """单流下载（断点续传），供小文件与不支持 Range 的服务器。"""
    part = destination.with_suffix(destination.suffix+'.part')
    done = 0
    headers = {'User-Agent': 'DDH-Crop-Component-Installer'}
    if part.exists():
        done = part.stat().st_size
        headers['Range'] = f'bytes={done}-'
    request = Request(url, headers=headers)
    with urlopen(request, timeout=TIMEOUT) as response:
        length = response.headers.get('Content-Length')
        length = int(length) if length else None
        if response.status == 206 and done:
            total = length+done if length is not None else None
        else:
            done = 0
            total = length
        if progress:
            progress(done, total, '下载中')
        with part.open('ab' if done else 'wb') as stream:
            while True:
                if cancelled and cancelled():
                    raise InterruptedError('下载已取消')
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                stream.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total, '下载中')
    part.replace(destination)


def unpack_wheel(whl, destination):
    """wheel 就是 zip，包目录（torch/ 等）直接解到 components 根目录。"""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(whl) as archive:
        archive.extractall(destination)


class ComponentService:
    """同一时间一个安装作业；进度轮询；完成后记录 installed.json。"""

    def __init__(self, root, lock_path):
        self.root = Path(root)
        self.components = ensure_components(self.root)
        self.lock_path = Path(lock_path)
        self.lock = None
        self.lock_error = None
        self.thread = None
        self.cancel = threading.Event()
        self.job = {'state': 'idle', 'lines': [], 'progress': None, 'error': None,
                    'result': None, 'variant': None}
        self._load_lock()

    def _load_lock(self):
        try:
            self.lock = load_lock(self.lock_path)
            self.lock_error = None
        except FileNotFoundError:
            self.lock, self.lock_error = None, None          # 离线版没有锁文件，属正常
        except (ValueError, json.JSONDecodeError) as error:
            self.lock, self.lock_error = None, str(error)

    def _log(self, text):
        self.job['lines'] = (self.job['lines']+[text])[-LINE_LIMIT:]

    def status(self):
        installed = self.installed()
        wheels = self.lock['wheels'] if self.lock else {}
        return {'torch_ready': torch_available(self.root),
                'lock_available': self.lock is not None,
                'lock_error': self.lock_error,
                'suggested': suggest_variant(),
                'installed': installed,
                'variants': {name: {'label': spec['label'], 'bytes': spec['bytes'],
                                    'missing_bytes': self._missing_bytes(installed, spec['wheels'])}
                             for name, spec in wheels.items()},
                'state': self.job['state'], 'running': self.job['state'] == 'running',
                'progress': self.job['progress'], 'lines': self.job['lines'][-8:],
                'error': self.job['error'], 'result': self.job['result'],
                'variant': self.job['variant']}

    def _missing_bytes(self, installed, wheels):
        return sum(w['bytes'] for name, w in wheels.items()
                   if installed.get(name) != w['sha256'])

    def installed(self):
        record = self.components/'installed.json'
        if not record.is_file():
            return {}
        try:
            return json.loads(record.read_text(encoding='utf-8'))
        except ValueError:
            return {}

    def install(self, variant):
        if self.lock is None:
            raise ValueError(self.lock_error or '组件清单缺失，无法在线安装')
        variants = self.lock.get('wheels', {})
        if variant not in variants:
            raise ValueError('未知组件类型：'+variant)
        if self.job['state'] == 'running':
            raise ValueError('已有组件安装在进行')
        wheels = {**self.lock.get('deps', {}), **variants[variant]['wheels']}
        self.cancel.clear()
        self.job = {'state': 'running', 'lines': [], 'progress': None, 'error': None,
                    'result': None, 'variant': variant}
        self.thread = threading.Thread(target=self._install, args=(variant, wheels), daemon=True)
        self.thread.start()
        return {'ok': True}

    def import_folder(self, folder):
        """从本机文件夹导入组件 wheel（逐个校验 sha256，零下载）。"""
        if self.lock is None:
            raise ValueError(self.lock_error or '组件清单缺失，无法导入')
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError('文件夹不存在：'+str(folder))
        if self.job['state'] == 'running':
            raise ValueError('已有组件安装在进行')
        catalog = {**self.lock.get('deps', {}), **{name: spec
                   for variant in self.lock.get('wheels', {}).values()
                   for name, spec in variant['wheels'].items()}}
        found = {name: spec for name, spec in catalog.items() if (folder/name).is_file()}
        if not found:
            raise ValueError('该文件夹里没有清单所列的组件文件（torch/torchvision 等 .whl）')
        self.cancel.clear()
        self.job = {'state': 'running', 'lines': [], 'progress': None, 'error': None,
                    'result': None, 'variant': 'import'}
        self.thread = threading.Thread(target=self._import, args=(folder, found), daemon=True)
        self.thread.start()
        return {'ok': True}

    def _import(self, folder, wheels):
        try:
            for filename, spec in wheels.items():
                if self.installed().get(filename) == spec['sha256']:
                    self._log(f'已安装，跳过 {filename}')
                    continue
                self.job['progress'] = {'file': filename, 'done': 0, 'total': spec['bytes'], 'stage': '导入中'}
                self._log(f'从本机导入 {filename}（{spec["bytes"]/1e6:.0f} MB）')
                source = Path(folder)/filename

                def progress(done, total, stage, name=filename):
                    self.job['progress'] = {'file': name, 'done': done, 'total': total, 'stage': stage}
                if _hash_file(source, self.cancel.is_set, progress, spec['bytes']) != spec['sha256']:
                    raise ValueError(f'{filename} 与清单哈希不符，已拒绝导入')
                self.job['progress'] = {'file': filename, 'done': spec['bytes'],
                                        'total': spec['bytes'], 'stage': '解包中'}
                unpack_wheel(source, self.components)
                self._record(filename, spec['sha256'])
            self.job['state'] = 'completed'
            self.job['progress'] = None
            self.job['result'] = {'variant': 'import'}
            self._log('本机组件导入完成，推理功能已就绪')
        except InterruptedError:
            self.job['state'] = 'cancelled'
            self.job['progress'] = None
            self._log('导入已取消')
        except Exception as error:
            self.job['state'] = 'failed'
            self.job['progress'] = None
            self.job['error'] = str(error)
            self._log('导入失败：'+str(error))

    def _install(self, variant, wheels):
        cache = self.root/'data'/'components_cache'
        try:
            for filename, spec in wheels.items():
                if self.installed().get(filename) == spec['sha256']:
                    self._log(f'已安装，跳过 {filename}')
                    continue
                target = cache/filename
                urls = self._urls(spec, filename)
                self._log(f'下载 {filename}（{spec["bytes"]/1e6:.0f} MB）')
                def progress(done, total, stage, name=filename):
                    self.job['progress'] = {'file': name, 'done': done, 'total': total, 'stage': stage}
                download_file(urls, target, spec['sha256'], progress, self.cancel.is_set,
                              segments=4 if spec['bytes'] > CHUNK*16 else 1)
                self.job['progress'] = {'file': filename, 'done': spec['bytes'],
                                        'total': spec['bytes'], 'stage': '解包中'}
                self._log(f'校验通过，解包 {filename}')
                unpack_wheel(target, self.components)
                self._record(filename, spec['sha256'])
            self.job['state'] = 'completed'
            self.job['progress'] = None
            self.job['result'] = {'variant': variant}
            self._log('组件安装完成，推理功能已就绪')
        except InterruptedError:
            self.job['state'] = 'cancelled'
            self.job['progress'] = None
            self._log('安装已取消（已下载的部分保留，下次续传）')
        except Exception as error:
            self.job['state'] = 'failed'
            self.job['progress'] = None
            self.job['error'] = str(error)
            self._log('安装失败：'+str(error))

    def _urls(self, spec, filename):
        urls = list(spec.get('urls', []))
        if not urls:
            raise ValueError(f'组件清单里 {filename} 没有下载地址')
        return urls

    def _record(self, filename, sha256):
        record = dict(self.installed())
        record[filename] = sha256
        record['updated_at'] = time.time()
        path = self.components/'installed.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)

    def stop(self):
        if self.job['state'] == 'running':
            self.cancel.set()
            return {'ok': True}
        return {'ok': True, 'note': '没有正在进行的安装'}
