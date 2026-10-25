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


def download_file(urls, destination, sha256=None, progress=None, cancelled=None):
    """按镜像顺序下载到 destination（.part 续传），流式 sha256 校验。

    progress(done_bytes, total_bytes, stage_text) 在每块写入后回调；
    cancelled() 返回 True 时中止（抛 InterruptedError）。
    任何镜像的字节数或哈希不符都会删除 .part 并尝试下一个镜像。
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix+'.part')
    errors = []
    for index, url in enumerate(urls):
        try:
            done = 0
            headers = {'User-Agent': 'DDH-Crop-Component-Installer'}
            if part.exists():
                done = part.stat().st_size
                headers['Range'] = f'bytes={done}-'
            request = Request(url, headers=headers)
            with urlopen(request, timeout=TIMEOUT) as response:
                length = response.headers.get('Content-Length')
                length = int(length) if length else None
                digest = hashlib.sha256()
                if response.status == 206 and done:
                    # 断点续传：已下载部分先读入哈希（写句柄打开前完成，避免同文件双开）
                    with part.open('rb') as reuse:
                        for chunk in iter(lambda: reuse.read(CHUNK), b''):
                            digest.update(chunk)
                    total = length+done if length is not None else None
                else:
                    done = 0                           # 服务器不支持续传：从头下
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
                        digest.update(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total, '下载中')
            if sha256 and digest.hexdigest() != sha256:
                raise ValueError(f'校验失败（期望 {sha256[:12]}…，实际 {digest.hexdigest()[:12]}…）')
            part.replace(destination)
            return destination
        except InterruptedError:
            raise
        except Exception as error:                        # 镜像失败：换下一个
            errors.append(f'{url.split("/")[2] if "/" in url else url}: {error}')
            if progress:
                progress(0, None, f'镜像不可用，尝试下一个（{len(errors)}）')
    raise ConnectionError('所有下载源都失败了：'+'；'.join(errors))


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
                download_file(urls, target, spec['sha256'], progress, self.cancel.is_set)
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
