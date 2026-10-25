"""组件下载/校验/加载的测试：本地 HTTP 假镜像，不需要真实 torch。"""
import hashlib
import http.server
import json
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from studio_components import ComponentService, download_file, suggest_variant


def make_wheel(path, package='fakepkg'):
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(f'{package}/__init__.py', f'VALUE = "{package}"\n')
        archive.writestr(f'{package}-1.0.0.dist-info/METADATA', 'Name: ' + package)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Mirror(http.server.BaseHTTPRequestHandler):
    files = {}
    fail_first = []                                   # 这些路径返回 404（模拟坏镜像）
    protocol_version = 'HTTP/1.1'

    def do_GET(self):
        if self.path in Mirror.fail_first or self.path.lstrip('/') not in Mirror.files:
            self.send_error(404)
            return
        body = Mirror.files[self.path.lstrip('/')]
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class ComponentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        self.root = Path(self.temp.name)
        # 直接构造 wheel 字节
        buffer = self.root/'fake-1.0.0-py3-none-any.whl'
        self.good_hash = make_wheel(buffer)
        self.body = buffer.read_bytes()
        self.bad_hash = '0'*64
        Mirror.files = {'fake-1.0.0-py3-none-any.whl': self.body}
        Mirror.fail_first = []
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Mirror)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}/'
        self.lock = self.root/'components.lock.json'
        self.write_lock(self.good_hash)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def write_lock(self, sha256):
        self.lock.write_text(json.dumps({
            'schema': 'ddh_components/v1', 'python': 'cp311', 'platform': 'test',
            'wheels': {'cpu': {'label': 'CPU', 'bytes': len(self.body),
                               'wheels': {'fake-1.0.0-py3-none-any.whl': {
                                   'sha256': sha256, 'bytes': len(self.body),
                                   'urls': [self.base+'mirror-a/fake-1.0.0-py3-none-any.whl',
                                            self.base+'fake-1.0.0-py3-none-any.whl']}}}},
            'deps': {}}), encoding='utf-8')

    def test_download_verify_unpack_and_become_importable(self):
        service = ComponentService(self.root, self.lock)
        self.assertFalse((self.root/'components'/'fakepkg').exists(), '安装前 components 目录应为空')
        service.install('cpu')
        for _ in range(100):
            if service.status()['state'] != 'running':
                break
            threading.Event().wait(.1)
        status = service.status()
        self.assertEqual(status['state'], 'completed', status['error'])
        self.assertTrue((self.root/'components'/'fakepkg'/'__init__.py').is_file())
        self.assertEqual(status['installed'].get('fake-1.0.0-py3-none-any.whl'), self.good_hash)
        # 装好后 find_spec 能看到包（与 torch 的检测逻辑一致）
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('fakepkg'))

    def test_hash_mismatch_falls_back_to_next_mirror(self):
        # 镜像 A 返回坏内容（404 由 fail_first 模拟不可达；这里用错误哈希触发回退）
        self.write_lock(self.bad_hash)
        Mirror.fail_first = ['mirror-a/fake-1.0.0-py3-none-any.whl']
        service = ComponentService(self.root, self.lock)
        service.install('cpu')
        for _ in range(100):
            if service.status()['state'] != 'running':
                break
            threading.Event().wait(.1)
        self.assertEqual(service.status()['state'], 'failed')
        self.assertIn('校验失败', service.status()['error'])
        self.assertFalse((self.root/'components'/'fakepkg').exists(), '校验失败不得解包')

    def test_unreachable_mirrors_fail_with_clear_error(self):
        self.lock.write_text(json.dumps({
            'schema': 'ddh_components/v1', 'wheels': {'cpu': {'label': 'X', 'bytes': 1, 'wheels': {
                'x.whl': {'sha256': '0'*64, 'bytes': 1, 'urls': ['http://127.0.0.1:1/x.whl']}}}},
            'deps': {}}), encoding='utf-8')
        service = ComponentService(self.root, self.lock)
        service.install('cpu')
        for _ in range(100):
            if service.status()['state'] != 'running':
                break
            threading.Event().wait(.1)
        self.assertEqual(service.status()['state'], 'failed')
        self.assertIn('所有下载源都失败', service.status()['error'])

    def test_second_install_rejected_while_running(self):
        service = ComponentService(self.root, self.lock)
        service.install('cpu')
        if service.status()['state'] == 'running':
            with self.assertRaises(ValueError):
                service.install('cpu')
        for _ in range(100):
            if service.status()['state'] != 'running':
                break
            threading.Event().wait(.1)
        # 结束后可以再次启动
        service.install('cpu')
        for _ in range(100):
            if service.status()['state'] != 'running':
                break
            threading.Event().wait(.1)
        self.assertEqual(service.status()['state'], 'completed')

    def test_suggest_variant_is_safe_without_gpu(self):
        self.assertIn(suggest_variant(), {'cpu', 'cuda'})


if __name__ == '__main__':
    unittest.main()
