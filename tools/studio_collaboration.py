"""Host-controlled start/stop of one password-protected reviewer port."""
import socket
import threading
import ctypes
import os
import struct
from urllib.parse import urlparse

from studio_review_auth import ReviewAuth
from studio_review_server import ReviewServer, private_address
from studio_review_tasks import ReviewTasks


def local_addresses():
    addresses = {'127.0.0.1'}
    if os.name == 'nt':
        size = ctypes.c_ulong(0)
        table = ctypes.windll.iphlpapi.GetIpAddrTable
        if table(None, ctypes.byref(size), False) != 122:
            raise ValueError('无法读取本机网络接口，请检查网络状态')
        buffer = ctypes.create_string_buffer(size.value)
        if table(buffer, ctypes.byref(size), False) != 0:
            raise ValueError('无法读取本机 IPv4 地址，请检查网络状态')
        count = struct.unpack_from('=I', buffer.raw)[0]
        for index in range(count):
            address = socket.inet_ntoa(buffer.raw[4+index*24:8+index*24])
            if private_address(address):
                addresses.add(address)
        return sorted(addresses)
    for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        if private_address(info[4][0]):
            addresses.add(info[4][0])
    return sorted(addresses)


def public_origin(value):
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or \
            parsed.path not in ('', '/') or parsed.query or parsed.fragment:
        raise ValueError('外网代理地址请填写 HTTPS 根地址，例如 https://review.example.com')
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError('外网代理端口无效') from error
    if ':' in parsed.hostname or any(char.isspace() for char in parsed.netloc):
        raise ValueError('代理地址请使用 IPv4 或域名')
    host = parsed.hostname+((':'+str(port)) if port and port != 443 else '')
    return host, 'https://'+host


class Collaboration:
    def __init__(self, service, assets):
        self.service, self.assets = service, assets
        self.server = self.tasks = self.thread = None
        self.lock = threading.RLock()
        self.stopping = False

    @property
    def active(self):
        return self.server is not None

    def owns(self, rid):
        return self.active and self.tasks.owns(rid)

    def status(self):
        with self.service.lock:
            if not self.active:
                return {'active': False, 'addresses': local_addresses(), 'urls': [], 'members': []}
            return {'active': True, 'addresses': local_addresses(), 'urls': sorted(set(self.server.origins.values())),
                    'port': self.server.server_port, 'run_id': self.tasks.run_id,
                    'members': [{'username': user, 'total': len(ids), 'reviewed': self.tasks.status(user)['reviewed']}
                                for user, ids in self.tasks.assignments.items()]}

    def start(self, config):
        with self.lock, self.service.lock:
            self.service.idle()
            if self.active or self.stopping:
                raise ValueError('协作审核已经开启，请先停止再修改设置')
            listen, port = config.get('listen', '0.0.0.0'), config.get('port', 8771)
            if listen != '0.0.0.0' and listen not in local_addresses():
                raise ValueError('监听地址必须是本机的局域网、组网或回环 IPv4 地址')
            if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
                raise ValueError('端口请填写 1–65535 之间的数字')
            proxy = public_origin(config.get('public_url', ''))
            auth = ReviewAuth(config.get('accounts'))
            tasks = ReviewTasks(self.service, list(auth.accounts))
            try:
                server = ReviewServer((listen, port), auth, tasks, self.assets)
            except OSError as error:
                raise ValueError('端口无法监听，请换一个端口或检查是否被其他程序占用') from error
            addresses = local_addresses() if listen == '0.0.0.0' else [listen]
            server.origins = {f'{address}:{server.server_port}': f'http://{address}:{server.server_port}' for address in addresses}
            if '127.0.0.1' in addresses:
                server.origins = {**server.origins, f'localhost:{server.server_port}': f'http://localhost:{server.server_port}'}
            if proxy:
                server.origins = {**server.origins, proxy[0]: proxy[1]}
            self.server, self.tasks = server, tasks
            self.thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .1}, daemon=True)
            self.thread.start()
            return self.status()

    def stop(self):
        with self.lock:
            if not self.active:
                return self.status()
            with self.service.lock:
                server, thread = self.server, self.thread
                server.closed.set()
                self.tasks.closed = True
                server.auth.revoke()
                self.stopping = True
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            with self.service.lock:
                self.server = self.tasks = self.thread = None
                self.stopping = False
            return self.status()
