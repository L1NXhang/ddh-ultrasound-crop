"""A restricted reviewer gateway; the desktop administration API stays on loopback."""
import ipaddress
import json
import logging
import secrets
import shutil
import threading
from http.cookies import SimpleCookie, CookieError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from studio_review_auth import ReviewDenied, ReviewLimited, SESSION_SECONDS
from studio_review_tasks import ReviewConflict

ASSETS = {'/': ('studio.html', 'text/html; charset=utf-8'),
          '/studio.css': ('studio.css', 'text/css; charset=utf-8'),
          '/desktop.js': ('desktop.js', 'text/javascript; charset=utf-8'),
          '/studio.js': ('studio.js', 'text/javascript; charset=utf-8'),
          '/studio_session.js': ('studio_session.js', 'text/javascript; charset=utf-8'),
          '/review_client.js': ('review_client.js', 'text/javascript; charset=utf-8'),
          '/collaboration.js': ('collaboration.js', 'text/javascript; charset=utf-8')}


def private_address(value):
    try:
        ip = ipaddress.ip_address(value)
        return ip.is_loopback or ip in ipaddress.ip_network('10.0.0.0/8') or ip in ipaddress.ip_network('172.16.0.0/12') or \
            ip in ipaddress.ip_network('192.168.0.0/16') or ip in ipaddress.ip_network('100.64.0.0/10')
    except ValueError:
        return False


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False
    request_queue_size = 16

    def __init__(self, address, auth, tasks, assets):
        super().__init__(address, ReviewHandler)
        self.auth, self.tasks, self.assets = auth, tasks, assets
        self.origins = {}
        self.closed = threading.Event()
        self.slots = threading.BoundedSemaphore(24)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = 'DDHReview'
    sys_version = ''

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, fmt, *args):
        logging.debug('Reviewer request completed: %s', self.command)

    def headers_for(self, content_type, length, code=200, cookie=None):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(length))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if cookie:
            self.send_header('Set-Cookie', cookie)
        self.end_headers()

    def send(self, data, content_type, code=200, cookie=None):
        self.headers_for(content_type, len(data), code, cookie)
        self.wfile.write(data)

    def json(self, data, code=200, cookie=None):
        self.send(json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8'),
                  'application/json; charset=utf-8', code, cookie)

    def boundary(self):
        if self.server.closed.is_set():
            raise ReviewDenied('负责人已停止协作审核')
        if not private_address(self.client_address[0]):
            raise ReviewDenied('外网请通过加密组网或 HTTPS 代理连接')
        origin = self.server.origins.get(self.headers.get('Host', ''))
        if not origin:
            raise ReviewDenied('访问地址未获允许，请使用负责人提供的地址')
        received = self.headers.get('Origin')
        if received is not None and received != origin:
            raise ReviewDenied('拒绝来自其他网站的请求')
        return origin

    def cookie_value(self):
        try:
            cookie = SimpleCookie(self.headers.get('Cookie', ''))
            return cookie['DDHReview'].value if 'DDHReview' in cookie else ''
        except CookieError:
            return ''

    def cookie_header(self, value, logout=False):
        secure = '; Secure' if self.server.origins.get(self.headers.get('Host', ''), '').startswith('https:') else ''
        return f'DDHReview={value}; Path=/; HttpOnly; SameSite=Strict; Max-Age={0 if logout else SESSION_SECONDS}'+secure

    def authenticated(self):
        try:
            return self.server.auth.session(self.cookie_value())
        except ReviewDenied as error:
            self.json({'error': str(error)}, 401)
            return None

    def do_GET(self):
        try:
            self.boundary()
            self.get_route()
        except ReviewDenied as error:
            self.json({'error': str(error)}, 403)
        except (ValueError, KeyError, TypeError) as error:
            self.json({'error': str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logging.info('Reviewer disconnected during response')
        except Exception:
            logging.exception('Reviewer read failed')
            self.json({'error': '读取失败，请联系负责人查看主机日志'}, 500)

    def get_route(self):
        parsed = urlparse(self.path)
        if parsed.path in ASSETS:
            name, content_type = ASSETS[parsed.path]
            data = (self.server.assets/name).read_bytes()
            if parsed.path == '/':
                data = data.replace(b'<body>', b'<body data-review-only="true">', 1)
            self.send(data, content_type)
            return
        if parsed.path == '/api/login-info':
            self.json({'nonce': self.server.auth.nonce})
            return
        session = self.authenticated()
        if session is None:
            return
        user, tasks = session['username'], self.server.tasks
        query = parse_qs(parsed.query)
        if parsed.path == '/api/bootstrap':
            self.json({'token': session['csrf'], 'status': tasks.status(user), 'recent_projects': [], 'default_work': ''})
        elif parsed.path == '/api/status':
            self.json(tasks.status(user))
        elif parsed.path == '/api/queue':
            self.json({'rows': tasks.queue(user, query.get('scope', ['pending'])[0])})
        elif parsed.path == '/api/predict':
            self.json(tasks.predict(user, query['id'][0]))
        elif parsed.path == '/api/image':
            path = tasks.image_path(user, query['id'][0])
            import mimetypes
            with path.open('rb') as stream:
                self.headers_for(mimetypes.guess_type(path.name)[0] or 'application/octet-stream', path.stat().st_size)
                shutil.copyfileobj(stream, self.wfile, length=1024*1024)
        else:
            raise ReviewDenied('审核账号无权执行此操作')

    def body(self):
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 < length <= 65536:
            raise ValueError('请求长度无效')
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError('请求内容不完整')
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            raise ValueError('请求必须为 JSON')
        value = json.loads(raw.decode('utf-8'))
        if not isinstance(value, dict):
            raise ValueError('请求必须为 JSON 对象')
        return value

    def do_POST(self):
        try:
            body = self.body()
            origin = self.boundary()
            if self.headers.get('Origin') != origin:
                raise ReviewDenied('请求缺少有效的来源地址')
            self.post_route(body)
        except ReviewLimited as error:
            self.json({'error': str(error)}, 429)
        except ReviewConflict as error:
            self.json({'error': str(error)}, 409)
        except ReviewDenied as error:
            self.json({'error': str(error)}, 403)
        except (ValueError, KeyError, TypeError) as error:
            self.json({'error': str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logging.info('Reviewer disconnected during save response')
        except Exception:
            logging.exception('Reviewer save failed')
            self.json({'error': '保存失败，请联系负责人查看主机日志'}, 500)

    def post_route(self, body):
        auth = self.server.auth
        if self.path == '/api/login':
            if not secrets.compare_digest(self.headers.get('X-DDH-Login', ''), auth.nonce):
                raise ReviewDenied('请从登录页面重试')
            try:
                cookie, session = auth.login(body.get('username'), body.get('password'), self.client_address[0])
            except ReviewLimited:
                raise
            except ReviewDenied as error:
                self.json({'error': str(error)}, 401)
                return
            self.json({'token': session['csrf'], 'username': session['username']}, cookie=self.cookie_header(cookie))
            return
        session = self.authenticated()
        if session is None:
            return
        if not secrets.compare_digest(self.headers.get('X-DDH-Token', ''), session['csrf']):
            raise ReviewDenied('页面会话已过期，请重新登录')
        tasks, user = self.server.tasks, session['username']
        if self.path == '/api/logout':
            auth.logout(self.cookie_value())
            self.json({'ok': True}, cookie=self.cookie_header('', logout=True))
        elif self.path == '/api/draft':
            self.json(tasks.draft(user, body))
        elif self.path == '/api/review':
            self.json(tasks.review(user, body))
        elif self.path == '/api/repredict':
            self.json(tasks.repredict(user, body))
        elif self.path == '/api/skip':
            self.json(tasks.skip(user, body['id']))
        else:
            raise ReviewDenied('审核账号无权执行此操作')
