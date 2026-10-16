"""Local crop studio: choose directories, batch or review, and opt-in ZIP export."""
import argparse
import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from studio_paths import application_root
ROOT = application_root(__file__, sys.executable, getattr(sys, 'frozen', False))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from studio_predictor import ModelRunner
from studio_service import StudioService
from crop_policy import DEFAULT_MARGIN

ASSETS = {'/': ('studio.html', 'text/html; charset=utf-8'),
          '/desktop.js': ('desktop.js', 'text/javascript; charset=utf-8'),
          '/studio.js': ('studio.js', 'text/javascript; charset=utf-8'),
          '/studio_session.js': ('studio_session.js', 'text/javascript; charset=utf-8'),
          '/review_client.js': ('review_client.js', 'text/javascript; charset=utf-8'),
          '/collaboration.js': ('collaboration.js', 'text/javascript; charset=utf-8'),
          '/studio.css': ('studio.css', 'text/css; charset=utf-8')}


def pick_directory(initial):
    if os.name != 'nt':
        raise ValueError('当前系统请直接粘贴文件夹的绝对路径')
    script = '''
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Windows.Forms
$picker = New-Object System.Windows.Forms.FolderBrowserDialog
$picker.Description = '选择文件夹'
$picker.SelectedPath = $env:DDH_PICKER_INITIAL
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
if ($picker.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::Write($picker.SelectedPath)
}
$picker.Dispose()
$owner.Dispose()
'''
    result = subprocess.run(['powershell.exe', '-NoProfile', '-STA', '-Command', script],
                            env={**os.environ, 'DDH_PICKER_INITIAL': str(initial)},
                            capture_output=True, encoding='utf-8', timeout=300,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        logging.error('Folder picker failed: %s', result.stderr)
        raise ValueError('目录选择器未能打开，请直接粘贴文件夹路径')
    return result.stdout.strip()


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, service):
        super().__init__(address, StudioHandler)
        self.service = service
        from studio_collaboration import Collaboration
        self.service.collaboration = Collaboration(service, ROOT/'tools/webapp')
        self.token = secrets.token_urlsafe(32)
        self.desktop = False
        self.stopped = threading.Event()


class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logging.info(fmt, *args)

    def send(self, data, content_type, code=200, filename=None):
        self.response_headers(content_type, len(data), code, filename)
        self.wfile.write(data)

    def send_file(self, file, content_type, filename=None):
        with file.open('rb') as stream:
            self.response_headers(content_type, file.stat().st_size, filename=filename)
            shutil.copyfileobj(stream, self.wfile, length=1024*1024)

    def response_headers(self, content_type, length, code=200, filename=None):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(length))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        if filename:
            from urllib.parse import quote
            self.send_header('Content-Disposition', "attachment; filename*=UTF-8''"+quote(filename))
        self.end_headers()

    def json(self, value, code=200):
        self.send(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8', code)

    def local_host(self):
        host = self.headers.get('Host', '')
        return host in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

    def do_GET(self):
        if not self.local_host():
            self.json({'error': '仅允许本机访问'}, 403)
            return
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            self.get_route(parsed.path, query)
        except (ValueError, KeyError) as error:
            self.json({'error': str(error)}, 400)
        except Exception as error:
            logging.exception('GET failed')
            self.json({'error': '读取失败：'+str(error)}, 500)

    def get_route(self, path, query):
        service = self.server.service
        if path in ASSETS:
            file, content_type = ASSETS[path]
            self.send((ROOT/'tools/webapp'/file).read_bytes(), content_type)
        elif path == '/api/bootstrap':
            self.json({'token': self.server.token, 'default_work': str(ROOT.parent),
                       'status': service.status(), 'recent_projects': service.recent_projects()})
        elif path == '/api/recent':
            self.json({'rows': service.recent_projects()})
        elif path == '/api/collaboration':
            self.json(service.collaboration.status())
        elif path == '/api/status':
            self.json(service.status())
        elif path == '/api/queue':
            self.json({'rows': service.queue(query.get('scope', ['pending'])[0])})
        elif path == '/api/predict':
            self.json(service.predict(query['id'][0]))
        elif path == '/api/image':
            if service.project is None:
                raise ValueError('尚未打开项目')
            file = service.project.assert_source(query['id'][0])
            import mimetypes
            self.send_file(file, mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
        elif path == '/api/archive/download':
            file = service.archives.get(query.get('id', [''])[0])
            if file is None:
                raise ValueError('该压缩包已过期，请重新打包')
            self.send_file(file, 'application/zip', filename=file.name)
        else:
            self.json({'error': '未找到页面'}, 404)

    def read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        if not 0 < length <= 65536:
            raise ValueError('请求长度无效')
        self.connection.settimeout(10)
        payload = self.rfile.read(length)
        if len(payload) != length:
            raise ValueError('请求内容不完整')
        origin = self.headers.get('Origin')
        expected = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
        if not self.local_host() or origin and origin not in expected:
            raise PermissionError('拒绝来自其他网站的请求')
        if not secrets.compare_digest(self.headers.get('X-DDH-Token', ''), self.server.token):
            raise PermissionError('页面会话已过期，请刷新')
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            raise ValueError('请求必须使用 JSON')
        body = json.loads(payload.decode('utf-8'))
        if not isinstance(body, dict):
            raise ValueError('请求必须为 JSON 对象')
        return body

    def do_POST(self):
        try:
            self.post_route(self.path, self.read_body())
        except PermissionError as error:
            self.json({'error': str(error)}, 403)
        except (ValueError, TypeError, KeyError) as error:
            self.json({'error': str(error)}, 400)
        except Exception as error:
            logging.exception('POST failed')
            self.json({'error': '操作失败：'+str(error)}, 500)

    def post_route(self, path, body):
        service = self.server.service
        if path == '/api/project':
            self.json(service.configure(body['source'], body['work']))
        elif path == '/api/folder':
            self.json({'path': pick_directory(body.get('initial') or ROOT.parent)})
        elif path == '/api/repredict':
            self.json(service.repredict(body['id'], body['window']))
        elif path == '/api/review':
            self.json(service.review(body['id'], body['window'], body['clean_box']))
        elif path == '/api/draft':
            self.json(service.save_draft(body['id'], body['window'], body['clean_box']))
        elif path == '/api/skip':
            service.idle()
            if body['id'] not in service.project.by_id:
                raise ValueError('未知图片')
            service.skipped.add(body['id'])
            self.json({'ok': True})
        elif path == '/api/batch/start':
            self.json(service.start_batch())
        elif path == '/api/batch/stop':
            service.cancel_event.set()
            self.json({'ok': True})
        elif path == '/api/archive':
            self.json(service.archive())
        elif path == '/api/collaboration/start':
            self.json(service.collaboration.start(body))
        elif path == '/api/collaboration/stop':
            self.json(service.collaboration.stop())
        elif path == '/api/output/open':
            service.idle()
            if os.name != 'nt':
                raise ValueError('输出目录：'+str(service.project.output))
            os.startfile(service.project.output)
            self.json({'ok': True})
        elif path == '/api/shutdown':
            if not self.server.desktop:
                raise ValueError('此操作仅供桌面软件退出时使用')
            from studio_lifecycle import request_shutdown
            self.json({'ok': True})
            request_shutdown(self.server)
        else:
            self.json({'error': '未知操作'}, 404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder')
    parser.add_argument('--work-dir', default=str(ROOT.parent))
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--checkpoint', default=str(ROOT/'runs_v2/crop.pt'))
    parser.add_argument('--margin', type=float, default=DEFAULT_MARGIN, help='可选额外外扩比例，默认0')
    parser.add_argument('--open-browser', action='store_true')
    parser.add_argument('--desktop', action='store_true')
    parser.add_argument('--session-file')
    parser.add_argument('--parent-pid', type=int)
    parser.add_argument('--state-dir', type=Path, default=ROOT/'data/studio_runtime')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    from studio_session import RecentProjects
    service = StudioService(ModelRunner(args.checkpoint, args.device, margin=args.margin),
                            RecentProjects(args.state_dir/'recent_projects.json'))
    if args.folder:
        service.configure(args.folder, args.work_dir)
    server = StudioServer(('127.0.0.1', args.port), service)
    server.desktop = args.desktop
    url = f'http://127.0.0.1:{server.server_port}/'
    from studio_lifecycle import write_session, watch_parent
    if args.parent_pid:
        watch_parent(server, args.parent_pid)
    if args.session_file:
        write_session(server, args.session_file)
    print('DDH 裁剪工作台已就绪：'+url, flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        service.collaboration.stop()
        server.stopped.set()
        server.server_close()
        if args.session_file:
            Path(args.session_file).unlink(missing_ok=True)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
