"""Exercise the real reviewer browser through a local TLS reverse proxy."""
import datetime
import http.client
import json
import ssl
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from PIL import Image
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'tests'))
from studio_collaboration import Collaboration
from studio_service import StudioService
from test_studio_service import FakeRunner


class Proxy(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def forward(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
        connection = http.client.HTTPConnection('127.0.0.1', self.server.upstream_port, timeout=10)
        try:
            connection.request(self.command, self.path, body=body or None,
                               headers={key: value for key, value in self.headers.items() if key.lower() != 'connection'})
            response = connection.getresponse()
            content = response.read()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in ('connection', 'server', 'date'):
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(content)
        finally:
            connection.close()

    do_GET = forward
    do_POST = forward


def test_certificate(root):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key()).serial_number(
        x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=1)).not_valid_after(
        now+datetime.timedelta(hours=1)).add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost')]), critical=False).sign(key, hashes.SHA256())
    private, public = root/'test_key.pem', root/'test_certificate.pem'
    private.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    public.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return private, public


def run():
    with tempfile.TemporaryDirectory(dir=ROOT/'data/experiments') as temp:
        root = Path(temp)
        source = root/'images'
        source.mkdir()
        Image.new('RGB', (200, 200), 'gray').save(source/'synthetic.png')
        service = StudioService(FakeRunner())
        service.configure(source, root/'work')
        manager = Collaboration(service, ROOT/'tools/webapp')
        proxy = ThreadingHTTPServer(('127.0.0.1', 0), Proxy)
        proxy.daemon_threads = True
        private, public = test_certificate(root)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(public, private)
        proxy.socket = context.wrap_socket(proxy.socket, server_side=True)
        url = f'https://localhost:{proxy.server_port}'
        manager.start({'accounts': [{'username': 'reviewer', 'password': 'https-test-2026'}],
                       'listen': '127.0.0.1', 'port': 0, 'public_url': url})
        proxy.upstream_port = manager.server.server_port
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        errors = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(channel='msedge', headless=True)
                client = browser.new_context(ignore_https_errors=True)
                page = client.new_page()
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(url)
                page.locator('#reviewUsername').fill('reviewer')
                page.locator('#reviewPassword').fill('https-test-2026')
                page.locator('#reviewLoginSubmit').click()
                expect(page.locator('#imageMeta')).to_contain_text('px', timeout=15000)
                cookies = client.cookies()
                assert len(cookies) == 1 and cookies[0]['secure'] and cookies[0]['httpOnly']
                page.locator('#saveReview').click()
                expect(page.locator('#finish')).to_be_visible()
                assert service.status()['reviewed'] == 1
                assert next(iter(service.project.records.values()))['reviewer']['username'] == 'reviewer'
                assert not errors, errors
                browser.close()
                print('PASS: local TLS proxy -> secure cookie -> same-origin login -> image prediction -> review saved on host')
        finally:
            proxy.shutdown()
            proxy.server_close()
            thread.join()
            manager.stop()


if __name__ == '__main__':
    run()
