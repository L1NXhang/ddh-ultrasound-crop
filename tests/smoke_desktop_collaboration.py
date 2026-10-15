"""Native desktop host, real model and a separate reviewer browser."""
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
from smoke_desktop import EXE, alive, images, native


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def wait_for(check, seconds=40):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        if check():
            return
        threading.Event().wait(.1)
    raise TimeoutError('Desktop readiness check timed out')


def run():
    recents = ROOT/'data/studio_runtime/recent_projects.json'
    previous = recents.read_bytes() if recents.exists() else None
    with tempfile.TemporaryDirectory(dir=ROOT/'data/experiments') as temp:
        root = Path(temp)
        source = root/'native_review_source'
        images(source, 4)
        session = root/'desktop_session.json'
        debug_port = free_port()
        process = subprocess.Popen([str(EXE), '--debug-port='+str(debug_port), '--session-file='+str(session)], cwd=EXE.parent)
        errors = []
        try:
            wait_for(lambda: session.exists() or process.poll() is not None)
            if not session.exists():
                raise RuntimeError('本机软件可能已打开；未操作现有窗口，请关闭后再运行桌面验收')
            info = json.loads(session.read_text(encoding='utf-8'))
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

            def cdp_ready():
                try:
                    return opener.open(f'http://127.0.0.1:{debug_port}/json/version', timeout=1).status == 200
                except (OSError, urllib.error.URLError):
                    return False

            wait_for(cdp_ready)
            assert 'DDH' in native(process.pid, 'Inspect')
            with sync_playwright() as p:
                desktop = p.chromium.connect_over_cdp(f'http://127.0.0.1:{debug_port}')
                host = desktop.contexts[0].pages[0]
                host.on('pageerror', lambda error: errors.append(str(error)))
                host.wait_for_selector('#setup:not(.hidden)')
                host.locator('#sourcePath').fill(str(source))
                host.locator('#workPath').fill(str(root/'work'))
                host.locator('#startProject').click()
                expect(host.locator('#imageMeta')).to_contain_text('px', timeout=45000)
                expect(host.locator('#loading')).to_be_hidden()
                host.locator('#collaborationOpen').click()
                host.locator('#collaborationListen').select_option('127.0.0.1')
                host.locator('#collaborationPort').fill('0')
                for index in range(2):
                    host.locator('#collaborationAccounts input[type=password]').nth(index).fill('native-review-'+str(index))
                host.locator('#collaborationStart').click()
                host.wait_for_selector('#collaborationRunning:not(.hidden)')
                url = host.locator('#collaborationLinks code').first.inner_text()
                peer = p.chromium.launch(channel='msedge', headless=True)
                reviewer = peer.new_page()
                reviewer.on('pageerror', lambda error: errors.append(str(error)))
                reviewer.goto(url)
                reviewer.locator('#reviewUsername').fill('member1')
                reviewer.locator('#reviewPassword').fill('native-review-0')
                reviewer.locator('#reviewLoginSubmit').click()
                expect(reviewer.locator('#imageMeta')).to_contain_text('px', timeout=45000)
                expect(reviewer.locator('#loading')).to_be_hidden()
                reviewer.locator('#saveReview').click()
                expect(reviewer.locator('#countReviewed')).to_have_text('1')
                project = next((root/'work/DDH裁剪结果').iterdir())
                record = json.loads((project/'results.jsonl').read_text(encoding='utf-8').splitlines()[0])
                assert record['reviewer']['username'] == 'member1'
                assert (project/record['outputs']['cropped']).exists()
                host.locator('#collaborationClose').click()
                native(process.pid, 'Capture', ROOT/'output/playwright/desktop_collaboration.png')
                native(process.pid, 'Close')
                wait_for(lambda: not alive(process.pid) and not alive(info['pid']), 30)
                assert not session.exists()
                try:
                    opener.open(url+'/api/login-info', timeout=2)
                    raise AssertionError('Reviewer port remained open after desktop exit')
                except (OSError, urllib.error.URLError):
                    pass
                assert not errors, errors
                peer.close()
                report = {'native_host': True, 'model_sha256': record['model']['model_sha256'],
                          'real_prediction': True, 'review_saved_on_host': True,
                          'reviewer_port_closed_on_exit': True, 'synthetic_only': True}
                (ROOT/'data/desktop_build/collaboration_acceptance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                print('PASS: native EXE -> real model -> restricted reviewer browser -> host JSON/PNG -> native close stops reviewer port')
        finally:
            if process.poll() is None:
                native(process.pid, 'Close')
                process.wait(timeout=30)
            if previous is None:
                recents.unlink(missing_ok=True)
            else:
                recents.write_bytes(previous)


if __name__ == '__main__':
    run()
