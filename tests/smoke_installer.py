"""Install, run, review and uninstall a data-free release inside an isolated workspace."""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import winreg
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
from smoke_portable import native, exercise
from smoke_desktop import images


def wait_for(check, seconds=60):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.2)
    raise TimeoutError('Installer acceptance timed out')


def installed_registration(label):
    key = 'Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\DDHCrop-'+label+'_is1'
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key):
            return True
    except FileNotFoundError:
        return False


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def connection_ready(url):
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def collaboration_flow(host, playwright, root):
    source = root/'协作图片'
    images(source, 4)
    host.locator('#newProject').click()
    host.locator('#sourcePath').fill(str(source))
    host.locator('#startProject').click()
    expect(host.locator('#projectName')).to_have_text('协作图片', timeout=45000)
    expect(host.locator('#imageMeta')).to_contain_text('px', timeout=45000)
    expect(host.locator('#loading')).to_be_hidden()
    host.locator('#collaborationOpen').click()
    host.wait_for_selector('#collaborationPanel:not(.hidden)')
    host.locator('#collaborationListen').select_option('127.0.0.1')
    host.locator('#collaborationPort').fill('0')
    for index in range(2):
        host.locator('#collaborationAccounts input[type=password]').nth(index).fill('installer-test-'+str(index))
    host.locator('#collaborationStart').click()
    host.wait_for_selector('#collaborationRunning:not(.hidden)')
    url = host.locator('#collaborationLinks code').first.inner_text()
    browser = playwright.chromium.launch(channel='msedge', headless=True)
    try:
        page = browser.new_page()
        page.goto(url)
        page.locator('#reviewUsername').fill('member1')
        page.locator('#reviewPassword').fill('installer-test-0')
        page.locator('#reviewLoginSubmit').click()
        expect(page.locator('#imageMeta')).to_contain_text('px', timeout=45000)
        expect(page.locator('#loading')).to_be_hidden()
        page.locator('#saveReview').click()
        expect(page.locator('#countReviewed')).to_have_text('1')
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(host.url+'api/status') as response:
            output = Path(json.load(response)['output'])
        row = json.loads((output/'results.jsonl').read_text(encoding='utf-8').splitlines()[0])
        assert row['reviewer']['username'] == 'member1' and not row['approved_for_training']
        return url
    finally:
        browser.close()
        host.locator('#collaborationClose').click()


def run(installer, label, expected_device, hide_gpu=False):
    if installed_registration(label):
        raise RuntimeError('Existing user installation detected; refusing to replace it during testing')
    manifest = json.loads((installer.parent/(label+'_installer_manifest.json')).read_text(encoding='utf-8'))
    record = next(row for row in manifest['files'] if row['name'] == installer.name)
    assert hashlib.sha256(installer.read_bytes()).hexdigest() == record['sha256']
    root = Path(tempfile.mkdtemp(prefix='installer_验收_', dir=ROOT/'data/experiments')).resolve()
    target = root/'安装 软件'
    process = None
    uninstaller = target/'unins000.exe'
    try:
        subprocess.run([str(installer), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-',
                        '/NOICONS', '/TASKS=', '/DIR='+str(target), '/LOG='+str(root/'install.log')], check=True, timeout=300)
        assert installed_registration(label) and uninstaller.is_file()
        release = json.loads((target/'release_manifest.json').read_text(encoding='utf-8'))
        for row in release['files']:
            file = target/row['path']
            assert file.stat().st_size == row['bytes'] and hashlib.sha256(file.read_bytes()).hexdigest() == row['sha256']
        environment = {key: value for key, value in os.environ.items() if key not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')}
        environment['PATH'] = os.environ['WINDIR']+'\\System32;'+os.environ['WINDIR']
        environment['HTTP_PROXY'] = environment['HTTPS_PROXY'] = 'http://127.0.0.1:9'
        if hide_gpu:
            environment['CUDA_VISIBLE_DEVICES'] = '-1'
        session = root/'desktop_session.json'
        debug_port = free_port()
        exe = target/'DDH裁剪工具.exe'
        process = subprocess.Popen([str(exe), '--debug-port='+str(debug_port), '--session-file='+str(session)],
                                   cwd=root, env=environment, creationflags=subprocess.CREATE_NO_WINDOW)
        wait_for(lambda: session.exists() or process.poll() is not None)
        assert session.exists(), 'Installed application exited before handshake'
        handshake = json.loads(session.read_text(encoding='utf-8'))
        wait_for(lambda: connection_ready(f'http://127.0.0.1:{debug_port}/json/version'))
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(f'http://127.0.0.1:{debug_port}')
            page = browser.contexts[0].pages[0]
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.wait_for_selector('#setup:not(.hidden)')
            output, draft = exercise(page, handshake['url'], root, expected_device)
            remote_url = collaboration_flow(page, p, root)
            assert not errors, errors
            native(exe, process.pid, 'Close')
            assert process.wait(timeout=45) == 0
        assert not session.exists()
        wait_for(lambda: not connection_ready(remote_url+'/api/login-info'), 10)
        saved = json.loads((output/'drafts'/(draft['id']+'.json')).read_text(encoding='utf-8'))
        assert saved['clean_box'] == draft['clean_box']
        marker = target/'ddh_multidevice_v2/data/studio_runtime/user_keep.txt'
        marker.write_text('User-created data must survive uninstall', encoding='utf-8')
        time.sleep(2)
        subprocess.run([str(uninstaller), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
                        '/LOG='+str(root/'uninstall.log')], check=True, timeout=120)
        wait_for(lambda: not installed_registration(label) and not exe.exists(), 20)
        assert marker.is_file() and (output/'results.jsonl').is_file()
        report = {'installer': installer.name, 'version': manifest['version'],
                  'default_margin': 0., 'label_semantics': 'final_crop_box',
                  'sha256': record['sha256'], 'installed': True,
                  'all_payload_hashes_match': True, 'python_removed_from_path': True,
                  'inference_device': expected_device, 'cuda_hidden': hide_gpu, 'manual_review': True,
                  'collaboration_review': True, 'draft_preserved': True, 'optional_zip': True,
                  'uninstalled': True, 'user_data_preserved': True, 'synthetic_only': True}
        directory = ROOT/'data/publication'/manifest['version']
        directory.mkdir(parents=True, exist_ok=True)
        (directory/('installer_'+label+'_'+expected_device+'.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)
    finally:
        if process is not None and process.poll() is None:
            native(target/'DDH裁剪工具.exe', process.pid, 'Close')
            process.wait(timeout=45)
        if installed_registration(label) and uninstaller.exists():
            subprocess.run([str(uninstaller), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART'], check=True, timeout=120)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--installer', type=Path, required=True)
    parser.add_argument('--label', choices=['CPU', 'CPU_GPU'], required=True)
    parser.add_argument('--expected-device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--hide-gpu', action='store_true')
    args = parser.parse_args()
    run(args.installer.resolve(), args.label, args.expected_device, args.hide_gpu)
