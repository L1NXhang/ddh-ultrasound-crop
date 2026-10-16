"""Unzip into a new path and test the native CPU app without Python on its PATH."""
import argparse
import ctypes
from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from build_portable import allowed_path


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


@contextmanager
def artifact_root():
    root = Path(tempfile.mkdtemp(prefix='portable_验收_', dir=ROOT/'data/experiments')).resolve()
    if not root.is_relative_to(ROOT.resolve()):
        raise ValueError('Acceptance artifact root leaves workspace')
    yield str(root)


def native(exe, pid, action):
    command = [shutil.which('pwsh') or 'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
               '-File', str(ROOT/'tests/desktop_window.ps1'), '-TargetPid', str(pid),
               '-ExpectedHash', hashlib.sha256(exe.read_bytes()).hexdigest(),
               '-ExpectedPath', str(exe), '-Action', action]
    completed = subprocess.run(command, capture_output=True, encoding='utf-8', errors='replace',
                               creationflags=subprocess.CREATE_NO_WINDOW, timeout=15)
    if completed.returncode:
        raise RuntimeError(completed.stderr)


def verify_extract(archive, root):
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP entries')
        for name in names:
            relative = name.split('/', 1)[1]
            if not allowed_path(relative):
                raise ValueError('Unexpected bundled path: '+relative)
            target = (root/name).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError('ZIP entry leaves extraction root')
        zipped.extractall(root)
    folder = next(root.iterdir())
    manifest = json.loads((folder/'release_manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        file = folder/row['path']
        if file.stat().st_size != row['bytes'] or hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Release file hash mismatch: '+row['path'])
    return folder


def exercise(page, url, root, device='cpu'):
    source = root/'外部图片目录'
    source.mkdir()
    for i in range(2):
        image = Image.new('RGB', (320, 260))
        image.paste(Image.effect_noise((280, 220), 45).convert('RGB'), (20, 20))
        image.save(source/f'synthetic_{i}.png')
    page.locator('#sourcePath').fill(str(source))
    page.locator('#workPath').fill(str(root/'外部输出目录'))
    page.locator('#startProject').click()
    page.wait_for_function('image !== null && !busy', timeout=60000)
    assert page.locator('#device').inner_text() == ('GPU' if device == 'cuda' else 'CPU')
    inference = get_json(url+'api/status')
    assert inference['device'] == device
    assert inference['model']['margin'] == 0
    assert inference['model']['crop_policy'] == 'final_bounds/v1'
    page.evaluate('''() => { cleanBox=[cleanBox[0]+5,cleanBox[1],cleanBox[2]-5,cleanBox[3]]; changedGeometry();draw();updateInfo(); }''')
    page.locator('#previewCrop').click()
    page.screenshot(path=str(ROOT/'output/playwright/portable_preview.png'))
    page.locator('#previewCrop').click()
    page.locator('#saveReview').click()
    page.wait_for_function('status.reviewed === 1 && image !== null && !busy')
    status = get_json(url+'api/status')
    output = Path(status['output'])
    candidates = (output/'reviewed_training_candidates.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(candidates) == 1 and not json.loads(candidates[0])['approved_for_training']
    assert json.loads(candidates[0])['label_semantics'] == 'final_crop_box'
    page.locator('#archiveSide').click()
    page.locator('#finishZip').click()
    page.wait_for_selector('#zipDownload:not(.hidden)')
    archive_url = page.locator('#zipDownload').get_attribute('href')
    assert page.request.get(url.rstrip('/')+archive_url).body().startswith(b'PK')
    page.locator('#finishLater').click()
    draft = page.evaluate('''() => {
        cleanBox=[cleanBox[0]+6,cleanBox[1],cleanBox[2],cleanBox[3]];
        changedGeometry();draw();updateInfo();clearTimeout(draftTimer);
        return {id:current.id,clean_box:cleanBox};
    }''')
    return output, draft


def run(archive, expected_device='cpu', hide_gpu=False):
    process = None
    with artifact_root() as temp:
        root = Path(temp).resolve()
        assert root.is_relative_to(ROOT.resolve())
        target = root/'独立解压 软件'
        target.mkdir()
        folder = verify_extract(archive, target)
        exe = folder/'DDH裁剪工具.exe'
        session = root/'desktop_session.json'
        environment = {key: value for key, value in os.environ.items() if key not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')}
        environment['PATH'] = os.environ['WINDIR']+'\\System32;'+os.environ['WINDIR']
        environment['HTTP_PROXY'] = environment['HTTPS_PROXY'] = 'http://127.0.0.1:9'
        if hide_gpu:
            environment['CUDA_VISIBLE_DEVICES'] = '-1'
        try:
            process = subprocess.Popen([str(exe), '--debug-port=9339', '--session-file='+str(session)],
                                       cwd=root, env=environment, creationflags=subprocess.CREATE_NO_WINDOW)
            deadline = time.monotonic()+60
            while not session.exists() and time.monotonic()<deadline:
                if process.poll() is not None:
                    raise RuntimeError('Portable desktop exited before backend handshake')
                time.sleep(.2)
            handshake = json.loads(session.read_text(encoding='utf-8'))
            url = handshake['url']
            assert not get_json(url+'api/status')['configured']
            browser_deadline = time.monotonic()+30
            while True:
                try:
                    get_json('http://127.0.0.1:9339/json/version')
                    break
                except (OSError, ValueError):
                    if time.monotonic() >= browser_deadline:
                        raise TimeoutError('Portable embedded browser did not become ready')
                    time.sleep(.2)
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp('http://127.0.0.1:9339')
                page = browser.contexts[0].pages[0]
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.wait_for_selector('#setup:not(.hidden)', timeout=30000)
                page.locator('#chooseSource').click()
                native(exe, process.pid, 'CancelDialog')
                page.wait_for_function('desktopRequests.size === 0')
                output, draft = exercise(page, url, root, expected_device)
                assert not errors, errors
                native(exe, process.pid, 'Close')
                assert process.wait(timeout=30) == 0
                assert not session.exists()
                saved = json.loads((output/'drafts'/(draft['id']+'.json')).read_text(encoding='utf-8'))
                assert saved['clean_box'] == draft['clean_box']
            model = json.loads((folder/'ddh_multidevice_v2/model_manifest.json').read_text(encoding='utf-8'))
            report = {'relocated_zip': True, 'path_contains_spaces_and_chinese': True,
                      'python_removed_from_path': True, 'external_network_proxy_disabled': True,
                      'inference_device': expected_device, 'gpu_hidden_for_fallback_test': hide_gpu,
                      'external_images': True, 'manual_review': True,
                      'optional_zip': True, 'draft_flushed_on_close': True, 'clean_exit': True,
                      'bundled_fixed_webview': True, 'equivalent_model_parameters': model['equivalent_parameters'],
                      'project_images_in_release': 0}
            build = 'portable_gpu_build' if 'CPU_GPU' in archive.name else 'portable_build'
            (ROOT/'data'/build/('acceptance_'+expected_device+'.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report))
        finally:
            if process is not None and process.poll() is None:
                native(exe, process.pid, 'Close')
                process.wait(timeout=60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--expected-device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--hide-gpu', action='store_true')
    args = parser.parse_args()
    run(args.archive.resolve(), args.expected_device, args.hide_gpu)
