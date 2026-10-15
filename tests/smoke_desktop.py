"""Real native host + WebView2 + CUDA smoke test, using synthetic images only."""
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT.parent/'DDH裁剪工具.exe'
OUTPUT = ROOT/'output/playwright'
KERNEL = ctypes.WinDLL('kernel32', use_last_error=True)
KERNEL.OpenProcess.restype = ctypes.c_void_p
KERNEL.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
KERNEL.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
KERNEL.CloseHandle.argtypes = [ctypes.c_void_p]


def alive(pid):
    handle = KERNEL.OpenProcess(0x100000, False, pid)
    if not handle:
        return False
    try:
        return KERNEL.WaitForSingleObject(handle, 0) == 258
    finally:
        KERNEL.CloseHandle(handle)


def native(pid, action, output=None):
    digest = hashlib.sha256(EXE.read_bytes()).hexdigest()
    args = [shutil.which('pwsh') or 'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ROOT/'tests/desktop_window.ps1'),
            '-TargetPid', str(pid), '-ExpectedHash', digest, '-Action', action]
    if output:
        args += ['-OutputPath', str(output)]
    result = subprocess.run(args, capture_output=True, encoding='utf-8', errors='replace', timeout=15,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, result.stderr
    return result.stdout


def status(url):
    with urllib.request.urlopen(url+'api/status', timeout=5) as response:
        return json.load(response)


def records(output, name='results.jsonl'):
    return [json.loads(line) for line in (Path(output)/name).read_text(encoding='utf-8').splitlines()]


def images(folder, count):
    folder.mkdir(parents=True)
    for i in range(count):
        image = Image.new('RGB', (320, 260))
        image.paste(Image.effect_noise((280, 220), 45).convert('RGB'), (20, 20))
        image.save(folder/f'synthetic_{i}.png')


def review(page, root, url):
    source = root/'review_source'
    images(source, 1)
    page.locator('#workPath').fill(str(root/'work'))
    page.locator('#sourcePath').fill(str(source))
    page.locator('#startProject').click()
    page.wait_for_function('image !== null && !busy', timeout=45000)
    assert page.locator('#device').inner_text() == 'GPU'
    assert status(url)['device'] == 'cuda'
    page.screenshot(path=str(OUTPUT/'desktop_review.png'))
    box = page.locator('#canvas').bounding_box()
    clean, zoom = page.evaluate('[cleanBox,scale]')
    x, y = box['x']+clean[2]*zoom, box['y']+(clean[1]+clean[3])/2*zoom
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x-12*zoom, y, steps=5)
    page.mouse.up()
    page.locator('#saveReview').click()
    page.wait_for_selector('#finish:not(.hidden)')
    result = status(url)
    row = records(result['output'])[0]
    assert row['human_reviewed'] and row['crop_modified'] and not row['approved_for_training']
    assert row['clean_box'][2] == clean[2]-12, row
    assert len(records(result['output'], 'reviewed_training_candidates.jsonl')) == 1
    assert not list(root.rglob('*.zip'))
    page.locator('#finishZip').click()
    page.wait_for_selector('#zipDownload:not(.hidden)')
    assert len(list(root.rglob('*.zip'))) == 1
    page.locator('#finishLater').click()


def automatic(page, root, url):
    source = root/'auto_source'
    images(source, 2)
    page.locator('#newProject').click()
    page.locator('#sourcePath').fill(str(source))
    page.locator('[name=runMode][value=auto]').check()
    page.locator('#startProject').click()
    page.wait_for_selector('#finish:not(.hidden)', timeout=45000)
    result = status(url)
    assert result['saved'] == 2 and result['reviewed'] == 0, result
    assert not records(result['output'], 'reviewed_training_candidates.jsonl')
    assert len(list(root.rglob('*.zip'))) == 1
    page.locator('#finishZip').click()
    page.wait_for_selector('#zipDownload:not(.hidden)')
    assert len(list(root.rglob('*.zip'))) == 2


def run_acceptance(args):
    session = json.loads(Path(args.session).read_text(encoding='utf-8'))
    url = session['url']
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert 'DDH' in native(args.pid, 'Inspect')
    duplicate = subprocess.Popen([str(EXE)], creationflags=subprocess.CREATE_NO_WINDOW)
    assert duplicate.wait(timeout=8) == 0 and alive(args.pid)
    with tempfile.TemporaryDirectory(dir=ROOT/'data/experiments') as temp, sync_playwright() as playwright:
        root = Path(temp)
        browser = playwright.chromium.connect_over_cdp(f'http://127.0.0.1:{args.debug_port}')
        page = browser.contexts[0].pages[0]
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.wait_for_selector('#setup:not(.hidden)')
        assert page.evaluate('Boolean(window.chrome.webview)')
        assert '桌面版' in page.locator('.local').inner_text()
        page.screenshot(path=str(OUTPUT/'desktop_start.png'))
        native(args.pid, 'Capture', OUTPUT/'desktop_window.png')
        page.locator('#chooseWork').click()
        native(args.pid, 'AcceptDirectory')
        page.wait_for_function("!document.getElementById('chooseWork').disabled")
        assert Path(page.locator('#workPath').input_value()) == ROOT.parent
        page.locator('#chooseSource').click()
        native(args.pid, 'CancelDialog')
        page.wait_for_function("!document.getElementById('chooseSource').disabled")
        assert not page.locator('#sourcePath').input_value()
        review(page, root, url)
        automatic(page, root, url)
        page.locator('#zipDownload').click()
        native(args.pid, 'CancelDialog')
        page.wait_for_function('desktopRequests.size === 0', timeout=10000)
        assert not errors, errors
        page.locator('#finishLater').click()
        page.locator('.queue-row').first.click()
        page.wait_for_function('image !== null && !busy', timeout=45000)
        draft_state = page.evaluate('''() => {
            cleanBox=[cleanBox[0]+5,cleanBox[1],cleanBox[2],cleanBox[3]];
            changedGeometry();draw();updateInfo();clearTimeout(draftTimer);
            return {id:current.id,clean_box:cleanBox};
        }''')
        output = Path(status(url)['output'])
        assert not (output/'drafts'/(draft_state['id']+'.json')).exists()
        native(args.pid, 'Close')
        deadline = time.monotonic()+30
        while (alive(args.pid) or alive(session['pid'])) and time.monotonic() < deadline:
            time.sleep(.2)
        assert not alive(args.pid) and not alive(session['pid']), 'Desktop/backend did not release'
        assert not Path(args.session).exists()
        draft = json.loads((output/'drafts'/(draft_state['id']+'.json')).read_text(encoding='utf-8'))
        assert draft['clean_box'] == draft_state['clean_box']
        assert not records(output, 'reviewed_training_candidates.jsonl')
    report = {'native_window': True, 'native_folder_picker': True, 'native_zip_dialog': True,
              'manual_review': True, 'automatic_batch': True, 'single_instance': True,
              'gpu': 'cuda', 'graceful_exit': True, 'draft_flushed_on_close': True, 'synthetic_only': True}
    (ROOT/'data/desktop_build/acceptance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS: native EXE -> owned folder dialogs -> CUDA -> review/auto -> optional ZIP -> native save dialog -> draft flush -> clean exit')


def run(args):
    recents = ROOT/'data/studio_runtime/recent_projects.json'
    previous = recents.read_bytes() if recents.exists() else None
    try:
        run_acceptance(args)
    finally:
        if previous is None:
            recents.unlink(missing_ok=True)
        else:
            recents.write_bytes(previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--debug-port', type=int, default=9337)
    run(parser.parse_args())
