"""Two browser accounts review separate tasks and persist results on the host."""
import json
import sys
import tempfile
import threading
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'tests'))
from crop_studio import StudioServer
from studio_service import StudioService
from test_studio_service import FakeRunner
from smoke_studio_session import drag_right


def ready(page):
    expect(page.locator('#imageMeta')).to_contain_text('px', timeout=15000)
    expect(page.locator('#loading')).to_be_hidden()
    expect(page.locator('#saveReview')).to_be_enabled()


def login(page, username, password):
    page.wait_for_selector('#reviewLogin:not(.hidden)')
    page.locator('#reviewUsername').fill(username)
    page.locator('#reviewPassword').fill(password)
    page.locator('#reviewLoginSubmit').click()
    ready(page)


def run():
    with tempfile.TemporaryDirectory(dir=ROOT/'data/experiments') as temp:
        root = Path(temp)
        source = root/'协作图片'
        source.mkdir()
        for i in range(4):
            Image.new('RGB', (200, 200), (80, 90, i*35)).save(source/f'example{i}.png')
        service = StudioService(FakeRunner())
        server = StudioServer(('127.0.0.1', 0), service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        errors = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(channel='msedge', headless=True)
                host = browser.new_page(viewport={'width': 1440, 'height': 1000})
                host.on('pageerror', lambda error: errors.append(str(error)))
                host.goto(f'http://127.0.0.1:{server.server_port}/')
                host.locator('#sourcePath').fill(str(source))
                host.locator('#workPath').fill(str(root/'work'))
                host.locator('#startProject').click()
                ready(host)
                host.locator('#collaborationOpen').click()
                host.wait_for_selector('#collaborationPanel:not(.hidden)')
                host.locator('#collaborationListen').select_option('127.0.0.1')
                host.locator('#collaborationPort').fill('0')
                passwords = ['review-one-2026', 'review-two-2026']
                for index, password in enumerate(passwords):
                    host.locator('#collaborationAccounts input[type=password]').nth(index).fill(password)
                host.locator('#collaborationStart').click()
                host.wait_for_selector('#collaborationRunning:not(.hidden)')
                info = service.collaboration.status()
                assert info['active'] and [member['total'] for member in info['members']] == [2, 2]
                assert host.locator('#collaborationAccounts input[type=password]').first.input_value() == ''
                (ROOT/'output/playwright').mkdir(parents=True, exist_ok=True)
                host.screenshot(path=str(ROOT/'output/playwright/collaboration_host.png'))
                url = next(value for value in info['urls'] if '127.0.0.1' in value)
                contexts = [browser.new_context(viewport={'width': 1440, 'height': 1000}) for _ in range(2)]
                one, two = [context.new_page() for context in contexts]
                for page in (one, two):
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.goto(url)
                one.screenshot(path=str(ROOT/'output/playwright/collaboration_login.png'))
                for index, page in enumerate((one, two)):
                    login(page, 'member'+str(index+1), passwords[index])
                    assert not page.locator('#newProject').is_visible()
                    assert not page.locator('#archiveSide').is_visible()
                ids = [page.evaluate('queue.map(row=>row.id)') for page in (one, two)]
                assert not set(ids[0]) & set(ids[1])
                assert one.request.get(url+'/api/image?id='+ids[1][0]).status == 403
                assert contexts[0].cookies()[0]['httpOnly']
                rid = one.evaluate('current.id')
                drag_right(one)
                one.keyboard.press('Control+z')
                assert one.evaluate('cleanBox[2]') == 190
                one.keyboard.press('Control+Shift+z')
                assert one.evaluate('cleanBox[2]') == 160
                expect(one.locator('#draftState')).to_have_text('草稿已保存')
                assert not service.project.records
                one.reload()
                ready(one)
                assert one.evaluate('cleanBox[2]') == 160
                assert one.locator('#draftState').inner_text() == '已恢复草稿'
                one.locator('#previewCrop').click()
                one.screenshot(path=str(ROOT/'output/playwright/collaboration_preview.png'))
                one.locator('#previewCrop').click()
                one.screenshot(path=str(ROOT/'output/playwright/collaboration_reviewer.png'))
                one.locator('#saveReview').click()
                expect(one.locator('#countReviewed')).to_have_text('1')
                ready(one)
                assert service.project.records[rid]['reviewer']['username'] == 'member1'
                assert service.project.records[rid]['clean_box'][2] == 160
                other = two.evaluate('current.id')
                two.locator('#saveReview').click()
                expect(two.locator('#countReviewed')).to_have_text('1')
                ready(two)
                assert service.project.records[other]['reviewer']['username'] == 'member2'
                records = [json.loads(line) for line in (service.project.output/'reviewed_training_candidates.jsonl').read_text(encoding='utf-8').splitlines()]
                assert len(records) == 2 and all(not row['approved_for_training'] for row in records)
                one.locator('#reviewLogout').click()
                one.wait_for_selector('#reviewLogin:not(.hidden)')
                assert one.request.get(url+'/api/queue').status == 401
                host.locator('#collaborationStop').click()
                host.wait_for_selector('#collaborationSettings:not(.hidden)')
                assert not service.collaboration.active and len(service.project.records) == 2
                assert not errors, errors
                browser.close()
                print('PASS: host settings -> two logins -> separate assignments -> undo/preview -> persisted draft/reload -> host records -> logout/stop')
        finally:
            service.collaboration.stop()
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    run()
