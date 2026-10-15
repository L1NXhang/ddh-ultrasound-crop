"""Browser checks for unconfirmed edits, undo, preview, recent tasks and retry."""
import json
import sys
import tempfile
import threading
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'tests'))
from crop_studio import StudioServer
from studio_service import StudioService
from studio_session import RecentProjects
from test_studio_service import FakeRunner


def drag_right(page, start=190, end=160):
    canvas = page.locator('#canvas').bounding_box()
    scale = canvas['width']/200
    page.mouse.move(canvas['x']+start*scale, canvas['y']+100*scale)
    page.mouse.down()
    page.mouse.move(canvas['x']+end*scale, canvas['y']+100*scale, steps=4)
    page.mouse.up()


def ready(page):
    page.wait_for_function('image !== null && !busy')


def check_editor(page, service):
    first = service.project.rows[0]['id']
    drag_right(page)
    page.keyboard.press('Control+z')
    assert page.evaluate('cleanBox[2]') == 190
    page.keyboard.press('Control+Shift+z')
    assert page.evaluate('cleanBox[2]') == 160
    page.locator('#previewCrop').click()
    assert page.evaluate('preview && Math.abs(canvas.width/canvas.height - 150/140)<.01')
    assert page.locator('#previewCrop').get_attribute('aria-pressed') == 'true'
    page.screenshot(path=str(ROOT/'output/playwright/studio_product_preview.png'))
    page.locator('#previewCrop').click()
    page.locator('#resetPrediction').click()
    assert page.evaluate('cleanBox[2]') == 190
    page.locator('#undoEdit').click()
    assert page.evaluate('cleanBox[2]') == 160
    page.locator('.queue-row').nth(1).click()
    ready(page)
    assert service.drafts.rows[first]['clean_box'][2] == 160
    assert not service.project.records
    page.locator('.queue-row').first.click()
    ready(page)
    assert page.evaluate('cleanBox[2]') == 160
    page.reload()
    ready(page)
    assert page.locator('#draftState').inner_text() == '已恢复草稿'
    assert page.evaluate('cleanBox[2]') == 160
    page.locator('#newProject').click()
    page.locator('#recentTasks summary').click()
    assert page.locator('.recent-row').count() == 1
    assert '1 草稿' in page.locator('.recent-row').inner_text()
    page.screenshot(path=str(ROOT/'output/playwright/studio_product_recent.png'))
    page.locator('.recent-row').click()
    ready(page)
    assert page.evaluate('cleanBox[2]') == 160
    page.set_viewport_size({'width': 1024, 'height': 700})
    for selector in ('#previewCrop', '#saveReview', '#undoEdit'):
        box = page.locator(selector).bounding_box()
        assert box and box['x'] >= 0 and box['x']+box['width'] <= 1025, (selector, box)
    page.set_viewport_size({'width': 1440, 'height': 1000})
    page.locator('#saveReview').click()
    ready(page)
    assert service.status()['reviewed'] == 1 and service.status()['drafts'] == 0


def check_failure_and_batch(page, service):
    second = page.evaluate('current.id')
    page.route('**/api/draft', lambda route: route.fulfill(
        status=500, content_type='application/json', body=json.dumps({'error': '模拟磁盘不可写'})))
    drag_right(page)
    page.locator('.queue-row').nth(1).click()
    page.wait_for_function('!busy')
    assert page.evaluate('current.id') == second
    assert page.locator('#draftState').inner_text() == '草稿保存失败'
    page.unroute('**/api/draft')
    page.locator('.queue-row').nth(1).click()
    ready(page)
    assert service.drafts.rows[second]['clean_box'][2] == 160
    page.locator('#newProject').click()
    page.locator('[name=runMode][value=auto]').check()
    page.locator('#startProject').click()
    page.wait_for_selector('#finish:not(.hidden)')
    assert service.job['total'] == 1
    assert second not in service.project.records
    assert service.status()['saved'] == 2 and service.status()['reviewed'] == 1
    assert service.drafts.rows[second]['clean_box'][2] == 160
    page.locator('#finishLater').click()
    page.locator('.queue-row').first.click()
    ready(page)
    assert page.evaluate('cleanBox[2]') == 160
    drag_right(page, 160, 150)
    page.evaluate('prepareDesktopClose()')
    assert page.evaluate('ddhCloseReady') is True
    assert service.drafts.rows[second]['clean_box'][2] == 150
    candidates = (service.project.output/'reviewed_training_candidates.jsonl').read_text(encoding='utf-8')
    assert len(candidates.splitlines()) == 1


def run():
    with tempfile.TemporaryDirectory(dir=ROOT/'data/experiments') as temp:
        root = Path(temp)
        source = root/'sample_images'
        source.mkdir()
        for i in range(3):
            image = Image.new('RGB', (200, 200))
            image.putdata([(x, y, 100) for y in range(200) for x in range(200)])
            image.save(source/f'example_{i}.png')
        service = StudioService(FakeRunner(), RecentProjects(root/'recent.json'))
        server = StudioServer(('127.0.0.1', 0), service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(channel='msedge', headless=True)
                page = browser.new_page(viewport={'width': 1440, 'height': 1000})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}')
                page.locator('#sourcePath').fill(str(source))
                page.locator('#workPath').fill(str(root/'work'))
                page.locator('#startProject').click()
                ready(page)
                check_editor(page, service)
                check_failure_and_batch(page, service)
                assert not errors, errors
                browser.close()
                print('PASS: undo/redo -> preview -> draft navigation/reload -> recents -> retry -> batch preserves drafts -> close flush')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    run()
