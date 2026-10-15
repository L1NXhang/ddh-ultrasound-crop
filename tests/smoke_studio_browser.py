"""Real-browser acceptance for both processing modes, with synthetic image/model."""
import json
import sys
import tempfile
import threading
from urllib.parse import urljoin
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'tests'))
from crop_studio import StudioServer
from studio_service import StudioService
from test_studio_service import FakeRunner


def prepare(root):
    source, auto = root/'review_source', root/'auto_source'
    source.mkdir()
    auto.mkdir()
    for i in range(2):
        Image.new('RGB', (200, 200), '#83827b').save(source/f'example_{i}.png')
    Image.new('RGB', (200, 200), '#83827b').save(auto/'example_auto.png')
    return source, auto


def review_flow(page, service, root, source):
    page.locator('#chooseWork').click()
    page.wait_for_function('document.getElementById("workPath").value.includes("test_work")')
    page.locator('#chooseSource').click()
    page.wait_for_function('document.getElementById("sourcePath").value.includes("review_source")')
    page.locator('#startProject').click()
    page.wait_for_function('image !== null && !busy')
    page.set_viewport_size({'width': 1024, 'height': 700})
    assert_visible_actions(page)
    page.screenshot(path=str(ROOT/'output/playwright/studio_review_compact.png'))
    page.set_viewport_size({'width': 1440, 'height': 1000})
    assert_visible_actions(page)
    page.screenshot(path=str(ROOT/'output/playwright/studio_review.png'))
    stage = page.locator('#canvas').bounding_box()
    scale = stage['width']/200
    page.mouse.move(stage['x']+190*scale, stage['y']+100*scale)
    page.mouse.down()
    page.mouse.move(stage['x']+160*scale, stage['y']+100*scale, steps=4)
    page.mouse.up()
    page.locator('#saveReview').click()
    page.wait_for_function('status.reviewed === 1 && !busy')
    record = list(service.project.records.values())[0]
    assert record['clean_box'][2] == 160, record
    assert record['human_reviewed'] and not record['approved_for_training']
    assert not list(root.rglob('*.zip'))
    page.locator('.coordinate-details summary').focus()
    page.keyboard.press('Enter')
    assert service.status()['reviewed'] == 1
    page.locator('#saveReview').click()
    page.wait_for_selector('#finish:not(.hidden)')
    assert service.status()['reviewed'] == 2
    page.locator('#finishZip').click()
    page.wait_for_selector('#zipDownload:not(.hidden)')
    assert len(list(root.rglob('*.zip'))) == 1
    response = page.request.get(urljoin(page.url, page.locator('#zipDownload').get_attribute('href')))
    assert response.body()[:2] == b'PK'
    page.locator('#finishLater').click()


def assert_visible_actions(page):
    viewport = page.viewport_size
    for selector in ('#cropMode', '#windowMode', '#repredict', '#skip', '#saveReview'):
        box = page.locator(selector).bounding_box()
        assert box and box['x'] >= 0 and box['y'] >= 0, selector
        assert box['x']+box['width'] <= viewport['width']+1, (selector, box)
        assert box['y']+box['height'] <= viewport['height']+1, (selector, box)


def auto_flow(page, service, root, source):
    page.locator('#newProject').click()
    page.locator('#sourcePath').fill(str(source))
    page.locator('[name=runMode][value=auto]').check()
    release = threading.Event()
    predict = service.runner.predict

    def delayed_predict(path, window=None):
        if not release.wait(10):
            raise TimeoutError('browser reload did not release inference')
        return predict(path, window)

    with patch.object(service.runner, 'predict', side_effect=delayed_predict):
        try:
            page.locator('#startProject').click()
            page.wait_for_selector('#batchPanel:not(.hidden)')
            page.reload()
            page.wait_for_selector('#setup', state='hidden')
            page.wait_for_function('pollTimer !== null', timeout=3000)
        finally:
            release.set()
    page.wait_for_selector('#finish:not(.hidden)')
    assert service.status()['saved'] == 1 and service.status()['reviewed'] == 0
    assert (service.project.output/'reviewed_training_candidates.jsonl').read_text() == ''
    assert len(list(root.rglob('*.zip'))) == 1
    page.locator('#finishLater').click()
    page.locator('.queue-row').first.click()
    page.wait_for_function('image !== null && !busy')
    page.locator('#saveReview').click()
    page.wait_for_selector('#finish:not(.hidden)')
    candidates = (service.project.output/'reviewed_training_candidates.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(candidates) == 1 and not json.loads(candidates[0])['approved_for_training']


def run():
    with tempfile.TemporaryDirectory(dir=ROOT/'data/experiments') as temp:
        root = Path(temp)
        source, auto = prepare(root)
        service = StudioService(FakeRunner())
        server = StudioServer(('127.0.0.1', 0), service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch('crop_studio.pick_directory', side_effect=[str(root/'test_work'), str(source)]), sync_playwright() as p:
                browser = p.chromium.launch(channel='msedge', headless=True)
                page = browser.new_page(viewport={'width': 1440, 'height': 1000})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                (ROOT/'output/playwright').mkdir(parents=True, exist_ok=True)
                page.goto(f'http://127.0.0.1:{server.server_port}')
                page.wait_for_selector('#setup:not(.hidden)')
                page.set_viewport_size({'width': 1024, 'height': 700})
                assert page.get_by_label('图片文件夹', exact=True).is_visible()
                assert page.get_by_label('保存位置', exact=True).is_visible()
                assert page.locator('#startProject').bounding_box()['y'] < 650
                page.screenshot(path=str(ROOT/'output/playwright/studio_compact.png'))
                page.set_viewport_size({'width': 1440, 'height': 1000})
                (ROOT/'output/playwright').mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(ROOT/'output/playwright/studio_start.png'))
                review_flow(page, service, root, source)
                auto_flow(page, service, root, auto)
                assert not errors, errors
                browser.close()
                print('PASS: choose directories -> manual 4-edge crop -> optional ZIP -> automatic batch -> reviewed training queue')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    run()
