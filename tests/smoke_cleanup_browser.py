"""Exercise the offline review page in a real browser using synthetic pixels."""
import json
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def run():
    with tempfile.TemporaryDirectory(dir=ROOT/'data/experiments') as temp:
        folder = Path(temp)
        (folder/'images').mkdir()
        Image.new('RGB', (200, 200), '#4d5563').save(folder/'images/001.png')
        task = {'task_id': 'synthetic_cleanup_'+folder.name, 'member': '测试成员',
                'rows': [{'id': 'sample', 'filename': '001.png', 'original_path': 'synthetic.png',
                          'image_wh': [200, 200], 'pixel_sha256': 'synthetic'}]}
        source = (ROOT/'product/边缘信息复核.html').read_text(encoding='utf-8')
        (folder/'review.html').write_text(source.replace('__TASK_JSON__', json.dumps(task)), encoding='utf-8')
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='msedge', headless=True)
            page = browser.new_page(viewport={'width': 1250, 'height': 1000}, accept_downloads=True)
            errors, external = [], []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.on('request', lambda req: external.append(req.url) if req.url.startswith(('http:', 'https:')) else None)
            page.goto((folder/'review.html').as_uri())
            page.locator('#files').set_input_files(str(folder/'images'))
            page.wait_for_function('image !== null && !loading')
            page.locator('#save').click()
            assert '请选择有、无' in page.locator('#toast').inner_text()
            assert page.evaluate('Object.keys(done).length') == 0

            def draw(mode, box):
                page.locator('[data-mode='+mode+']').click()
                canvas = page.locator('#canvas').bounding_box()
                scale = canvas['width']/200
                page.mouse.move(canvas['x']+box[0]*scale, canvas['y']+box[1]*scale)
                page.mouse.down()
                page.mouse.move(canvas['x']+box[2]*scale, canvas['y']+box[3]*scale, steps=4)
                page.mouse.up()

            draw('window', [10, 10, 190, 190])
            page.locator('#kind').select_option('body_marker')
            draw('overlay', [160, 140, 185, 180])
            draw('clean', [20, 20, 150, 180])
            page.locator('#save').click()
            with page.expect_download() as downloaded:
                page.locator('#download').click()
            target = folder/'result.jsonl'
            downloaded.value.save_as(str(target))
            record = json.loads(target.read_text(encoding='utf-8'))
            assert record['window'] == [10, 10, 190, 190], record
            assert record['clean_box'] == [20, 20, 150, 180], record
            assert record['overlays'] == [{'kind': 'body_marker', 'box': [160, 140, 185, 180]}]
            assert record['approved_for_training'] is False and record['anatomy_verified'] is False
            page.reload()
            page.locator('#files').set_input_files(str(folder/'images'))
            page.wait_for_function('image !== null && !loading')
            assert page.evaluate('draft.clean_box') == [20, 20, 150, 180]
            draw('clean', [10, 10, 190, 190])
            page.locator('#save').click()
            assert '仍包含覆盖物' in page.locator('#toast').inner_text()
            assert page.evaluate('done.sample.clean_box') == [20, 20, 150, 180]
            output = ROOT/'output/playwright'
            output.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(output/'cleanup_review.png'))
            assert not errors, errors
            assert not external, external
            browser.close()
            print('PASS: offline input -> independent boxes -> original-coordinate export -> cache restore -> overlap validation')


if __name__ == '__main__':
    run()
