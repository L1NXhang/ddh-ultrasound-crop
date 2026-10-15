"""Real-browser workflow test with a deterministic substitute for model inference."""
import json
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import crop_app


def run():
    with tempfile.TemporaryDirectory(dir=ROOT / 'data/experiments') as temp:
        folder = Path(temp)
        Image.new('RGB', (200, 200), '#404855').save(folder / 'sample.png')
        initial = {'id': 'sample', 'file': 'sample.png', 'window': [10, 20, 100, 180],
                   'crop': [40, 150], 'window_flags': [], 'window_confidence': 1.0}
        crop_app.STATE.update(folder=folder, out=folder/'out', done={}, session={}, dry=False,
                              rows=[{'id': 'sample', 'file': 'sample.png'}],
                              predictions={'sample': initial}, latest_predictions={'sample': initial})
        server = ThreadingHTTPServer(('127.0.0.1', 0), crop_app.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(crop_app, 'predict', return_value=initial), patch.object(
                    crop_app, 'predict_crop', return_value={'crop': [50, 160], 'std': [0, 0]}), sync_playwright() as p:
                browser = p.chromium.launch(channel='msedge', headless=True)
                page = browser.new_page(viewport={'width': 1200, 'height': 850})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto((ROOT/'tools/webapp/crop.html').as_uri())
                assert '启动裁剪工具.bat' in page.locator('#info').inner_text()
                assert page.get_by_role('link', name='打开本机裁剪工具').get_attribute('href') == 'http://127.0.0.1:8770/'
                assert '需要启动本机服务' in page.locator('#progress').inner_text()
                assert not errors, errors
                page.goto(f'http://127.0.0.1:{server.server_port}')
                page.wait_for_function('img !== null && !loading')
                assert 'sample.png' in page.locator('#info').inner_text()
                canvas = page.locator('#cv').bounding_box()
                scale = canvas['width']/200
                page.keyboard.press('w')
                page.mouse.move(canvas['x']+100*scale, canvas['y']+100*scale)
                page.mouse.down()
                page.mouse.move(canvas['x']+120*scale, canvas['y']+100*scale, steps=5)
                page.mouse.up()
                assert '窗口已修改' in page.locator('#info').inner_text()
                page.keyboard.press('Enter')
                page.wait_for_function("document.getElementById('toast').textContent.includes('先按 R')")
                assert not (folder/'out/results.jsonl').exists()
                page.get_by_role('button', name='按当前窗口重新预测', exact=True).click()
                page.wait_for_function('!busy && human.crop[0] === 50')
                assert '窗口已修改' not in page.locator('#info').inner_text()
                output = ROOT / 'output/playwright'
                output.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(output/'crop_repredict.png'))
                page.keyboard.press('Enter')
                page.wait_for_function("document.getElementById('done').style.display === 'flex'")
                record = json.loads((folder/'out/results.jsonl').read_text(encoding='utf-8'))
                assert record['window'] == [10, 20, 120, 180], record
                assert record['crop'] == [50, 160], record
                assert record['model_window'] == initial['window']
                assert record['model_crop'] == initial['crop']
                with Image.open(folder/'out/cropped/sample.png') as im:
                    assert im.size == (110, 110), im.size
                assert not errors, errors
                browser.close()
                print('PASS: edit window -> stale save blocked -> re-predict -> export 110x110 -> provenance retained')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    run()
