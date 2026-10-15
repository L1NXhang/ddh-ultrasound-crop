"""Offline browser export -> file audit; synthetic test data never enters the inbox."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import audit_cleanup_returns as audit


def prepare(folder):
    (folder/'images').mkdir()
    (folder/'inbox').mkdir()
    image = Image.new('RGB', (200, 200), '#4d5563')
    image.save(folder/'images/001.png')
    digest = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
    row = {'id': 'synthetic', 'filename': '001.png', 'original_path': 'images/001.png',
           'image_wh': [200, 200], 'pixel_sha256': digest}
    for member in (1, 2):
        assignment = folder/f'assignment/成员{member}_试标40张'
        assignment.mkdir(parents=True)
        (assignment/'任务清单.jsonl').write_text(json.dumps(row)+'\n', encoding='utf-8')
    task = {'task_id': 'cleanup_pilot_20261013_member1', 'member': '成员1', 'rows': [row]}
    source = (ROOT/'product/边缘信息复核.html').read_text(encoding='utf-8')
    (folder/'review.html').write_text(source.replace('__TASK_JSON__', json.dumps(task)), encoding='utf-8')


def export_in_browser(folder):
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1250, 'height': 1000}, accept_downloads=True)
        page.goto((folder/'review.html').as_uri())
        page.locator('#files').set_input_files(str(folder/'images'))
        page.wait_for_function('image !== null && !loading')
        canvas = page.locator('#canvas').bounding_box()
        scale = canvas['width']/200
        page.mouse.move(canvas['x']+10*scale, canvas['y']+10*scale)
        page.mouse.down()
        page.mouse.move(canvas['x']+190*scale, canvas['y']+190*scale, steps=4)
        page.mouse.up()
        page.locator('[name=overlayStatus][value=absent]').check()
        page.locator('#copyWindow').click()
        page.locator('#save').click()
        with page.expect_download() as download:
            page.locator('#download').click()
        download.value.save_as(str(folder/'inbox/member1.jsonl'))
        browser.close()


def run():
    with tempfile.TemporaryDirectory(dir=ROOT/'data/experiments') as temp:
        folder = Path(temp)
        prepare(folder)
        export_in_browser(folder)
        args = SimpleNamespace(assignment='assignment', inbox='inbox', out='report', inputs=[],
                               window_tolerance=8, clean_tolerance=16, overlay_iou=.5, render=True)
        with patch.object(audit, 'ROOT', folder):
            result = audit.audit_returns(args)
        assert not result['rejected'], result['rejected']
        assert result['tasks']['cleanup_pilot_20261013_member1']['validated_proposals'] == 1
        assert result['tasks']['cleanup_pilot_20261013_member2']['validated_proposals'] == 0
        assert result['approved_for_training'] == 0
        record = audit.read(folder/'report/validated_proposals.jsonl')[0]
        assert record['window'] == [10, 10, 190, 190]
        assert record['approved_for_training'] is False
        print('PASS: actual offline export -> source hash verified -> missing member reported -> proposal remains unapproved')


if __name__ == '__main__':
    run()
