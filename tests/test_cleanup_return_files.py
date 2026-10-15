import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import audit_cleanup_returns as audit
from test_cleanup_returns import sample


class ReturnFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        self.root = Path(self.temp.name)
        image = Image.new('RGB', (200, 200), '#4d5563')
        image.save(self.root/'sample.png')
        digest = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        self.rows = []
        (self.root/'inbox').mkdir()
        for member in (1, 2):
            row, reference = sample()
            reference = dict(reference, original_path='sample.png', pixel_sha256=digest)
            row = dict(row, **reference, task_id=f'cleanup_pilot_20261013_member{member}', reviewer=f'成员{member}')
            self.rows.append(row)
            folder = self.root/f'assignment/成员{member}_试标40张'
            folder.mkdir(parents=True)
            (folder/'任务清单.jsonl').write_text(json.dumps(reference)+'\n', encoding='utf-8')
        self.args = SimpleNamespace(assignment='assignment', inbox='inbox', out='out', inputs=[],
                                    window_tolerance=8, clean_tolerance=16, overlay_iou=.5, render=True)
        self.root_patch = patch.object(audit, 'ROOT', self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def write(self, name, rows):
        path = self.root/'inbox'/name
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        return path

    def test_complete_roundtrip_preserves_exports_and_creates_comparison(self):
        self.write('one.jsonl', [self.rows[0]])
        self.write('two.jsonl', [dict(self.rows[1], window=[20, 10, 190, 190])])
        before = {p: p.read_bytes() for p in (self.root/'inbox').glob('*')}
        result = audit.audit_returns(self.args)
        self.assertEqual(result['shared']['both_returned_and_valid'], 1)
        self.assertTrue(result['shared']['pairs'][0]['disagreement'])
        self.assertEqual(result['approved_for_training'], 0)
        self.assertTrue((self.root/'out/对照_01.png').exists())
        records = audit.read(self.root/'out/validated_proposals.jsonl')
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r['approved_for_training'] is False for r in records))
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_invalid_records_are_reported_without_promoting_partial_data(self):
        path = self.write('invalid.jsonl', [dict(self.rows[0], task_id='wrong'), []])
        with path.open('a', encoding='utf-8') as stream:
            stream.write('{"id": NaN}\n')
            stream.write('{broken}\n')
        result = audit.audit_returns(self.args)
        self.assertEqual(len(result['rejected']), 4)
        self.assertEqual((self.root/'out/validated_proposals.jsonl').read_text(), '')

    def test_changed_snapshot_is_quarantined(self):
        self.write('old.jsonl', [self.rows[0]])
        self.write('new.jsonl', [dict(self.rows[0], clean_box=[21, 20, 150, 180])])
        result = audit.audit_returns(self.args)
        self.assertEqual(len(result['version_conflicts']), 1)
        self.assertEqual(result['shared']['both_returned_and_valid'], 0)

    def test_source_image_changes_are_detected(self):
        self.write('one.jsonl', [self.rows[0]])
        Image.new('RGB', (200, 200), 'red').save(self.root/'sample.png')
        result = audit.audit_returns(self.args)
        self.assertEqual(result['rejected'][0]['errors'], ['source_image_unavailable_or_changed'])

    def test_empty_inbox_produces_waiting_report(self):
        result = audit.audit_returns(self.args)
        self.assertEqual(result['status'], '等待成员回传')
        self.assertEqual(result['shared']['pending_shared_ids'], ['sample'])


if __name__ == '__main__':
    unittest.main()
