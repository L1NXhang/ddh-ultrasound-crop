import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from studio_service import StudioService
from studio_session import RecentProjects
from test_studio_service import FakeRunner


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        Image.new('RGB', (200, 200), 'gray').save(self.source/'sample.png')
        self.recents = RecentProjects(self.root/'recent.json')
        self.service = StudioService(FakeRunner(), self.recents)
        self.service.configure(self.source, self.root/'work')
        self.rid = self.service.project.rows[0]['id']

    def tearDown(self):
        deadline = time.monotonic()+5
        while self.service.job['state'] == 'running' and time.monotonic() < deadline:
            time.sleep(.01)
        self.temp.cleanup()

    def reopen(self):
        service = StudioService(FakeRunner(), self.recents)
        service.configure(self.source, self.root/'work')
        return service

    def test_draft_survives_restart_without_becoming_review_or_export(self):
        self.service.predict(self.rid)
        self.service.save_draft(self.rid, [10, 10, 190, 190], [25, 30, 160, 170])
        reopened = self.reopen()
        prediction = reopened.predict(self.rid)
        self.assertTrue(prediction['draft_restored'])
        self.assertEqual(prediction['current_clean_box'], [25, 30, 160, 170])
        self.assertEqual(prediction['crop_box'], [10, 30, 190, 170])
        self.assertEqual(reopened.status()['saved'], 0)
        self.assertEqual(reopened.status()['reviewed'], 0)
        self.assertFalse((reopened.project.output/'reviewed_training_candidates.jsonl').exists())
        self.assertEqual(reopened.status()['drafts'], 1)

    def test_stale_window_draft_is_restored_but_still_requires_reprediction(self):
        self.service.predict(self.rid)
        self.service.save_draft(self.rid, [20, 20, 180, 180], [10, 30, 190, 170])
        reopened = self.reopen()
        prediction = reopened.predict(self.rid)
        self.assertEqual(prediction['current_window'], [20, 20, 180, 180])
        self.assertEqual(prediction['prediction_window'], [10, 10, 190, 190])
        with self.assertRaises(ValueError):
            reopened.review(self.rid, prediction['current_window'], [25, 30, 160, 170])

    def test_reprediction_provenance_survives_draft_and_review(self):
        self.service.runner.metadata = {'model_sha256': 'initial'}
        self.service.predict(self.rid)
        self.service.runner.metadata = {'model_sha256': 'replacement'}
        self.service.repredict(self.rid, [5, 5, 195, 195])
        self.service.save_draft(self.rid, [5, 5, 195, 195], [25, 30, 160, 170])
        reopened = self.reopen()
        prediction = reopened.predict(self.rid)
        self.assertEqual(prediction['prediction_window'], [5, 5, 195, 195])
        record = reopened.review(self.rid, prediction['current_window'], prediction['current_clean_box'])
        self.assertEqual(record['model']['model_sha256'], 'initial')
        self.assertEqual(record['latest_prediction']['model']['model_sha256'], 'replacement')
        self.assertEqual(reopened.status()['drafts'], 0)
        self.assertFalse(reopened.predict(self.rid)['draft_restored'])
        candidates = (reopened.project.output/'reviewed_training_candidates.jsonl').read_text(encoding='utf-8')
        self.assertEqual(len(candidates.splitlines()), 1)
        self.assertFalse(json.loads(candidates)['approved_for_training'])

    def test_changed_original_invalidates_draft(self):
        self.service.predict(self.rid)
        self.service.save_draft(self.rid, [10, 10, 190, 190], [25, 30, 160, 170])
        Image.new('RGB', (200, 200), 'red').save(self.source/'sample.png')
        reopened = self.reopen()
        self.assertEqual(reopened.status()['drafts'], 0)
        self.assertFalse(reopened.predict(self.rid)['draft_restored'])

    def test_new_draft_on_confirmed_image_returns_to_pending_queue(self):
        self.service.predict(self.rid)
        self.service.review(self.rid, [10, 10, 190, 190], [25, 30, 160, 170])
        self.service.save_draft(self.rid, [10, 10, 190, 190], [30, 30, 150, 170])
        reopened = self.reopen()
        self.assertEqual(len(reopened.queue('pending')), 1)
        self.assertTrue(reopened.queue('pending')[0]['draft'])
        self.assertEqual(reopened.project.records[self.rid]['clean_box'], [25, 30, 160, 170])
        self.assertEqual(reopened.predict(self.rid)['current_clean_box'], [30, 30, 150, 170])

    def test_batch_leaves_manual_draft_untouched(self):
        self.service.predict(self.rid)
        self.service.save_draft(self.rid, [10, 10, 190, 190], [25, 30, 160, 170])
        job = self.service.start_batch()
        self.assertEqual(job['total'], 0)
        self.assertEqual(self.service.project.records, {})

    def test_invalid_geometry_and_unknown_ids_cannot_write_drafts(self):
        self.service.predict(self.rid)
        for rid, window, clean in [('unknown', [0, 0, 200, 200], None),
                                   (self.rid, [0, 0, 201, 200], None),
                                   (self.rid, [0, 0, 200, 200], [20, 20, 10, 10])]:
            with self.subTest(rid=rid, window=window, clean=clean):
                with self.assertRaises(ValueError):
                    self.service.save_draft(rid, window, clean)
        self.assertEqual(self.service.status()['drafts'], 0)

    def test_old_revision_draft_is_ignored_after_confirming_new_output(self):
        self.service.predict(self.rid)
        self.service.save_draft(self.rid, [10, 10, 190, 190], [25, 30, 160, 170])
        draft_path = self.service.project.output/'drafts'/(self.rid+'.json')
        old = draft_path.read_bytes()
        self.service.review(self.rid, [10, 10, 190, 190], [30, 30, 150, 170])
        draft_path.write_bytes(old)
        reopened = self.reopen()
        self.assertFalse(reopened.predict(self.rid)['draft_restored'])
        self.assertEqual(reopened.predict(self.rid)['current_clean_box'], [30, 30, 150, 170])

    def test_corrupt_draft_is_reported_without_deleting_evidence(self):
        self.service.predict(self.rid)
        self.service.save_draft(self.rid, [10, 10, 190, 190], [25, 30, 160, 170])
        path = self.service.project.output/'drafts'/(self.rid+'.json')
        for value in ('broken json', '[]', '{"source_sha256":"'+self.service.project.rows[0]['source_sha256']+'"}'):
            path.write_text(value, encoding='utf-8')
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, '草稿文件损坏'):
                self.reopen()
            self.assertEqual(path.read_text(encoding='utf-8'), value)

    def test_corrupt_recent_entries_are_reported(self):
        for value in ('broken json', '{}', '[{}]'):
            self.recents.path.write_text(value, encoding='utf-8')
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, '最近任务记录损坏'):
                self.recents.entries()

    def test_recents_are_deduplicated_and_include_confirmed_progress(self):
        self.service.predict(self.rid)
        self.service.review(self.rid, [10, 10, 190, 190], [25, 30, 160, 170])
        self.reopen()
        entries = RecentProjects(self.root/'recent.json').entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['reviewed'], 1)
        self.assertEqual(entries[0]['source'], str(self.source.resolve()))
        for i in range(8):
            self.recents.remember(dict(entries[0], work=str(self.root/f'work{i}')))
        self.assertEqual(len(self.recents.entries()), 6)
        self.assertEqual(self.recents.entries()[0]['work'], str(self.root/'work7'))


if __name__ == '__main__':
    unittest.main()
