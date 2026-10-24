import sys
import tempfile
import time
import unittest
import threading
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from studio_service import StudioService


class FakeRunner:
    metadata = {'checkpoint': 'synthetic', 'model_sha256': 'test'}
    device = 'test'

    def predict(self, path, window=None):
        if Path(path).name == 'bad.png' and window is None:
            return {'window': None, 'crop_box': None, 'flags': ['no_content']}
        box = window or [10, 10, 190, 190]
        return {'window': list(box), 'crop_box': [box[0], box[1]+20, box[2], box[3]-20], 'flags': []}


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        for name in ('sample.png', 'bad.png'):
            Image.new('RGB', (200, 200), 'gray').save(self.source/name)
        self.service = StudioService(FakeRunner())
        self.service.configure(str(self.source), str(self.root/'work'))

    def tearDown(self):
        self.temp.cleanup()

    def test_batch_saves_predictions_but_requires_manual_review_for_training(self):
        self.service.start_batch()
        for _ in range(100):
            if self.service.status()['job']['state'] != 'running':
                break
            time.sleep(.01)
        status = self.service.status()
        self.assertEqual(status['job']['state'], 'completed')
        self.assertEqual(status['saved'], 1)
        self.assertEqual(status['reviewed'], 0)
        self.assertEqual(status['failed'], 1)
        self.assertEqual(len(self.service.queue('pending')), 2)
        self.assertEqual((self.service.project.output/'reviewed_training_candidates.jsonl').read_text(), '')

    def test_manual_review_stores_independent_final_crop(self):
        row = next(r for r in self.service.project.rows if r['file'] == 'sample.png')
        pred = self.service.predict(row['id'])
        result = self.service.review(row['id'], pred['window'], [25, 30, 160, 170])
        self.assertTrue(result['human_reviewed'])
        self.assertEqual(result['clean_box'], [25, 30, 160, 170])
        self.assertEqual(len(self.service.queue('reviewed')), 1)

    def test_status_reports_average_manual_adjustment(self):
        self.assertIsNone(self.service.status()['avg_adjust_px'])
        row = next(r for r in self.service.project.rows if r['file'] == 'sample.png')
        pred = self.service.predict(row['id'])
        # FakeRunner 预测 clean_box=[10,30,190,170]；人工改为 [25,45,160,155]：上下各调 15px
        self.service.review(row['id'], pred['window'], [25, 45, 160, 155])
        self.assertEqual(self.service.status()['avg_adjust_px'], 15)

    def test_window_edit_requires_reprediction(self):
        row = next(r for r in self.service.project.rows if r['file'] == 'sample.png')
        self.service.predict(row['id'])
        with self.assertRaises(ValueError):
            self.service.review(row['id'], [5, 5, 195, 195], [20, 30, 160, 170])
        self.service.repredict(row['id'], [5, 5, 195, 195])
        result = self.service.review(row['id'], [5, 5, 195, 195], [20, 30, 160, 170])
        self.assertTrue(result['window_modified'])

    def test_revisit_retains_repredicted_window_and_original_provenance(self):
        row = next(r for r in self.service.project.rows if r['file'] == 'sample.png')
        self.service.predict(row['id'])
        self.service.repredict(row['id'], [5, 5, 195, 195])
        revisited = self.service.predict(row['id'])
        self.assertEqual(revisited['current_window'], [5, 5, 195, 195])
        self.assertEqual(revisited['window'], [10, 10, 190, 190])
        result = self.service.review(row['id'], revisited['current_window'], [20, 30, 160, 170])
        self.service.configure(self.source, self.root/'work')
        resumed = self.service.predict(row['id'])
        self.assertTrue(resumed['reviewed'])
        self.assertEqual(resumed['current_clean_box'], [20, 30, 160, 170])
        self.assertEqual(result['latest_prediction']['window'], resumed['prediction_window'])

    def test_stop_finishes_current_image_and_resume_skips_saved(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingRunner(FakeRunner):
            def predict(self, path, window=None):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('test inference did not release')
                return super().predict(path, window or [10, 10, 190, 190])

        self.service.runner = BlockingRunner()
        self.service.start_batch()
        self.assertTrue(entered.wait(5))
        try:
            with self.assertRaises(ValueError):
                self.service.configure(self.source, self.root/'other_work')
            self.service.cancel_event.set()
        finally:
            release.set()
        self.wait_for_batch()
        self.assertEqual(self.service.status()['job']['state'], 'cancelled')
        self.assertEqual(self.service.status()['saved'], 1)
        self.service.start_batch()
        self.wait_for_batch()
        self.assertEqual(self.service.status()['saved'], 2)
        self.assertEqual(self.service.status()['job']['total'], 1)

    def test_feedback_keeps_original_and_reprediction_model_versions(self):
        row = next(r for r in self.service.project.rows if r['file'] == 'sample.png')
        self.service.runner.metadata = {'model_sha256': 'initial'}
        self.service.predict(row['id'])
        self.service.runner.metadata = {'model_sha256': 'new_model'}
        self.service.repredict(row['id'], [5, 5, 195, 195])
        record = self.service.review(row['id'], [5, 5, 195, 195], [20, 30, 160, 170])
        self.assertEqual(record['model']['model_sha256'], 'initial')
        self.assertEqual(record['latest_prediction']['model']['model_sha256'], 'new_model')

    def wait_for_batch(self):
        deadline = time.monotonic()+5
        while self.service.status()['job']['state'] == 'running' and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertNotEqual(self.service.status()['job']['state'], 'running')


if __name__ == '__main__':
    unittest.main()
