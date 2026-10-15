import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import crop_app
from crop_geometry import validate_window, validate_crop, modification_details


class GeometryTests(unittest.TestCase):
    def test_original_image_bounds_are_preserved(self):
        self.assertEqual(validate_window([10, 20, 100, 180], (200, 200)), [10, 20, 100, 180])
        self.assertEqual(validate_crop([40, 150], [10, 20, 100, 180]), [40, 150])

    def test_bad_coordinates_fail_before_export(self):
        for box in ([0, 0, 201, 100], [-1, 0, 100, 100], [10, 20, 10, 30],
                    [0, 0, float('nan'), 100], [0, 0, 1.5, 100], [True, 0, 10, 20], None):
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_window(box, (200, 200))
        for crop in ([0, 150], [40, 181], [150, 40], None):
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                validate_crop(crop, [10, 20, 100, 180])

    def test_actual_changes_are_separate_from_clicks(self):
        d = modification_details([10, 20, 100, 180], [40, 150], [10, 20, 100, 180], [40, 150])
        self.assertFalse(d['modified'])
        d = modification_details([10, 20, 100, 180], [40, 150], [10, 20, 110, 180], [40, 150])
        self.assertTrue(d['window_modified'])
        self.assertFalse(d['crop_modified'])
        self.assertEqual(d['window_delta'], [0, 0, 10, 0])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        self.folder = Path(self.temp.name)
        Image.new('RGB', (200, 200), 'gray').save(self.folder / 'sample.png')
        self.state = dict(crop_app.STATE)
        crop_app.STATE.update(rows=[{'id': 'sample', 'file': 'sample.png'}], folder=self.folder,
                              out=self.folder / 'out', done={}, session={}, dry=False,
                              predictions={'sample': {'window': [10, 20, 100, 180], 'crop': [40, 150]}},
                              latest_predictions={}, session_done=set())
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), crop_app.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        crop_app.STATE.clear()
        crop_app.STATE.update(self.state)
        self.temp.cleanup()

    def post(self, endpoint, payload):
        url = f'http://127.0.0.1:{self.server.server_port}{endpoint}'
        req = urllib.request.Request(url, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_repredict_uses_edited_window_and_preserves_original(self):
        with patch.object(crop_app, 'predict_crop', return_value={'crop': [50, 160], 'std': [0, 0]}) as mock:
            code, response = self.post('/api/repredict', {'id': 'sample', 'window': [5, 10, 190, 190]})
        self.assertEqual(code, 200)
        self.assertEqual(response['crop'], [50, 160])
        self.assertEqual(mock.call_args.args[1], [5, 10, 190, 190])
        self.assertEqual(crop_app.STATE['predictions']['sample']['crop'], [40, 150])

    def test_save_exports_original_pixels_and_computes_actual_modification(self):
        payload = {'id': 'sample', 'window': [10, 20, 100, 180], 'crop': [40, 150], 'modified': True}
        code, _ = self.post('/api/save', payload)
        self.assertEqual(code, 200)
        with Image.open(self.folder / 'out/cropped/sample.png') as im:
            self.assertEqual(im.size, (90, 110))
        rec = json.loads((self.folder / 'out/results.jsonl').read_text(encoding='utf-8'))
        self.assertFalse(rec['modified'])
        self.assertEqual(rec['model_window'], payload['window'])
        self.assertEqual(rec['model_crop'], payload['crop'])

    def test_invalid_save_and_repredict_do_not_write(self):
        for endpoint, payload in [('/api/save', {'id': 'sample', 'window': [0, 0, 200, 200], 'crop': [-1, 20]}),
                                  ('/api/repredict', {'id': 'sample', 'window': [0, 0, 201, 200]})]:
            code, _ = self.post(endpoint, payload)
            self.assertEqual(code, 400)
        self.assertFalse((self.folder / 'out/results.jsonl').exists())

    def test_backend_rejects_edited_window_with_stale_prediction(self):
        code, _ = self.post('/api/save', {'id': 'sample', 'window': [5, 10, 190, 190], 'crop': [40, 150]})
        self.assertEqual(code, 400)
        self.assertFalse((self.folder / 'out/results.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
