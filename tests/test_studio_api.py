import json
import http.client
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from crop_studio import StudioServer
from studio_service import StudioService
from studio_session import RecentProjects
from test_studio_service import FakeRunner


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        Image.new('RGB', (200, 200), 'gray').save(self.source/'sample.png')
        self.service = StudioService(FakeRunner(), RecentProjects(self.root/'recent.json'))
        self.server = StudioServer(('127.0.0.1', 0), self.service)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, path, body=None, token=None, origin=None):
        headers = {'Content-Type': 'application/json', 'X-DDH-Token': token if token is not None else self.server.token}
        if origin:
            headers['Origin'] = origin
        request = urllib.request.Request(self.base+path, json.dumps(body).encode() if body is not None else None, headers)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def test_configuration_review_zip_and_download_roundtrip(self):
        code, data = self.request('/api/bootstrap')
        self.assertEqual(code, 200)
        self.assertFalse(json.loads(data)['status']['configured'])
        self.assertEqual(self.request('/api/project', {'source': str(self.source), 'work': str(self.root/'work')})[0], 200)
        row = self.service.project.rows[0]
        prediction = json.loads(self.request('/api/predict?id='+row['id'])[1])
        self.assertEqual(prediction['current_clean_box'], [10, 30, 190, 170])
        self.assertEqual(self.request('/api/review', {'id': row['id'], 'window': prediction['window'], 'clean_box': [20, 30, 160, 170]})[0], 200)
        self.assertEqual(list(self.root.rglob('*.zip')), [])
        archive = json.loads(self.request('/api/archive', {})[1])
        code, content = self.request('/api/archive/download?id='+archive['id'])
        self.assertEqual(code, 200)
        self.assertEqual(content[:2], b'PK')

    def test_cross_origin_or_wrong_token_cannot_choose_or_write(self):
        payload = {'source': str(self.source), 'work': str(self.root/'work')}
        self.assertEqual(self.request('/api/project', payload, token='wrong')[0], 403)
        self.assertEqual(self.request('/api/project', payload, origin='http://example.invalid')[0], 403)
        self.assertFalse((self.root/'work').exists())

    def test_rejected_post_returns_json_when_body_arrives_after_headers(self):
        payload = json.dumps({'source': str(self.source), 'work': str(self.root/'work')}).encode()
        for token, origin in [('wrong', self.base), (self.server.token, 'http://example.invalid')]:
            with self.subTest(origin=origin):
                connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
                try:
                    connection.putrequest('POST', '/api/project')
                    for key, value in {'Content-Type': 'application/json', 'Content-Length': str(len(payload)),
                                       'X-DDH-Token': token, 'Origin': origin}.items():
                        connection.putheader(key, value)
                    connection.endheaders()
                    time.sleep(.05)
                    connection.send(payload)
                    response = connection.getresponse()
                    self.assertEqual(response.status, 403)
                    self.assertIn('error', json.loads(response.read()))
                finally:
                    connection.close()
        self.assertIsNone(self.service.project)

    def test_folder_picker_and_static_assets(self):
        with patch('crop_studio.pick_directory', return_value=str(self.source)):
            code, body = self.request('/api/folder', {'initial': str(self.root)})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)['path'], str(self.source))
        for path in ('/', '/studio.css', '/studio.js'):
            self.assertEqual(self.request(path)[0], 200)

    def test_open_output_creates_missing_directory_and_reports_path(self):
        self.assertEqual(self.request('/api/project', {'source': str(self.source), 'work': str(self.root/'work')})[0], 200)
        output = Path(json.loads(self.request('/api/status')[1])['output'])
        with patch('crop_studio.subprocess.Popen') as launch:
            code, body = self.request('/api/output/open', {})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)['path'], str(output))
        launch.assert_called_once()
        self.assertIn('explorer', str(launch.call_args[0][0][0]).lower())
        self.assertTrue(output.is_dir(), '输出目录被删除后应自动重建')

    def test_default_work_prefers_recent_existing_then_documents(self):
        from crop_studio import default_work
        work = self.root/'work'
        self.service.configure(str(self.source), str(work))
        self.assertEqual(default_work(self.service), str(work))
        with patch('crop_studio.documents_dir', return_value=self.root/'docs'), \
                patch.object(RecentProjects, 'entries', return_value=[]):
            self.assertEqual(default_work(self.service), str(self.root/'docs'/'DDH裁剪结果'))

    def test_bootstrap_default_work_uses_last_used_directory(self):
        work = self.root/'somewhere'
        self.service.configure(str(self.source), str(work))
        self.service.remember()
        payload = json.loads(self.request('/api/bootstrap')[1])
        self.assertEqual(payload['default_work'], str(work))

    def test_shutdown_is_authenticated_and_desktop_only(self):
        self.assertEqual(self.request('/api/shutdown', {})[0], 400)
        self.server.desktop = True
        with patch('studio_lifecycle.request_shutdown') as stop:
            self.assertEqual(self.request('/api/shutdown', {}, token='wrong')[0], 403)
            stop.assert_not_called()
            self.assertEqual(self.request('/api/shutdown', {})[0], 200)
            stop.assert_called_once_with(self.server)

    def test_draft_api_checks_session_and_preserves_unconfirmed_edits(self):
        self.request('/api/project', {'source': str(self.source), 'work': str(self.root/'work')})
        rid = self.service.project.rows[0]['id']
        self.request('/api/predict?id='+rid)
        payload = {'id': rid, 'window': [10, 10, 190, 190], 'clean_box': [25, 30, 160, 170]}
        self.assertEqual(self.request('/api/draft', payload, token='wrong')[0], 403)
        self.assertEqual(self.service.status()['drafts'], 0)
        self.assertEqual(self.request('/api/draft', payload)[0], 200)
        self.assertEqual(self.service.status()['saved'], 0)
        bootstrap = json.loads(self.request('/api/bootstrap')[1])
        self.assertEqual(bootstrap['recent_projects'][0]['drafts'], 1)
        self.assertEqual(json.loads(self.request('/api/recent')[1])['rows'][0]['source'], str(self.source))
        self.request('/api/project', {'source': str(self.source), 'work': str(self.root/'work')})
        prediction = json.loads(self.request('/api/predict?id='+rid)[1])
        self.assertTrue(prediction['draft_restored'])
        self.assertEqual(prediction['current_clean_box'], payload['clean_box'])


if __name__ == '__main__':
    unittest.main()
