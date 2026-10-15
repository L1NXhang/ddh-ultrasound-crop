import http.cookiejar
import json
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'tests'))
from test_studio_service import FakeRunner
from studio_service import StudioService
from studio_review_auth import ReviewAuth, ReviewDenied, ReviewLimited
from studio_review_tasks import ReviewTasks, ReviewConflict
from studio_collaboration import Collaboration

ACCOUNTS = [{'username': 'member1', 'password': 'review-one-2026'},
            {'username': 'member2', 'password': 'review-two-2026'}]


class AuthTests(unittest.TestCase):
    def test_password_hash_sessions_expiry_and_logout(self):
        auth = ReviewAuth(ACCOUNTS)
        self.assertNotIn('review-one-2026', repr(auth.accounts))
        cookie, session = auth.login('member1', ACCOUNTS[0]['password'], '127.0.0.1')
        self.assertEqual(auth.session(cookie)['username'], 'member1')
        self.assertTrue(session['csrf'])
        auth.logout(cookie)
        with self.assertRaises(ReviewDenied):
            auth.session(cookie)
        cookie, _ = auth.login('member2', ACCOUNTS[1]['password'], '127.0.0.1')
        with patch('studio_review_auth.time.time', return_value=time.time()+36000):
            with self.assertRaises(ReviewDenied):
                auth.session(cookie)

    def test_bad_credentials_throttled_and_input_validated(self):
        auth = ReviewAuth(ACCOUNTS)
        for _ in range(5):
            with self.assertRaises(ReviewDenied):
                auth.login('member1', 'wrong', '127.0.0.1')
        with self.assertRaises(ReviewLimited):
            auth.login('member1', ACCOUNTS[0]['password'], '127.0.0.1')
        for rows in ([], [{'username': '../escape', 'password': 'long-password'}],
                     [{'username': 'a', 'password': 'short'}], [ACCOUNTS[0], ACCOUNTS[0]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                ReviewAuth(rows)


class ReviewFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'data/experiments')
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        for i in range(4):
            Image.new('RGB', (200, 200), (i*20, 80, 120)).save(self.source/f'image{i}.png')
        self.service = StudioService(FakeRunner())
        self.service.configure(self.source, self.root/'work')

    def tearDown(self):
        self.temp.cleanup()


class TaskTests(ReviewFixture):
    def setUp(self):
        super().setUp()
        self.tasks = ReviewTasks(self.service, ['member1', 'member2'])

    def test_assignments_are_disjoint_and_foreign_image_denied(self):
        one = self.tasks.queue('member1', 'all')
        two = self.tasks.queue('member2', 'all')
        self.assertEqual(len(one), 2)
        self.assertFalse({r['id'] for r in one} & {r['id'] for r in two})
        with self.assertRaises(ReviewDenied):
            self.tasks.image_path('member1', two[0]['id'])
        with self.assertRaises(ValueError):
            self.tasks.queue('member1', 'invalid')

    def test_draft_resume_and_record_provenance(self):
        rid = self.tasks.queue('member1', 'all')[0]['id']
        prediction = self.tasks.predict('member1', rid)
        payload = {'id': rid, 'window': prediction['current_window'],
                   'clean_box': [20, 30, 160, 170], 'task_token': prediction['task_token']}
        self.tasks.draft('member1', payload)
        self.assertFalse(self.service.project.records)
        resumed = ReviewTasks(self.service, ['member1', 'member2'])
        restored = resumed.predict('member1', rid)
        self.assertTrue(restored['draft_restored'])
        self.assertEqual(restored['current_clean_box'], payload['clean_box'])
        payload['task_token'] = restored['task_token']
        result = resumed.review('member1', payload)
        row = self.service.project.records[rid]
        self.assertTrue(result['ok'])
        self.assertEqual(row['reviewer']['username'], 'member1')
        self.assertEqual(row['reviewer']['channel'], 'remote_browser')
        self.assertEqual(row['clean_box'], payload['clean_box'])
        training = json.loads((self.service.project.output/'reviewed_training_candidates.jsonl').read_text())
        self.assertEqual(training['reviewer'], row['reviewer'])
        self.assertFalse(training['approved_for_training'])
        self.assertEqual(resumed.status('member1')['reviewed'], 1)

    def test_version_conflict_and_cross_tab_token_do_not_overwrite(self):
        rid = self.tasks.queue('member1', 'all')[0]['id']
        pred = self.tasks.predict('member1', rid)
        payload = {'id': rid, 'window': pred['current_window'], 'clean_box': pred['current_clean_box'],
                   'task_token': 'stale'}
        with self.assertRaises(ReviewConflict):
            self.tasks.review('member1', payload)
        payload['task_token'] = pred['task_token']
        self.service.project.save(rid, pred['current_window'], pred['current_clean_box'], pred, True)
        with self.assertRaises(ReviewConflict):
            self.tasks.draft('member1', payload)
        self.assertEqual(self.service.project.records[rid]['version'], 1)

    def test_window_reprediction_and_skip(self):
        rid = self.tasks.queue('member1', 'all')[0]['id']
        pred = self.tasks.predict('member1', rid)
        payload = {'id': rid, 'window': [5, 5, 195, 195], 'clean_box': [20, 30, 160, 170],
                   'task_token': pred['task_token']}
        with self.assertRaises(ValueError):
            self.tasks.review('member1', payload)
        result = self.tasks.repredict('member1', payload)
        self.assertEqual(result['window'], payload['window'])
        self.tasks.review('member1', payload)
        rid = self.tasks.queue('member1', 'pending')[0]['id']
        self.tasks.skip('member1', rid)
        self.assertFalse(self.tasks.queue('member1', 'pending'))
        self.assertEqual(len(self.tasks.queue('member1', 'all')), 2)

    def test_invalid_boxes_and_changed_original_are_rejected(self):
        rid = self.tasks.queue('member1', 'all')[0]['id']
        pred = self.tasks.predict('member1', rid)
        payload = {'id': rid, 'window': pred['current_window'], 'clean_box': [0, 0, 200, 200],
                   'task_token': pred['task_token']}
        with self.assertRaises(ValueError):
            self.tasks.review('member1', payload)
        self.tasks.image_path('member1', rid).write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.tasks.image_path('member1', rid)

    def test_local_drafts_and_confirmed_images_are_excluded(self):
        rid = self.service.project.rows[0]['id']
        pred = self.service.predict(rid)
        self.service.save_draft(rid, pred['current_window'], pred['current_clean_box'])
        rid2 = self.service.project.rows[1]['id']
        pred2 = self.service.predict(rid2)
        self.service.review(rid2, pred2['current_window'], pred2['current_clean_box'])
        tasks = ReviewTasks(self.service, ['member1', 'member2'])
        ids = {r['id'] for user in ['member1', 'member2'] for r in tasks.queue(user, 'all')}
        self.assertEqual(len(ids), 2)
        self.assertNotIn(rid, ids)
        self.assertNotIn(rid2, ids)

    def test_restart_after_partial_completion_preserves_other_members_draft(self):
        one = self.tasks.queue('member1', 'all')[0]['id']
        two = self.tasks.queue('member2', 'all')[0]['id']
        pred = self.tasks.predict('member2', two)
        self.tasks.draft('member2', {'id': two, 'window': pred['current_window'],
                                   'clean_box': [20, 30, 160, 170], 'task_token': pred['task_token']})
        pred = self.tasks.predict('member1', one)
        self.tasks.review('member1', {'id': one, 'window': pred['current_window'],
                                    'clean_box': pred['current_clean_box'], 'task_token': pred['task_token']})
        resumed = ReviewTasks(self.service, ['member1', 'member2'])
        restored = resumed.predict('member2', two)
        self.assertTrue(restored['draft_restored'])
        self.assertEqual(restored['current_clean_box'][2], 160)


class GatewayTests(ReviewFixture):
    def setUp(self):
        super().setUp()
        self.manager = Collaboration(self.service, ROOT/'tools/webapp')
        self.service.collaboration = self.manager
        self.manager.start({'accounts': ACCOUNTS, 'listen': '127.0.0.1', 'port': 0})
        self.base = self.manager.status()['urls'][0]
        self.cookiejar = http.cookiejar.CookieJar()
        self.client = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(self.cookiejar))
        self.csrf = ''

    def tearDown(self):
        self.manager.stop()
        super().tearDown()

    def request(self, path, body=None, headers=None):
        hdr = {'Content-Type': 'application/json', 'X-DDH-Token': self.csrf, 'Origin': self.base.rstrip('/')}
        hdr.update(headers or {})
        req = urllib.request.Request(self.base.rstrip('/')+path,
                                     json.dumps(body).encode() if body is not None else None, hdr)
        try:
            with self.client.open(req, timeout=10) as response:
                return response.status, response.read(), response.headers
        except urllib.error.HTTPError as error:
            return error.code, error.read(), error.headers

    def login(self, user=0):
        info = json.loads(self.request('/api/login-info')[1])
        code, content, headers = self.request('/api/login', ACCOUNTS[user], {'X-DDH-Login': info['nonce']})
        self.assertEqual(code, 200, content)
        self.csrf = json.loads(content)['token']
        return headers

    def test_login_image_review_logout_and_files_on_host(self):
        self.assertEqual(self.request('/api/queue')[0], 401)
        self.assertEqual(self.request('/api/image?id=anything')[0], 401)
        self.assertIn(b'data-review-only', self.request('/')[1])
        headers = self.login()
        self.assertIn('HttpOnly', headers['Set-Cookie'])
        self.assertIn('SameSite=Strict', headers['Set-Cookie'])
        data = json.loads(self.request('/api/bootstrap')[1])
        self.assertNotIn(str(self.root), json.dumps(data))
        row = json.loads(self.request('/api/queue?scope=pending')[1])['rows'][0]
        pred = json.loads(self.request('/api/predict?id='+row['id'])[1])
        self.assertNotIn('checkpoint', json.dumps(pred))
        self.assertEqual(self.request('/api/image?id='+row['id'])[1][:8], b'\x89PNG\r\n\x1a\n')
        payload = {'id': row['id'], 'window': pred['current_window'],
                   'clean_box': [20, 30, 160, 170], 'task_token': pred['task_token']}
        self.assertEqual(self.request('/api/draft', payload)[0], 200)
        self.assertEqual(self.request('/api/review', payload)[0], 200)
        record = self.service.project.records[row['id']]
        self.assertTrue((self.service.project.output/record['outputs']['cropped']).is_file())
        self.assertEqual(self.request('/api/logout', {})[0], 200)
        self.assertEqual(self.request('/api/bootstrap')[0], 401)

    def test_csrf_unknown_hosts_and_privileged_routes_denied(self):
        self.login()
        for route in ('/api/project', '/api/folder', '/api/archive', '/api/batch/start', '/api/shutdown', '/api/collaboration/start'):
            self.assertEqual(self.request(route, {})[0], 403, route)
        self.assertEqual(self.request('/api/logout', {}, {'X-DDH-Token': 'wrong'})[0], 403)
        self.assertEqual(self.request('/api/logout', {}, {'Origin': 'https://evil.example'})[0], 403)
        self.assertEqual(self.request('/api/status', headers={'Host': 'evil.example'})[0], 403)
        self.assertEqual(self.request('/api/recent')[0], 403)

    def test_foreign_assigned_image_and_invalid_body_rejected(self):
        self.login()
        other = self.manager.tasks.queue('member2', 'all')[0]['id']
        self.assertEqual(self.request('/api/image?id='+other)[0], 403)
        self.assertEqual(self.request('/api/review', {})[0], 400)
        self.assertEqual(self.request('/api/status', headers={'Origin': 'null'})[0], 403)

    def test_host_configuration_and_batch_locked_until_stop(self):
        with self.assertRaises(ValueError):
            self.service.configure(self.source, self.root/'other')
        with self.assertRaises(ValueError):
            self.service.start_batch()
        rid = self.manager.tasks.queue('member1', 'all')[0]['id']
        self.service.predict(rid)
        with self.assertRaises(ValueError):
            self.service.save_draft(rid, [10, 10, 190, 190], [20, 30, 160, 170])
        self.manager.stop()
        self.assertFalse(self.manager.status()['active'])
        self.service.configure(self.source, self.root/'other')

    def test_bad_login_limits_no_passwords_in_status_and_stop_revokes(self):
        info = json.loads(self.request('/api/login-info')[1])
        for _ in range(5):
            self.assertEqual(self.request('/api/login', {'username': 'member1', 'password': 'wrong'}, {'X-DDH-Login': info['nonce']})[0], 401)
        self.assertEqual(self.request('/api/login', ACCOUNTS[0], {'X-DDH-Login': info['nonce']})[0], 429)
        self.assertNotIn('review-one-2026', json.dumps(self.manager.status()))

    def test_invalid_network_config_and_https_proxy_origin(self):
        self.manager.stop()
        for config in ({'listen': '8.8.8.8'}, {'port': -1}, {'public_url': 'http://example.com'},
                       {'public_url': 'https://user:secret@example.com'}, {'public_url': 'https://example.com/path'}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.manager.start({'accounts': ACCOUNTS, 'listen': '127.0.0.1', 'port': 0, **config})
        self.manager.start({'accounts': ACCOUNTS, 'listen': '127.0.0.1', 'port': 0,
                            'public_url': 'https://review.example.test'})
        self.base = next(url for url in self.manager.status()['urls'] if url.startswith('http:'))
        info = json.loads(self.request('/api/login-info', headers={'Host': 'review.example.test', 'Origin': 'https://review.example.test'})[1])
        code, _, headers = self.request('/api/login', ACCOUNTS[0],
            {'Host': 'review.example.test', 'Origin': 'https://review.example.test', 'X-DDH-Login': info['nonce']})
        self.assertEqual(code, 200)
        self.assertIn('Secure', headers['Set-Cookie'])


if __name__ == '__main__':
    unittest.main()
