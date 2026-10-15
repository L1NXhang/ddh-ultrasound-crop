"""Disjoint reviewer tasks with durable drafts and optimistic version checks."""
import copy
import json
import secrets
import time

from PIL import Image

from crop_geometry import validate_window
from studio_project import atomic_text
from studio_review_auth import ReviewDenied


class ReviewConflict(ValueError):
    pass


def public_prediction(row):
    allowed = ('window', 'crop_box', 'flags', 'bounds_std')
    return copy.deepcopy({key: row[key] for key in allowed if key in row})


class ReviewTasks:
    def __init__(self, service, usernames):
        self.service, self.project = service, service.project
        self.lock = service.lock
        self.run_id = secrets.token_hex(12)
        self.closed = False
        self.states, self.skipped = {}, {user: set() for user in usernames}
        eligible = [row['id'] for row in self.project.rows if row['id'] not in service.drafts.rows
                    and not self.project.records.get(row['id'], {}).get('human_reviewed')]
        if not eligible:
            raise ValueError('没有可分配图片：已审核图片和本机草稿不会分配给组员')
        self.assignments = self.allocate(usernames, eligible)
        audit = {'schema': 'ddh_review_assignment/v1', 'run_id': self.run_id,
                 'created_at': time.time(), 'assignments': self.assignments}
        atomic_text(self.project.output/'collaboration'/(self.run_id+'.json'), json.dumps(audit, ensure_ascii=False))
        atomic_text(self.project.output/'collaboration/last_assignment.json', json.dumps(audit, ensure_ascii=False))

    def allocate(self, users, eligible):
        result = {user: [] for user in users}
        path = self.project.output/'collaboration/last_assignment.json'
        previous = {}
        if path.exists():
            try:
                previous = json.loads(path.read_text(encoding='utf-8'))['assignments']
                if not isinstance(previous, dict) or any(not isinstance(ids, list) or
                        any(not isinstance(rid, str) for rid in ids) for ids in previous.values()):
                    raise ValueError('invalid assignments')
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError('协作分配记录损坏，请联系负责人检查') from error
        assigned = set()
        for user in users:
            result[user] = [rid for rid in previous.get(user, []) if rid in eligible and rid not in assigned]
            assigned.update(result[user])
        for rid in eligible:
            if rid not in assigned:
                user = min(users, key=lambda name: len(result[name]))
                result[user] = result[user]+[rid]
                assigned.add(rid)
        return result

    def check(self, user, rid):
        if self.closed:
            raise ReviewDenied('负责人已停止协作审核')
        if rid not in self.assignments.get(user, []):
            raise ReviewDenied('这张图片未分配给当前账号')
        return rid

    def owns(self, rid):
        return any(rid in ids for ids in self.assignments.values())

    def image_path(self, user, rid):
        self.check(user, rid)
        return self.project.assert_source(rid)

    def queue(self, user, scope='pending'):
        if scope not in ('pending', 'reviewed', 'all'):
            raise ValueError('无效的队列筛选')
        with self.lock:
            rows = []
            for rid in self.assignments.get(user, []):
                record = self.project.records.get(rid, {})
                reviewed = bool(record.get('human_reviewed'))
                if scope == 'pending' and (reviewed or rid in self.skipped[user]) or scope == 'reviewed' and not reviewed:
                    continue
                rows.append({'id': rid, 'file': self.project.by_id[rid]['file'], 'reviewed': reviewed,
                             'saved': bool(record.get('exported')), 'flags': record.get('flags', []),
                             'draft': self.draft_path(user, rid).exists(), 'skipped': rid in self.skipped[user]})
            return rows

    def status(self, user):
        rows = self.queue(user, 'all')
        return {'configured': True, 'source': '分配给 '+user+' 的图片', 'work': '', 'output': '主机项目目录',
                'total': len(rows), 'saved': sum(r['saved'] for r in rows), 'reviewed': sum(r['reviewed'] for r in rows),
                'drafts': sum(r['draft'] for r in rows), 'skipped': len(self.skipped[user]), 'failed': 0, 'flagged': 0,
                'job': {'state': 'idle'}, 'device': self.service.runner.device, 'model': {}, 'username': user}

    def draft_path(self, user, rid):
        return self.project.output/'remote_drafts'/user/(rid+'.json')

    def initial_state(self, user, rid):
        record = self.project.records.get(rid, {})
        path = self.draft_path(user, rid)
        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding='utf-8'))
                if saved['source_sha256'] == self.project.by_id[rid]['source_sha256'] and saved['base_record'] == record.get('record_id'):
                    self.validate(rid, saved['window'], saved['clean_box'])
                    if not all(isinstance(saved[key], dict) for key in ('prediction', 'latest_prediction')):
                        raise ValueError('missing provenance')
                    return {**saved, 'draft_restored': True}
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError('远程草稿损坏，请联系负责人检查') from error
        prediction = copy.deepcopy(record.get('prediction') or self.service.runner.predict(self.project.assert_source(rid)))
        latest = copy.deepcopy(record.get('latest_prediction') or prediction)
        return {'source_sha256': self.project.by_id[rid]['source_sha256'], 'base_record': record.get('record_id'),
                'prediction': prediction, 'latest_prediction': latest,
                'window': record.get('window', latest.get('window')),
                'clean_box': record.get('clean_box', latest.get('crop_box')), 'draft_restored': False,
                'model': copy.deepcopy(record.get('model') or self.service.runner.metadata)}

    def predict(self, user, rid):
        with self.lock:
            path = self.image_path(user, rid)
            key = (user, rid)
            state = self.states.get(key)
            if not state or state['base_record'] != self.project.records.get(rid, {}).get('record_id'):
                state = self.initial_state(user, rid)
            state = {**state, 'task_token': secrets.token_urlsafe(24)}
            self.states = {**self.states, key: state}
            with Image.open(path) as image:
                size = list(image.size)
            return {**public_prediction(state['prediction']), 'id': rid, 'file': self.project.by_id[rid]['file'],
                    'original_size': size, 'current_window': state['window'], 'current_clean_box': state['clean_box'],
                    'prediction_window': state['latest_prediction'].get('window'),
                    'latest_prediction': public_prediction(state['latest_prediction']),
                    'draft_restored': state['draft_restored'], 'reviewed': bool(self.project.records.get(rid, {}).get('human_reviewed')),
                    'task_token': state['task_token']}

    def state_for(self, user, payload):
        rid = self.check(user, payload['id'])
        state = self.states.get((user, rid))
        if not state or not secrets.compare_digest(str(payload.get('task_token', '')), state['task_token']):
            raise ReviewConflict('这张图片已在其他页面打开，请重新加载后继续')
        if state['base_record'] != self.project.records.get(rid, {}).get('record_id'):
            raise ReviewConflict('这张图片已有更新，请重新加载，原记录未被覆盖')
        self.project.assert_source(rid)
        return rid, state

    def validate(self, rid, window, clean):
        with Image.open(self.project.assert_source(rid)) as image:
            for box in (window, clean):
                if box is not None:
                    validate_window(box, image.size)

    def write_draft(self, user, rid, state):
        saved = {key: value for key, value in state.items() if key != 'task_token'}
        atomic_text(self.draft_path(user, rid), json.dumps(saved, ensure_ascii=False, allow_nan=False))

    def draft(self, user, payload):
        with self.lock:
            rid, state = self.state_for(user, payload)
            self.validate(rid, payload.get('window'), payload.get('clean_box'))
            state = {**state, 'window': payload.get('window'), 'clean_box': payload.get('clean_box'),
                     'updated_at': time.time(), 'draft_restored': True}
            self.write_draft(user, rid, state)
            self.states = {**self.states, (user, rid): state}
            return {'ok': True, 'updated_at': state['updated_at']}

    def repredict(self, user, payload):
        with self.lock:
            rid, state = self.state_for(user, payload)
            self.validate(rid, payload['window'], None)
            if payload['window'] is None:
                raise ValueError('请先框选成像窗口')
            latest = self.service.runner.predict(self.project.assert_source(rid), payload['window'])
            state = {**state, 'latest_prediction': latest, 'window': latest['window'], 'clean_box': latest['crop_box'],
                     'updated_at': time.time(), 'draft_restored': True}
            self.write_draft(user, rid, state)
            self.states = {**self.states, (user, rid): state}
            return public_prediction(latest)

    def review(self, user, payload):
        with self.lock:
            rid, state = self.state_for(user, payload)
            if payload['window'] != state['latest_prediction'].get('window'):
                raise ValueError('窗口已修改，请先按当前窗口重新预测')
            record = self.project.save(rid, payload['window'], payload['clean_box'], state['prediction'], True,
                state['model'], state['latest_prediction'],
                reviewer={'username': user, 'channel': 'remote_browser', 'collaboration_id': self.run_id})
            self.draft_path(user, rid).unlink(missing_ok=True)
            self.states = {key: value for key, value in self.states.items() if key != (user, rid)}
            self.skipped[user].discard(rid)
            self.service.predictions.pop(rid, None)
            self.service.latest.pop(rid, None)
            return {'ok': True, 'record_id': record['record_id']}

    def skip(self, user, rid):
        with self.lock:
            self.check(user, rid)
            self.skipped[user].add(rid)
            return {'ok': True}
