"""Unconfirmed edits and local recent-project shortcuts, separate from training records."""
import copy
import json
import threading
import time

from PIL import Image

from crop_geometry import validate_window
from studio_project import atomic_text

RECENT_LIMIT = 6


class RecentProjects:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()

    def entries(self):
        with self.lock:
            if not self.path.exists():
                return []
            try:
                rows = json.loads(self.path.read_text(encoding='utf-8'))
                if not isinstance(rows, list) or any(not isinstance(row, dict) or
                        not all(isinstance(row.get(key), str) for key in ('source', 'work')) for row in rows):
                    raise ValueError('invalid entries')
                return rows[:RECENT_LIMIT]
            except (ValueError, TypeError) as error:
                raise ValueError('最近任务记录损坏，请保留文件并检查：'+str(self.path)) from error

    def remember(self, status):
        with self.lock:
            entry = {key: status[key] for key in ('source', 'work', 'total', 'saved', 'reviewed', 'drafts')}
            entry['opened_at'] = time.time()
            identity = lambda row: (row['source'].casefold(), row['work'].casefold())
            rows = [entry]+[row for row in self.entries() if identity(row) != identity(entry)]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_text(self.path, json.dumps(rows[:RECENT_LIMIT], ensure_ascii=False, indent=2))


class DraftStore:
    def __init__(self, project):
        self.project = project
        self.rows = self.restore()

    def restore(self):
        rows = {}
        for path in (self.project.output/'drafts').glob('*.json'):
            if path.stem not in self.project.by_id:
                continue
            try:
                row = json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(row, dict):
                    raise ValueError('invalid draft')
                rid = path.stem
                record = self.project.records.get(rid, {})
                if row.get('source_sha256') != self.project.by_id[rid]['source_sha256'] or \
                        row.get('base_record') != record.get('record_id'):
                    continue
                if not all(isinstance(row.get(key), dict) for key in ('prediction', 'latest_prediction')):
                    raise ValueError('missing prediction provenance')
                self.validate(rid, row.get('window'), row.get('clean_box'))
                rows[rid] = row
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError('草稿文件损坏，请保留文件并检查：'+str(path)) from error
        return rows

    def validate(self, rid, window, clean):
        with Image.open(self.project.assert_source(rid)) as image:
            for box in (window, clean):
                if box is not None:
                    validate_window(box, image.size)

    def save(self, rid, window, clean, prediction, latest):
        self.validate(rid, window, clean)
        record = self.project.records.get(rid, {})
        row = copy.deepcopy({'schema': 'ddh_crop_draft/v1', 'id': rid,
                             'source_sha256': self.project.by_id[rid]['source_sha256'],
                             'base_record': record.get('record_id'), 'window': window, 'clean_box': clean,
                             'prediction': prediction, 'latest_prediction': latest, 'updated_at': time.time()})
        folder = self.project.output/'drafts'
        folder.mkdir(parents=True, exist_ok=True)
        atomic_text(folder/(rid+'.json'), json.dumps(row, ensure_ascii=False, allow_nan=False))
        self.rows = {**self.rows, rid: row}
        return row

    def discard(self, rid):
        (self.project.output/'drafts'/(rid+'.json')).unlink(missing_ok=True)
        self.rows = {key: value for key, value in self.rows.items() if key != rid}
