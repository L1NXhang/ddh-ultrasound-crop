"""Project lifecycle, human review and cancellable automatic batches."""
import copy
import logging
import threading
import time

from PIL import Image

from crop_geometry import validate_window
from studio_project import CropProject
from studio_session import DraftStore


class StudioService:
    def __init__(self, runner, recents=None):
        self.runner = runner
        self.recents = recents
        self.drafts = None
        self.project = None
        self.collaboration = None
        self.predictions = {}
        self.latest = {}
        self.archives = {}
        self.skipped = set()
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.job = {'state': 'idle', 'processed': 0, 'total': 0, 'saved': 0, 'failed': 0}

    def idle(self):
        if self.job['state'] == 'running':
            raise ValueError('批量处理正在运行，请完成或停止后再操作')
        if self.project is None:
            raise ValueError('请先选择图片文件夹和工作目录')

    def configure(self, source, work):
        with self.lock:
            if self.collaboration and self.collaboration.active:
                raise ValueError('请先停止协作审核，再切换图片文件夹')
            if self.job['state'] == 'running':
                raise ValueError('请先停止当前批量任务')
            project = CropProject(source, work)
            drafts = DraftStore(project)
            self.project = project
            self.drafts = drafts
            self.predictions, self.latest, self.archives = {}, {}, {}
            self.skipped = set()
            self.job = {'state': 'idle', 'processed': 0, 'total': 0, 'saved': 0, 'failed': 0}
            self.remember()
            return self.status()

    def remember(self):
        if self.recents:
            self.recents.remember(self.status())

    def recent_projects(self):
        return self.recents.entries() if self.recents else []

    def status(self):
        project = self.project
        records = list(project.records.values()) if project else []
        adjustments = []
        for record in records:
            if not (record.get('human_reviewed') and record.get('exported')):
                continue
            prediction = record.get('latest_prediction') or record.get('prediction') or {}
            clean, model = record.get('clean_box'), prediction.get('crop_box')
            if clean and model and len(clean) == 4 and len(model) == 4:
                adjustments.append((abs(clean[1]-model[1])+abs(clean[3]-model[3]))/2)
        return {'configured': project is not None, 'source': str(project.source) if project else '',
                'work': str(project.work) if project else '', 'output': str(project.output) if project else '',
                'total': len(project.rows) if project else 0, 'saved': sum(bool(r.get('exported')) for r in records),
                'reviewed': sum(bool(r.get('human_reviewed')) for r in records),
                'failed': sum(not r.get('exported') for r in records), 'skipped': len(self.skipped),
                'drafts': len(self.drafts.rows) if self.drafts else 0,
                'avg_adjust_px': round(sum(adjustments)/len(adjustments)) if adjustments else None,
                'flagged': sum(bool(r.get('flags')) for r in records), 'job': copy.deepcopy(self.job),
                'device': self.runner.device, 'model': self.runner.metadata}

    def queue(self, scope='pending'):
        if self.project is None:
            return []
        rows = []
        for row in self.project.rows:
            record = self.project.records.get(row['id'], {})
            reviewed = bool(record.get('human_reviewed'))
            draft = row['id'] in self.drafts.rows
            if scope == 'reviewed' and not reviewed or scope == 'pending' and (reviewed and not draft or row['id'] in self.skipped):
                continue
            rows.append(dict(row, reviewed=reviewed, saved=bool(record.get('exported')),
                             flags=record.get('flags', []), error=record.get('error'),
                             draft=draft, skipped=row['id'] in self.skipped))
        return rows

    def predict(self, rid):
        with self.lock:
            self.idle()
            path = self.project.assert_source(rid)
            record = self.project.records.get(rid, {})
            draft = self.drafts.rows.get(rid, {})
            if rid not in self.predictions:
                prediction = draft.get('prediction') or (record.get('prediction') if record.get('exported') else None)
                origin_model = record.get('model') if prediction else self.runner.metadata
                prediction = prediction or self.runner.predict(path)
                prediction = dict(prediction, model=copy.deepcopy(prediction.get('model') or origin_model or self.runner.metadata))
                self.predictions[rid] = copy.deepcopy(prediction)
                self.latest[rid] = copy.deepcopy(draft.get('latest_prediction') or record.get('latest_prediction') or prediction)
            initial = copy.deepcopy(self.predictions[rid])
            window = draft.get('window', record.get('window', self.latest[rid].get('window')))
            clean = draft.get('clean_box', record.get('clean_box', self.latest[rid].get('crop_box')))
            with Image.open(path) as image:
                size = list(image.size)
            return dict(initial, id=rid, file=self.project.by_id[rid]['file'], original_size=size,
                        current_window=window, current_clean_box=clean,
                        prediction_window=self.latest[rid].get('window'), latest_prediction=copy.deepcopy(self.latest[rid]),
                        draft_restored=bool(draft), reviewed=bool(record.get('human_reviewed')))

    def save_draft(self, rid, window, clean):
        with self.lock:
            self.idle()
            self.local_edit(rid)
            if rid not in self.project.by_id or rid not in self.predictions:
                raise ValueError('请先加载本图的预测')
            row = self.drafts.save(rid, window, clean, self.predictions[rid], self.latest[rid])
            self.remember()
            return {'ok': True, 'updated_at': row['updated_at']}

    def repredict(self, rid, window):
        with self.lock:
            self.idle()
            self.local_edit(rid)
            path = self.project.assert_source(rid)
            with Image.open(path) as image:
                window = validate_window(window, image.size)
            result = dict(self.runner.predict(path, window), model=copy.deepcopy(self.runner.metadata))
            self.latest[rid] = copy.deepcopy(result)
            return result

    def review(self, rid, window, clean):
        with self.lock:
            self.idle()
            self.local_edit(rid)
            if rid not in self.predictions:
                raise ValueError('请先加载本图的预测')
            if window != self.latest[rid].get('window'):
                raise ValueError('窗口已修改，请先按当前窗口重新预测')
            record = self.project.save(rid, window, clean, self.predictions[rid], True,
                                       self.predictions[rid]['model'], copy.deepcopy(self.latest[rid]))
            self.skipped.discard(rid)
            self.drafts.discard(rid)
            self.remember()
            return record

    def start_batch(self):
        with self.lock:
            self.idle()
            if self.collaboration and self.collaboration.active:
                raise ValueError('请先停止协作审核，再开始批量裁剪')
            rows = [r for r in self.project.rows if not self.project.records.get(r['id'], {}).get('exported')
                    and r['id'] not in self.drafts.rows]
            self.cancel_event.clear()
            self.job = {'state': 'running', 'processed': 0, 'total': len(rows), 'saved': 0,
                        'failed': 0, 'current': '', 'started': time.time()}
            threading.Thread(target=self.batch_worker, args=(self.project, rows), daemon=True).start()
            return copy.deepcopy(self.job)

    def local_edit(self, rid):
        if self.collaboration and self.collaboration.owns(rid):
            raise ValueError('这张图片正在分配给组员审核，请先停止协作或选择其他图片')

    def batch_worker(self, project, rows):
        try:
            for row in rows:
                if self.cancel_event.is_set():
                    break
                self.job = dict(self.job, current=row['file'])
                prediction = None
                try:
                    prediction = self.runner.predict(project.assert_source(row['id']))
                    if not prediction.get('crop_box'):
                        raise ValueError('没有可靠的成像窗口，需要人工框选')
                    project.save(row['id'], prediction['window'], prediction['crop_box'], prediction, False, self.runner.metadata)
                    self.job = dict(self.job, saved=self.job['saved']+1)
                except Exception as error:
                    logging.exception('Automatic crop failed: %s', row['file'])
                    project.failure(row['id'], str(error), prediction)
                    self.job = dict(self.job, failed=self.job['failed']+1)
                self.job = dict(self.job, processed=self.job['processed']+1)
            self.remember()
            self.job = dict(self.job, state='cancelled' if self.cancel_event.is_set() else 'completed', finished=time.time())
        except Exception as error:
            logging.exception('Batch storage failed')
            self.job = dict(self.job, state='error', error=str(error), finished=time.time())

    def archive(self):
        with self.lock:
            self.idle()
            path = self.project.archive()
            key = str(time.time_ns())
            self.archives[key] = path
            return {'id': key, 'path': str(path), 'name': path.name, 'bytes': path.stat().st_size}
