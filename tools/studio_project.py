"""Local project storage, provenance and opt-in archives for the crop studio."""
import hashlib
import json
import os
import threading
import time
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from crop_geometry import validate_window

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def json_lines(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows)


class CropProject:
    def __init__(self, source, work):
        self.source, self.work = Path(source).resolve(), Path(work).resolve()
        if not self.source.is_dir():
            raise ValueError('图片文件夹不存在')
        if self.work == self.source or self.work.is_relative_to(self.source):
            raise ValueError('工作目录请选择图片文件夹之外，避免重复读入裁剪结果')
        key = hashlib.sha256(str(self.source).casefold().encode()).hexdigest()[:10]
        self.output = self.work/'DDH裁剪结果'/(self.source.name+'_'+key)
        self.lock = threading.RLock()
        self.rows = self.inventory()
        if not self.rows:
            raise ValueError('所选文件夹里没有 JPG、PNG 或 BMP 图片')
        self.by_id = {r['id']: r for r in self.rows}
        self.records = self.restore()
        self.output.mkdir(parents=True, exist_ok=True)
        meta = {'schema': 'ddh_crop_project/v1', 'source': str(self.source), 'work': str(self.work),
                'output': str(self.output), 'images': len(self.rows), 'updated_at': time.time()}
        atomic_text(self.output/'project.json', json.dumps(meta, ensure_ascii=False, indent=2))
        if self.records or (self.output/'results.jsonl').exists():
            self.snapshots()

    def inventory(self):
        rows = []
        for path in sorted(self.source.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXT:
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(self.source):
                raise ValueError('图片文件链接指向所选目录之外：'+path.name)
            relative = path.relative_to(self.source).as_posix()
            rid = hashlib.sha256(relative.encode()).hexdigest()[:20]
            rows.append({'id': rid, 'file': relative, 'source_sha256': file_hash(path)})
        return rows

    def restore(self):
        records = {}
        history = self.output/'review_history.jsonl'
        if not history.exists():
            return records
        for number, line in enumerate(history.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ValueError(f'项目历史第{number}行损坏，请保留文件并检查备份') from error
            row = self.by_id.get(record['id'])
            if row and row['source_sha256'] == record.get('source_sha256'):
                files = record.get('outputs', {})
                if not record.get('exported') or all((self.output/p).is_file() for p in files.values()):
                    records[record['id']] = record
        return records

    def image_path(self, rid):
        if rid not in self.by_id:
            raise ValueError('未知图片')
        path = (self.source/self.by_id[rid]['file']).resolve()
        if not path.is_relative_to(self.source):
            raise ValueError('图片路径已改变')
        return path

    def assert_source(self, rid):
        path = self.image_path(rid)
        if file_hash(path) != self.by_id[rid]['source_sha256']:
            raise ValueError('原图已变化，请重新打开该项目')
        return path

    def persist(self, record):
        with self.lock:
            self.output.mkdir(parents=True, exist_ok=True)
            with (self.output/'review_history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json_lines([record]))
                stream.flush()
                os.fsync(stream.fileno())
            self.records = {**self.records, record['id']: record}
            self.snapshots()
        return record

    def snapshots(self):
        rows = list(self.records.values())
        atomic_text(self.output/'results.jsonl', json_lines(rows))
        candidates = [self.training_row(r) for r in rows if r.get('human_reviewed') and r.get('exported')]
        atomic_text(self.output/'reviewed_training_candidates.jsonl', json_lines(candidates))

    def training_row(self, row):
        return {'id': row['source_sha256'], 'path': str(self.source/row['file']),
                'window': row['window'], 'crop': [row['clean_box'][1], row['clean_box'][3]],
                'clean_box': row['clean_box'], 'source_sha256': row['source_sha256'],
                'human_modified': row['modified'], 'human_reviewed': True, 'approved_for_training': False,
                'annotation_source': 'crop_studio/human', 'label_semantics': 'final_crop_box',
                'quality': None, 'safe_box': None,
                'style_group': '', 'source_group': '', 'leak_group': '', 'split': '',
                'model': row.get('model', {}), 'review_record': row['record_id'],
                'reviewer': row.get('reviewer')}

    def make_record(self, rid, window, clean, prediction, reviewed, model):
        row = self.by_id[rid]
        previous = self.records.get(rid, {})
        version = previous.get('version', 0)+1
        original = prediction.get('crop_box')
        modified = window != prediction.get('window') or clean != original
        return {**row, 'schema': 'ddh_crop_studio/v1', 'record_id': f'{rid}_v{version}', 'version': version,
                'coordinate_system': 'original_pixels_xyxy_half_open',
                'path': str(self.source/row['file']), 'window': window, 'clean_box': clean, 'crop_box': clean,
                'crop': [clean[1], clean[3]], 'model_window': prediction.get('window'),
                'model_crop_box': original, 'prediction': prediction, 'model': model or {},
                'modified': modified, 'window_modified': window != prediction.get('window'),
                'crop_modified': clean != original, 'human_reviewed': reviewed, 'approved_for_training': False,
                'review_status': 'human_confirmed' if reviewed else 'automatic_unreviewed',
                'flags': prediction.get('flags', []), 'time': time.time(), 'exported': True}

    def save(self, rid, window, clean, prediction, reviewed, model=None, latest_prediction=None, reviewer=None):
        path = self.assert_source(rid)
        with Image.open(path) as raw:
            image = raw.convert('RGB')
        window = validate_window(window, image.size)
        clean = validate_window(clean, image.size)
        if not (window[0] <= clean[0] < clean[2] <= window[2] and window[1] <= clean[1] < clean[3] <= window[3]):
            raise ValueError('最终裁剪框必须位于成像窗口内')
        with self.lock:
            record = self.make_record(rid, window, clean, prediction, reviewed, model)
            if reviewer is not None:
                record['reviewer'] = reviewer
            record['latest_prediction'] = latest_prediction or prediction
            record['original_size'] = list(image.size)
            stem = Path(path).stem+'__'+record['record_id']
            record['outputs'] = {'cropped': 'cropped/'+stem+'.png', 'annotated': 'annotated/'+stem+'.png'}
            self.write_images(image, record)
            return self.persist(record)

    def write_images(self, image, record):
        for relative in record['outputs'].values():
            (self.output/relative).parent.mkdir(parents=True, exist_ok=True)
        image.crop(tuple(record['clean_box'])).save(self.output/record['outputs']['cropped'])
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        draw.rectangle(record['window'], outline='#ff7575', width=3)
        draw.rectangle(record['clean_box'], outline='#59e4b2', width=3)
        annotated.save(self.output/record['outputs']['annotated'])

    def failure(self, rid, reason, prediction=None):
        row = self.by_id[rid]
        return self.persist({**row, 'schema': 'ddh_crop_studio/v1', 'exported': False,
                             'human_reviewed': False, 'approved_for_training': False,
                             'review_status': 'needs_review', 'error': str(reason),
                             'flags': (prediction or {}).get('flags', []),
                             'prediction': prediction or {}, 'time': time.time()})

    def archive(self):
        with self.lock:
            if not self.records:
                raise ValueError('尚无保存结果，暂时不能打包')
            destination = self.work/'DDH压缩包'
            destination.mkdir(parents=True, exist_ok=True)
            path = destination/(self.output.name+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns())[-6:]+'.zip')
            files = {'project.json', 'results.jsonl', 'reviewed_training_candidates.jsonl'}
            files |= {p for r in self.records.values() for p in r.get('outputs', {}).values()}
            with zipfile.ZipFile(path.with_suffix('.zip.tmp'), 'w', zipfile.ZIP_DEFLATED) as archive:
                for name in sorted(files):
                    file = self.output/name
                    if file.exists():
                        archive.write(file, name)
            path.with_suffix('.zip.tmp').replace(path)
            return path
