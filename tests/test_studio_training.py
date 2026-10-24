"""训练服务的端到端测试：真实子进程在合成小数据上跑 1 个 epoch（CPU，无预训练）。"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from studio_training import TrainingService


def make_manifest(root, count=12):
    images = root/'images'
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []
    for i in range(count):
        name = f'{i}.png'
        arr = np.zeros((100, 120, 3), np.uint8)
        arr[10:90, 20:100] = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
        Image.fromarray(arr).save(images/name)
        rows.append({'id': str(i), 'path': f'images/{name}', 'window': [20, 10, 100, 90],
                     'crop': [20, 80], 'safe_box': [30, 25, 90, 75], 'quality': i % 2,
                     'leak_group': f'G{i}', 'style_group': 'synthetic',
                     'split': 'train' if i < 8 else 'val'})
    manifest = root/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    return manifest


class TrainingServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        self.root = Path(self.temp.name)
        self.service = TrainingService(self.root/'runtime')
        self.manifest = make_manifest(self.root)

    def tearDown(self):
        self.service.stop()
        self.temp.cleanup()

    def wait_for(self, timeout=600):
        deadline = time.time()+timeout
        while time.time() < deadline:
            job = self.service.status()
            if not job['running']:
                return job
            time.sleep(.5)
        self.fail('训练超时未结束')

    def test_train_job_completes_and_writes_checkpoint(self):
        output = self.root/'models/crop.pt'
        self.service.start({'manifest': str(self.manifest), 'root': str(self.root),
                            'output': str(output), 'epochs': 1, 'size': 96, 'batch': 2,
                            'device': 'cpu', 'no_pretrained': True})
        job = self.wait_for()
        self.assertEqual(job['state'], 'completed', '日志：'+'\n'.join(job['lines'][-12:]))
        self.assertEqual(job['result']['output'], str(output))
        self.assertTrue(output.is_file(), '训练产物应写到指定位置')
        self.assertTrue(any('TRAIN_RESULT' in x for x in job['lines']))
        # 已结束的作业允许再启动（不会误报“已有训练在进行”）
        self.service.start({'manifest': str(self.manifest), 'root': str(self.root),
                            'output': str(self.root/'models/again.pt'), 'epochs': 1,
                            'size': 96, 'batch': 2, 'device': 'cpu', 'no_pretrained': True})
        job = self.wait_for()
        self.assertEqual(job['state'], 'completed')

    def test_start_rejects_missing_inputs(self):
        with self.assertRaises(ValueError):
            self.service.start({'manifest': str(self.root/'nope.jsonl'), 'root': str(self.root),
                                'output': str(self.root/'m.pt')})
        with self.assertRaises(ValueError):
            self.service.start({'manifest': '', 'root': '', 'output': ''})
        self.assertEqual(self.service.status()['state'], 'idle')

    def test_manifest_without_split_is_audited_and_split(self):
        rows = [json.loads(x) for x in self.manifest.read_text(encoding='utf-8').splitlines()]
        for r in rows:
            r.pop('split')
        path = self.root/'no_split.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        output = self.root/'models/ns.pt'
        self.service.start({'manifest': str(path), 'root': str(self.root), 'output': str(output),
                            'epochs': 1, 'size': 96, 'batch': 2, 'device': 'cpu', 'no_pretrained': True})
        job = self.wait_for()
        self.assertEqual(job['state'], 'completed', '日志：'+'\n'.join(job['lines'][-12:]))
        split_file = output.parent/'_audit_split.jsonl'
        self.assertTrue(split_file.is_file(), '缺 split 时应自动产出划分清单')
        splits = {json.loads(x)['split'] for x in split_file.read_text(encoding='utf-8').splitlines() if x.strip()}
        self.assertIn('train', splits)
        self.assertIn('val', splits)


if __name__ == '__main__':
    unittest.main()
