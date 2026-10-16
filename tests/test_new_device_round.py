import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import prepare_new_device_round as prepare
import audit_new_device_round as audit


class NewDeviceRoundTests(unittest.TestCase):
    def test_balancing_preserves_whole_similarity_groups(self):
        groups = [[{'id': f'{i}-{j}', 'collect_group': str(i % 5), 'image_wh': [100, 100]}
                   for j in range(n)] for i, n in enumerate([3, 2, 2, 1, 1, 1, 1, 1])]
        before = json.dumps(groups)
        buckets = prepare.balanced(groups)
        owner = {r['id']: i for i, bucket in enumerate(buckets) for r in bucket}
        self.assertEqual(len(owner), 12)
        for group in groups:
            self.assertEqual(len({owner[r['id']] for r in group}), 1)
        self.assertLessEqual(max(map(len, buckets)) - min(map(len, buckets)), 1)
        self.assertEqual(json.dumps(groups), before)
        self.assertEqual(prepare.balanced(groups), buckets)

    def test_pixel_fingerprint_ignores_png_metadata_and_filenames(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "data/experiments") as temp:
            base = Path(temp)
            image = Image.new('RGB', (40, 60), '#536275')
            image.save(base / 'a.png')
            image.save(base / 'b.png', compress_level=0)
            rows = [prepare.feature({'id': name, 'path': str(base / name)}) for name in ['a.png', 'b.png']]
            self.assertEqual(rows[0]['pixel_sha256'], rows[1]['pixel_sha256'])
            groups, _ = prepare.group_features(rows)
            self.assertEqual(len(groups), 1)

    def test_different_features_do_not_form_a_near_duplicate_group(self):
        rows = [{'id': 'a', 'pixel_sha256': 'a', 'vector': np.array([1., 0.]), 'bits': np.zeros(64, dtype=bool)},
                {'id': 'b', 'pixel_sha256': 'b', 'vector': np.array([0., 1.]), 'bits': np.ones(64, dtype=bool)}]
        groups, pairs = prepare.group_features(rows)
        self.assertEqual(len(groups), 2)
        self.assertEqual(pairs, [])

    def test_registry_checks_declared_count_and_unique_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "data/experiments") as temp:
            base = Path(temp)
            manifest = base / 'tasks.jsonl'
            manifest.write_text('{"id":"a"}\n', encoding='utf-8')
            package = {'manifest': 'tasks.jsonl', 'task_id': 't1', 'member': '成员1', 'images': 1}
            config = base / '分配审计.json'
            config.write_text(json.dumps({'packages': [package]}), encoding='utf-8')
            with patch.object(audit, 'ROOT', base):
                self.assertEqual(audit.registry(base)['t1']['reviewer'], '成员1')
                manifest.write_text('{"id":"a"}\n{"id":"a"}\n', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'count or identities'):
                    audit.registry(base)


if __name__ == '__main__':
    unittest.main()
