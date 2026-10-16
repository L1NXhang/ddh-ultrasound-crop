import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from studio_project import CropProject


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        self.root = Path(self.temp.name)
        self.source = self.root/'原图'
        self.source.mkdir()
        Image.new('RGB', (200, 200), 'gray').save(self.source/'sample.png')
        self.work = self.root/'工作目录'
        self.project = CropProject(self.source, self.work)
        self.row = self.project.rows[0]

    def tearDown(self):
        self.temp.cleanup()

    def save(self, reviewed=False, clean=None):
        return self.project.save(self.row['id'], [10, 10, 190, 190], clean or [20, 30, 150, 170],
                                 {'window': [10, 10, 190, 190], 'crop_box': [10, 30, 190, 170]}, reviewed)

    def test_export_uses_original_pixels_and_separates_review(self):
        rec = self.save(False)
        with Image.open(self.project.output/rec['outputs']['cropped']) as im:
            self.assertEqual(im.size, (130, 140))
        self.assertFalse(rec['human_reviewed'])
        self.assertFalse(rec['approved_for_training'])
        self.assertEqual((self.project.output/'reviewed_training_candidates.jsonl').read_text(), '')
        rec = self.save(True, [22, 30, 150, 170])
        candidates = [json.loads(s) for s in (self.project.output/'reviewed_training_candidates.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['clean_box'], [22, 30, 150, 170])
        self.assertTrue(rec['modified'])
        self.assertEqual(len((self.project.output/'review_history.jsonl').read_text(encoding='utf-8').splitlines()), 2)

    def test_duplicate_names_do_not_overwrite_and_source_is_unchanged(self):
        (self.source/'sub').mkdir()
        Image.new('RGB', (200, 200), 'red').save(self.source/'sub/sample.png')
        Image.new('RGB', (200, 200), 'blue').save(self.source/'sample.jpg')
        before = {p: p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        project = CropProject(self.source, self.work)
        outputs = []
        for row in project.rows:
            rec = project.save(row['id'], [0, 0, 200, 200], [10, 10, 100, 100], {}, False)
            outputs.append(rec['outputs']['cropped'])
        self.assertEqual(len(set(outputs)), 3)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_invalid_geometry_and_output_inside_source_are_rejected(self):
        with self.assertRaises(ValueError):
            CropProject(self.source, self.source/'results')
        with self.assertRaises(ValueError):
            self.save(clean=[0, 0, 150, 170])
        self.assertFalse((self.project.output/'results.jsonl').exists())

    def test_resume_does_not_trust_changed_source(self):
        self.save(True)
        resumed = CropProject(self.source, self.work)
        self.assertTrue(resumed.records[self.row['id']]['human_reviewed'])
        Image.new('RGB', (200, 200), 'red').save(self.source/'sample.png')
        resumed = CropProject(self.source, self.work)
        self.assertNotIn(self.row['id'], resumed.records)
        self.assertEqual((resumed.output/'reviewed_training_candidates.jsonl').read_text(), '')

    def test_final_labels_survive_old_margin_metadata_and_reopening(self):
        clean = [22, 30, 150, 170]
        prediction = {'window': [10, 10, 190, 190], 'crop_box': [10, 26, 190, 174]}
        record = self.project.save(self.row['id'], [10, 10, 190, 190], clean, prediction, True,
                                   model={'margin': .02})
        for _ in range(3):
            reopened = CropProject(self.source, self.work)
            self.assertEqual(reopened.records[self.row['id']]['clean_box'], clean)
            rows = [json.loads(line) for line in (reopened.output/'reviewed_training_candidates.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows[0]['crop'], [30, 170])
            self.assertEqual(rows[0]['clean_box'], clean)
            self.assertEqual(rows[0]['label_semantics'], 'final_crop_box')
            self.assertFalse(rows[0]['approved_for_training'])

    def test_zip_is_optional_and_contains_current_outputs_only(self):
        self.save(False)
        self.assertEqual(list(self.work.rglob('*.zip')), [])
        archive = self.project.archive()
        with zipfile.ZipFile(archive) as z:
            self.assertIn('results.jsonl', z.namelist())
            self.assertIn('reviewed_training_candidates.jsonl', z.namelist())
            self.assertFalse(any(name.startswith('original') for name in z.namelist()))
            self.assertIsNone(z.testzip())


if __name__ == '__main__':
    unittest.main()
