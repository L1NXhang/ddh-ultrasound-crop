import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from build_portable import allowed_path, clean_inventory


class PackageTests(unittest.TestCase):
    def test_only_application_assets_are_accepted(self):
        for name in ('DDH裁剪工具.exe', 'ddh_multidevice_v2/runs_v2/crop.pt',
                     'ddh_multidevice_v2/backend/_internal/torch/lib/torch_cpu.dll',
                     'ddh_multidevice_v2/tools/webapp/studio.html'):
            self.assertTrue(allowed_path(name), name)
        for name in ('../images/sample.png', 'ddh_multidevice_v2/data/images/sample.png',
                     'ddh_multidevice_v2/data/studio_runtime/recent_projects.json',
                     'ddh_multidevice_v2/backend/annotations.jsonl',
                     'ddh_multidevice_v2/backend/..\\..\\patient.png',
                     'ddh_multidevice_v2/backend/file.dll:private',
                     'ddh_multidevice_v2/backend/patient.jpg', 'ddh_multidevice_v2/backend/patient.png',
                     'results.jsonl', '.venv/pyvenv.cfg'):
            self.assertFalse(allowed_path(name), name)

    def test_inventory_fails_closed_for_unexpected_dataset_files(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp:
            root = Path(temp)
            (root/'DDH裁剪工具.exe').write_bytes(b'synthetic executable')
            self.assertEqual(len(clean_inventory(root)), 1)
            (root/'results.jsonl').write_text('private labels')
            with self.assertRaises(ValueError):
                clean_inventory(root)


if __name__ == '__main__':
    unittest.main()
