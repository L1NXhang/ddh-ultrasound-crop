import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from studio_paths import application_root


class PortablePathsTests(unittest.TestCase):
    def test_source_mode_finds_project_from_entry_file(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(application_root(root/'tools/crop_studio.py'), root)

    def test_frozen_mode_uses_executable_location_after_relocation(self):
        root = Path(__file__).resolve().parents[2]/'portable app'/'ddh_multidevice_v2'
        self.assertEqual(application_root(root/'backend/_internal/crop_studio.py',
                                         root/'backend/DDHBackend.exe', True), root)


if __name__ == '__main__':
    unittest.main()
