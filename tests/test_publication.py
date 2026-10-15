import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from audit_publication import audit_files
from build_installer import installer_defines


class PublicationTests(unittest.TestCase):
    def test_private_data_and_credentials_fail_publication(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/experiments') as temp:
            root = Path(temp)
            (root/'docs').mkdir()
            (root/'docs/passwords.md').write_text('| 用户 | 密码 |\n| x | `123456789` |', encoding='utf-8')
            (root/'README.md').write_text('public instructions', encoding='utf-8')
            result = audit_files(root, ['README.md', 'docs/passwords.md', 'runs_v2/test.json', '.env', 'data/images/patient.png'])
            self.assertEqual(result['checked'], 5)
            self.assertEqual(len(result['rejected']), 4)
            self.assertNotIn('123456789', json.dumps(result))

    def test_public_source_docs_and_synthetic_examples_allowed(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/experiments') as temp:
            root = Path(temp)
            (root/'docs').mkdir()
            (root/'docs/guide.md').write_text('账号与密码由负责人私下提供', encoding='utf-8')
            (root/'annotations.example.jsonl').write_text('{"id":"synthetic_example"}', encoding='utf-8')
            self.assertFalse(audit_files(root, ['docs/guide.md', 'annotations.example.jsonl'])['rejected'])

    def test_installer_sources_and_outputs_are_explicit(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/experiments') as temp:
            root = Path(temp)
            source, out = root/'app', root/'out'
            source.mkdir()
            (source/'DDH裁剪工具.exe').write_bytes(b'MZ')
            args = installer_defines(source, out, 'cpu', '0.5.0')
            self.assertIn('/DVariant=CPU', args)
            self.assertIn('/DAppVersion=0.5.0', args)
            self.assertIn('/DDiskParts=0', args)
            self.assertIn('/DDiskParts=0', installer_defines(source, out, 'auto', '0.5.0'))
            self.assertIn('/DDiskParts=1', installer_defines(source, out, 'auto', '0.5.0', split=True))
            for version in ('../escape', '0.5.0 & bad'):
                with self.assertRaises(ValueError):
                    installer_defines(source, out, 'cpu', version)


if __name__ == '__main__':
    unittest.main()
