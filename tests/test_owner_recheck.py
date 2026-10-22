import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from prepare_owner_recheck import render_html, select_records


class OwnerRecheckTests(unittest.TestCase):
    def test_selects_only_uncertain_and_orders_by_filename(self):
        rows = [{'id': 'b', 'filename': '0002.jpg', 'needs_review': True},
                {'id': 'a', 'filename': '0001.jpg', 'needs_review': True},
                {'id': 'c', 'filename': '0003.jpg', 'needs_review': False}]
        result = select_records(rows)
        self.assertEqual([r['id'] for r in result], ['a', 'b'])
        self.assertEqual(rows[0]['id'], 'b')

    def test_rejects_empty_duplicates_and_unsafe_filenames(self):
        for rows in [[], [{'id': 'a', 'filename': '../a.jpg', 'needs_review': True}],
                     [{'id': 'a', 'filename': 'a.jpg', 'needs_review': True}] * 2]:
            with self.assertRaises(ValueError):
                select_records(rows)

    def test_html_prefill_is_distinct_from_confirmed_cache(self):
        template = (Path(__file__).resolve().parents[1] / 'product/边缘信息复核.html').read_text(encoding='utf-8')
        task = {'task_id': 'test', 'member': '成员1', 'rows': [{'id': 'a'}],
                'prefill': {'a': {'note': '</script><script>bad()</script>'}}, 'source_sha256': 'abc'}
        html = render_html(template, task)
        self.assertNotIn('__TASK_JSON__', html)
        self.assertNotIn('</script><script>bad()', html)
        self.assertIn('task.prefill[row.id]', html)
        self.assertIn('owner-recheck-v1:', html)
        self.assertIn("review_author:'张林'", html)
        self.assertIn("done={}", html)
        self.assertIn('覆盖物类型', html)


if __name__ == '__main__':
    unittest.main()
