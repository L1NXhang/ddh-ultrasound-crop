import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_cleanup_tasks import assign_groups, deduplicate_candidates


class CleanupTaskTests(unittest.TestCase):
    def test_shared_review_and_reserved_groups_are_separate(self):
        groups = [[{'id': str(i), 'group': str(i)}] for i in range(150)]
        result = assign_groups(groups, exclusive=30, common=10, reserve=40)
        a = {r['id'] for r in result['A']}
        b = {r['id'] for r in result['B']}
        reserved = {r['id'] for r in result['reserved']}
        self.assertEqual((len(a), len(b), len(a & b)), (40, 40, 10))
        self.assertFalse((a | b) & reserved)
        self.assertEqual(len(reserved), 40)
        self.assertEqual(groups[0][0], {'id': '0', 'group': '0'})

    def test_entire_similarity_group_is_reserved(self):
        groups = [[{'id': f'{i}a'}, {'id': f'{i}b'}] for i in range(20)]
        result = assign_groups(groups, exclusive=2, common=2, reserve=2)
        self.assertEqual(len(result['reserved']), 4)
        reserved_prefixes = {r['id'][:-1] for r in result['reserved']}
        self.assertFalse(any(r['id'][:-1] in reserved_prefixes for r in result['A']+result['B']))

    def test_insufficient_groups_fail_instead_of_reusing_images(self):
        with self.assertRaises(ValueError):
            assign_groups([[{'id': 'only'}]], exclusive=2, common=2, reserve=2)

    def test_id_collision_quarantines_both_images(self):
        rows = [{'id': 'same', 'pixel_sha256': 'one'}, {'id': 'same', 'pixel_sha256': 'two'},
                {'id': 'ok', 'pixel_sha256': 'three'}]
        valid, conflicts = deduplicate_candidates(rows)
        self.assertEqual([r['id'] for r in valid], ['ok'])
        self.assertEqual(len(conflicts), 2)


if __name__ == '__main__':
    unittest.main()
