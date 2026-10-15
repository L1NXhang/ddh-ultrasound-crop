import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from cleanup_returns import validate_record, reconcile_records, compare_pair, match_overlays


def sample():
    reference = {'id': 'sample', 'filename': '001.png', 'original_path': 'data/images/sample.png',
                 'image_wh': [200, 200], 'pixel_sha256': 'digest'}
    row = dict(reference, task_id='task1', reviewer='成员1', schema='ddh_edge_cleanup_review/v1',
               window=[10, 10, 190, 190], clean_box=[20, 20, 150, 180],
               overlays=[{'kind': 'body_marker', 'box': [160, 140, 185, 180]}],
               overlay_status='present', needs_review=False, note='', anatomy_verified=False,
               approved_for_training=False, coordinate_space='original_pixels', original_size=[200, 200],
               window_source='human_drawn')
    return row, reference


class ReturnValidationTests(unittest.TestCase):
    def test_valid_proposal_remains_unapproved_and_input_unchanged(self):
        row, reference = sample()
        before = copy.deepcopy(row)
        result = validate_record(row, reference)
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['review_reasons'], [])
        self.assertFalse(row['approved_for_training'])
        self.assertEqual(before, row)

    def test_wrong_image_and_bad_coordinates_are_rejected(self):
        changes = [{'pixel_sha256': 'other'}, {'original_path': '../other.png'},
                   {'original_size': [100, 100]}, {'window': [0, 0, 201, 200]},
                   {'window': [0, 0, float('nan'), 200]}, {'clean_box': [0, 0, 150, 180]},
                   {'approved_for_training': True}, {'needs_review': 1}, {'overlays': None}]
        changes += [{'overlay_status': []}, {'overlays': [{'kind': [], 'box': [1, 1, 10, 10]}]}]
        for change in changes:
            row, reference = sample()
            with self.subTest(change=change):
                self.assertTrue(validate_record(dict(row, **change), reference)['errors'])

    def test_missing_fields_are_not_treated_as_negative_labels(self):
        row, reference = sample()
        for key in ('overlay_status', 'approved_for_training', 'needs_review'):
            changed = {k: v for k, v in row.items() if k != key}
            with self.subTest(key=key):
                self.assertTrue(validate_record(changed, reference)['errors'])

    def test_uncertain_without_geometry_is_a_review_case(self):
        row, reference = sample()
        row = dict(row, window=None, clean_box=None, overlays=[], overlay_status='uncertain',
                   needs_review=True, note='窗口下方看不清')
        result = validate_record(row, reference)
        self.assertEqual(result['errors'], [])
        self.assertIn('annotator_uncertain', result['review_reasons'])

    def test_overlap_cannot_be_silently_accepted(self):
        row, reference = sample()
        result = validate_record(dict(row, clean_box=[10, 10, 190, 190]), reference)
        self.assertIn('clean_box_contains_overlay', result['errors'])
        uncertain = dict(row, clean_box=[10, 10, 190, 190], needs_review=True, note='需要裁定')
        result = validate_record(uncertain, reference)
        self.assertEqual(result['errors'], [])
        self.assertIn('clean_box_contains_overlay', result['review_reasons'])


class ReturnComparisonTests(unittest.TestCase):
    def test_duplicate_snapshots_do_not_override_changed_annotation(self):
        row, _ = sample()
        unchanged = dict(row, reviewed_at='later')
        accepted, conflicts, duplicates = reconcile_records([row, unchanged])
        self.assertEqual((len(accepted), len(conflicts), duplicates), (1, 0, 1))
        accepted, conflicts, _ = reconcile_records([row, dict(row, clean_box=[21, 20, 150, 180])])
        self.assertEqual((len(accepted), len(conflicts)), (0, 1))

    def test_overlays_are_matched_by_type_and_location_not_list_order(self):
        row, _ = sample()
        row['overlays'] += [{'kind': 'device_text', 'box': [10, 10, 19, 50]}]
        other = dict(row, overlays=list(reversed(row['overlays'])))
        result = compare_pair(row, other)
        self.assertEqual(result['overlay_matched'], 2)
        self.assertFalse(result['disagreement'])
        result = compare_pair(row, dict(other, overlays=other['overlays'][:1]))
        self.assertTrue(result['disagreement'])

    def test_large_window_difference_and_status_difference_need_review(self):
        row, _ = sample()
        result = compare_pair(row, dict(row, window=[20, 10, 190, 190]))
        self.assertTrue(result['disagreement'])
        self.assertEqual(result['window_max_edge_difference'], 10)
        result = compare_pair(row, dict(row, overlay_status='absent', overlays=[]))
        self.assertIn('overlay_status_difference', result['reasons'])

    def test_overlay_matching_handles_ambiguous_candidates(self):
        a = [{'kind': 'other', 'box': [0, 0, 100, 100]}, {'kind': 'other', 'box': [0, 0, 70, 100]}]
        b = [{'kind': 'other', 'box': [0, 0, 100, 100]}, {'kind': 'other', 'box': [30, 0, 100, 100]}]
        self.assertEqual(len(match_overlays(a, b)), 2)


if __name__ == '__main__':
    unittest.main()
