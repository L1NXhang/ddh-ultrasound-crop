import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from window_lab import grouped_split, metrics


class WindowExperimentTests(unittest.TestCase):
    def test_related_images_stay_together_and_inputs_are_unchanged(self):
        rows = [{'id': str(i)} for i in range(20)]
        pairs = [{'a': '0', 'b': '1'}, {'a': '1', 'b': '2'}]
        result = grouped_split(rows, pairs)
        self.assertEqual(len({r['split'] for r in result[:3]}), 1)
        self.assertNotIn('split', rows[0])
        self.assertEqual(result, grouped_split(rows, pairs))

    def test_missing_prediction_is_in_metric_denominator(self):
        rows = [{'box': [0, 0, 100, 100], 'window': [0, 0, 100, 100]},
                {'box': None, 'window': [0, 0, 100, 100]}]
        result = metrics(rows)
        self.assertEqual(result['within4'], .5)
        self.assertEqual(result['detected'], 1)

    def test_inward_error_is_distinct_from_outward_error(self):
        inward = metrics([{'box': [40, 0, 100, 100], 'window': [0, 0, 100, 100]}])
        outward = metrics([{'box': [0, 0, 140, 100], 'window': [0, 0, 100, 100]}])
        self.assertEqual(inward['inward_over30'], 1)
        self.assertEqual(outward['inward_over30'], 0)


if __name__ == '__main__':
    unittest.main()
