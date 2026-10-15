import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from layout_profiles import fit_profiles, route


def sample(value=100):
    im = Image.new('RGB', (200, 200))
    draw = ImageDraw.Draw(im)
    draw.rectangle([40, 20, 159, 179], fill=(100, 100, 100))
    draw.rectangle([60, 40, 139, 159], fill=(value, value, value))
    draw.rectangle([180, 50, 184, 150], fill='white')
    return im


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.bank = fit_profiles([{'image': sample(v), 'window': [40, 20, 160, 180]}
                                  for v in (30, 100, 200)])

    def test_anatomy_change_inside_window_does_not_change_layout(self):
        result = route(sample(250), self.bank)
        self.assertEqual(result['route'], 'known_layout_candidate')
        self.assertEqual(result['box'], [40, 20, 160, 180])

    def test_unseen_ui_and_image_size_are_rejected(self):
        for im in (Image.new('RGB', (200, 200), 'white'), sample().resize((400, 400))):
            self.assertEqual(route(im, self.bank)['route'], 'unknown_layout')

    def test_single_reference_is_not_a_valid_profile(self):
        bank = fit_profiles([{'image': sample(), 'window': [40, 20, 160, 180]}])
        self.assertEqual(route(sample(), bank)['route'], 'unknown_layout')


if __name__ == '__main__':
    unittest.main()
