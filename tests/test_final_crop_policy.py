import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from crop_policy import DEFAULT_MARGIN, FINAL_CROP_POLICY, apply_margin, validate_margin
from studio_predictor import ModelRunner
import crop_app
import ddh
import ddh_v2


class FinalCropPolicyTests(unittest.TestCase):
    def test_default_output_matches_final_bounds_without_expansion(self):
        self.assertEqual(DEFAULT_MARGIN, 0.)
        bounds = [.25, .75]
        self.assertEqual(apply_margin(bounds), bounds)
        self.assertEqual(ddh.raw_crop([20, 100, 520, 700], apply_margin(bounds)), [20, 250, 520, 550])
        self.assertEqual(bounds, [.25, .75])

    def test_expansion_is_explicit_and_clips_at_window(self):
        self.assertEqual(ddh.raw_crop([20, 100, 520, 700], apply_margin([.25, .75], .02)), [20, 238, 520, 562])
        self.assertEqual(apply_margin([0., 1.], .02), [0., 1.])

    def test_invalid_margins_fail_at_boundary(self):
        for value in [-.01, .5, float('nan'), float('inf'), True, '0.02', None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_margin(value)
        for bounds in [[.8, .2], [-.1, .8], [.2, 1.1], [math.nan, .8], [.5, .5]]:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                apply_margin(bounds)

    def test_studio_and_legacy_cli_share_final_box_default(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data/experiments') as temp:
            checkpoint = Path(temp) / 'synthetic.pt'
            checkpoint.write_bytes(b'synthetic')
            runner = ModelRunner(checkpoint)
            self.assertEqual(runner.margin, 0.)
            self.assertEqual(runner.metadata['crop_policy'], FINAL_CROP_POLICY)
        self.assertEqual(crop_app.parser().parse_args(['--folder', 'images']).margin, 0.)

    def test_research_calibration_defaults_do_not_reintroduce_expansion(self):
        args = ['calibrate', '--manifest', 'm.jsonl', '--window-checkpoint', 'w.pt', '--crop-checkpoint', 'c.pt', '--output', 'out.json']
        for module in [ddh, ddh_v2]:
            self.assertEqual(module.parser().parse_args(args).margin, 0.)

    def test_real_predict_entry_uses_zero_margin_by_default(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data/experiments') as temp:
            path = Path(temp) / 'synthetic.png'
            Image.new('RGB', (640, 800), '#696969').save(path)
            fake = SimpleNamespace(letterbox=lambda image, size: (torch.zeros(3, 4, 4), torch.tensor([0., 0., 4., 4.])),
                                   mc_crop=lambda *args: {'bounds': torch.tensor([[.25, .75]]), 'std': torch.zeros(1, 2)}, raw_crop=ddh.raw_crop)
            values = {'models': (None, {'size': 384}, 'cpu'), 'ddh': fake, 'margin': DEFAULT_MARGIN}
            with patch.dict(crop_app.STATE, values):
                self.assertEqual(crop_app.predict_crop(path, [20, 100, 520, 700])['crop'], [250, 550])
                crop_app.STATE['margin'] = .02
                self.assertEqual(crop_app.predict_crop(path, [20, 100, 520, 700])['crop'], [238, 562])

    def test_training_target_is_the_saved_final_crop(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data/experiments') as temp:
            path = Path(temp) / 'synthetic.png'
            Image.new('RGB', (640, 800), '#696969').save(path)
            row = {'id': 'synthetic', 'path': str(path), 'window': [20, 100, 520, 700],
                   'crop': [250, 550], 'model': {'margin': .02}, 'label_semantics': 'final_crop_box'}
            for module in [ddh, ddh_v2]:
                target = module.CropData([row], temp, False, 384)[0]['bounds']
                self.assertEqual(target.tolist(), [.25, .75])


if __name__ == '__main__':
    unittest.main()
