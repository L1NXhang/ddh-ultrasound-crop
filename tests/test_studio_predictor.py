import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from studio_predictor import ModelRunner
import detect_window


class PredictorTests(unittest.TestCase):
    def test_auto_falls_back_when_gpu_model_initialization_fails(self):
        self.check_fallback(load_failure=True)

    def test_auto_falls_back_when_gpu_prediction_fails(self):
        self.check_fallback(load_failure=False)

    def check_fallback(self, load_failure):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp:
            checkpoint = Path(temp)/'model.pt'
            checkpoint.write_bytes(b'synthetic model')
            runner = ModelRunner(checkpoint)
            initialized = []

            def load(args):
                initialized.append(args.device)
                if load_failure and args.device == 'cuda':
                    raise RuntimeError('synthetic CUDA initialization failure')

            def predict(path, window):
                if not load_failure and runner.device == 'cuda':
                    raise RuntimeError('synthetic GPU kernel failure')
                return {'crop': [30, 170], 'std': [.01, .02]}

            tool = types.SimpleNamespace(load_models=load, STATE={}, predict_crop=predict)
            torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
            with patch.dict(sys.modules, {'torch': torch, 'crop_app': tool}):
                with self.assertLogs(level='WARNING'):
                    result = runner.predict(Path(temp)/'image.png', [10, 10, 190, 190])
            self.assertEqual(initialized, ['cuda', 'cpu'])
            self.assertEqual(runner.device, 'cpu')
            self.assertIn('CPU', runner.metadata['device_note'])
            self.assertEqual(result['crop_box'], [10, 30, 190, 170])

    def test_explicit_gpu_mode_does_not_hide_failures(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp:
            checkpoint = Path(temp)/'model.pt'
            checkpoint.write_bytes(b'synthetic model')
            runner = ModelRunner(checkpoint, device='cuda')
            tool = types.SimpleNamespace(load_models=Mock(side_effect=RuntimeError('GPU failed')), STATE={})
            torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
            with patch.dict(sys.modules, {'torch': torch, 'crop_app': tool}), self.assertRaises(RuntimeError):
                runner.load()

    def test_lazy_load_device_detection_reprediction_and_missing_window(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp:
            checkpoint = Path(temp)/'model.pt'
            checkpoint.write_bytes(b'synthetic model')
            runner = ModelRunner(checkpoint)
            self.assertIsNone(runner.tool)
            tool = types.SimpleNamespace(load_models=Mock(), STATE={},
                                         predict_crop=Mock(return_value={'crop': [30, 170], 'std': [.01, .02]}))
            torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
            detector = types.SimpleNamespace(detect=Mock(return_value={'box': [10, 10, 190, 190], 'flags': []}))
            with patch.dict(sys.modules, {'torch': torch, 'crop_app': tool, 'detect_window': detector}):
                first = runner.predict(Path(temp)/'image.png')
                second = runner.predict(Path(temp)/'image.png', [5, 5, 195, 195])
                detector.detect.return_value = {'box': None, 'flags': ['no_content']}
                third = runner.predict(Path(temp)/'image.png')
            self.assertEqual(runner.device, 'cuda')
            tool.load_models.assert_called_once()
            self.assertEqual(tool.STATE['margin'], 0.)
            self.assertEqual(first['crop_box'], [10, 30, 190, 170])
            self.assertEqual(second['crop_box'], [5, 30, 195, 170])
            self.assertIsNone(third['crop_box'])
            self.assertEqual(third['flags'], ['no_content'])


if __name__ == '__main__':
    unittest.main()
