"""Guard against silent drift between the CropNet definitions in this repo.

The project carries the CropNet architecture in two places:
  - ddh.py       (models/research baseline; ALSO what the product loads via
                  tools/crop_app.py -> load_models -> ddh.load_checkpoint)
  - ddh_v2.py    (new-policy training core used by tools/train_v2.py and
                  tools/experiments/*)

`tools/studio_predictor.py` is only a lazy loader; it delegates to
`tools/crop_app.py`, which imports `ddh`. It does not define its own network.

If ddh.py and ddh_v2.py drift apart, a checkpoint trained by one will not load
into the other. This test pins the state_dict signature so any structural
change must be deliberate.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import ddh
import ddh_v2


def signature(module):
    """Return {param name: shape} for the CropNet inside a module."""
    model = module.CropNet(pretrained=False)
    return {name: tuple(t.shape) for name, t in model.state_dict().items()}


class CropNetConsistencyTest(unittest.TestCase):
    def test_state_dict_identical_between_ddh_and_ddh_v2(self):
        reference = signature(ddh)
        self.assertTrue(reference, 'ddh.CropNet produced no parameters')
        self.assertEqual(signature(ddh_v2), reference,
                         'ddh_v2.CropNet structure drifted from ddh.CropNet')

    def test_parameter_counts_match(self):
        counts = {name: sum(t.numel() for t in module.CropNet(pretrained=False).parameters())
                  for name, module in (('ddh', ddh), ('ddh_v2', ddh_v2))}
        self.assertEqual(len(set(counts.values())), 1, f'parameter counts differ: {counts}')

    def test_production_checkpoint_loads_into_ddh(self):
        """The checkpoint shipped to users must load into the network the product uses."""
        checkpoint_path = ROOT / 'runs_v2' / 'crop.pt'
        if not checkpoint_path.exists():
            self.skipTest('production checkpoint not present in this checkout')
        import torch
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = state.get('state', state)
        model = ddh.CropNet(pretrained=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        self.assertEqual(list(missing), [], f'missing keys: {missing}')
        self.assertEqual(list(unexpected), [], f'unexpected keys: {unexpected}')


if __name__ == '__main__':
    unittest.main()