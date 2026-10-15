"""Export inference-only parameters, removing research metadata from distributable weights."""
import argparse
import hashlib
import json

import torch


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    original = torch.load(args.source, map_location='cpu', weights_only=True)
    if original['stage'] != 'crop':
        raise ValueError('Expected a crop checkpoint')
    minimal = {key: original[key] for key in ('state', 'stage', 'size', 'baseline')}
    torch.save(minimal, args.destination)
    restored = torch.load(args.destination, map_location='cpu', weights_only=True)
    expected, actual = tensor_hash(original['state']), tensor_hash(restored['state'])
    if expected != actual:
        raise ValueError('Runtime weights differ from production parameters')
    print(json.dumps({'schema': 'ddh_runtime_model/v1', 'tensor_sha256': actual,
                      'checkpoint_keys': sorted(restored), 'equivalent_parameters': True,
                      'stage': minimal['stage'], 'size': minimal['size'], 'baseline': minimal['baseline']}))
