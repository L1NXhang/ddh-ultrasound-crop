"""Offline end-to-end API smoke test on generated noise, NOT a trained medical model.

Run: python tests/smoke_train.py --workdir /path/to/new/smoke-directory
No pretrained weights are downloaded. Outputs must never be used clinically.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from PIL import Image
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ddh


def main():
    p=argparse.ArgumentParser();p.add_argument('--workdir',required=True);args=p.parse_args()
    root=Path(args.workdir);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);rng=np.random.default_rng(31);rows=[]
    for i in range(16):
        arr=np.zeros((100,120,3),np.uint8)
        arr[10:90,20:100]=rng.integers(0,255,(80,80,3),dtype=np.uint8)
        Image.fromarray(arr).save(root/f'{i}.png')
        s='train' if i<8 else 'val' if i<12 else 'calib' if i<14 else 'test'
        rows.append({'id':str(i),'path':f'{i}.png','window':[20,10,100,90],
            'crop':[20,80] if i%2 else None,'safe_box':[30,25,90,75] if i%2 else None,
            'quality':i%2,'leak_group':f'G{i}','style_group':'synthetic','split':s})
    manifest=root/'labels.jsonl';ddh.write_rows(manifest,rows)
    for stage in ('window','crop'):
        ddh.training(NS(seed=42,manifest=str(manifest),root=str(root),stage=stage,
            device='cpu',no_pretrained=True,baseline=False,size=96,batch=2,epochs=1,patience=2,
            output=str(root/f'{stage}.pt')))
    common=dict(seed=42,manifest=str(manifest),root=str(root),device='cpu',samples=2,
        window_checkpoint=str(root/'window.pt'),crop_checkpoint=str(root/'crop.pt'),calibration='')
    ddh.evaluate(NS(**common,output=str(root/'eval.json'),split='test',oracle_window=False,tolerance=.02))
    ddh.evaluate(NS(**common,output=str(root/'oracle.json'),split='test',oracle_window=True,tolerance=.02))
    ddh.predict(NS(**common,output=str(root/'pred.json'),image=str(root/'15.png'),style='synthetic',preview=''))
    # A random detector may produce no window. That is a valid rejection, not accuracy evidence.
    result=json.loads((root/'pred.json').read_text())
    assert result['review_required'] and not result['eligible_for_quick_confirmation']
    print('SMOKE PASS: both training loops, validation, checkpoint reload, full/oracle evaluation, safe refusal')


if __name__=='__main__':main()
