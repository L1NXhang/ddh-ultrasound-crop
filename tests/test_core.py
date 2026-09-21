"""Synthetic, offline tests. No clinical accuracy is measured."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ddh

torch.set_num_threads(2)


class CoreTests(unittest.TestCase):
    def test_order(self):
        z=torch.tensor([[1e4,-1e4,0.],[-1e4,1e4,0.],[0.,0.,0.]],requires_grad=True)
        b=ddh.ordered(z)
        self.assertTrue(((b[:,0]>=0)&(b[:,1]<=1)&(b[:,0]<b[:,1])).all())
        b.sum().backward();self.assertTrue(torch.isfinite(z.grad).all())

    def test_coordinate_mapping(self):
        self.assertEqual(ddh.raw_crop([10,20,110,220],[.1,.8]),[10,40,110,180])
        self.assertEqual(ddh.raw_crop([10,20,110,221],[.1,.8]),[10,40,110,181])
        with self.assertRaises(ValueError):ddh.raw_crop([0,0,20,20],[.9,.1])

    def test_letterbox(self):
        x,roi=ddh.letterbox(Image.new('RGB',(200,100)),384)
        self.assertEqual(tuple(x.shape),(3,384,384))
        self.assertEqual(roi.tolist(),[0.,96.,384.,288.])

    def test_iou(self):
        a=torch.tensor([[.1,.8],[.1,.4]])
        b=torch.tensor([[.1,.8],[.6,.9]])
        self.assertTrue(torch.allclose(ddh.iou1(a,b),torch.tensor([1.,0.])))

    def test_binomial_bound(self):
        self.assertAlmostEqual(ddh.cp_upper(0,59),1-.05**(1/59),places=7)
        self.assertLess(ddh.cp_upper(0,59),.05)
        self.assertGreater(ddh.cp_upper(0,30),.05)
        self.assertEqual(ddh.cp_upper(0,0),1.)

    def test_model_gradient_mc_checkpoint(self):
        model=ddh.CropNet(False);model.train()
        x=torch.randn(2,3,96,96);roi=torch.tensor([[0.,0.,96.,96.],[0.,10.,96.,86.]])
        out=model(x,roi)
        self.assertEqual(tuple(out['bounds'].shape),(2,2))
        self.assertEqual(tuple(out['embedding'].shape),(2,512))
        batch={'bounds':torch.tensor([[.1,.9],[.2,.8]]),'safe':torch.tensor([[.2,.8],[.3,.7]]),
            'crop_valid':torch.tensor([True,False]),'safe_valid':torch.tensor([True,False]),'quality':torch.tensor([1.,0.])}
        loss=ddh.loss_crop(out,batch);loss.backward()
        for head in (model.head,model.safe_head,model.quality_head):
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        o=ddh.mc_crop(model,x,roi,3)
        self.assertTrue((o['std']>=0).all())
        self.assertFalse(model.training)
        self.assertTrue(all(not m.training for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.pt';torch.save({'state':model.state_dict(),'stage':'crop','size':96,'baseline':False},p)
            loaded,meta=ddh.load_checkpoint(p,'cpu')
            self.assertTrue(torch.allclose(model(x,roi)['bounds'],loaded(x,roi)['bounds']))

    def test_missing_labels(self):
        p=torch.tensor([[.2,.8]],requires_grad=True)
        batch={'crop_valid':torch.tensor([False]),'safe_valid':torch.tensor([False]),'quality':torch.tensor([-1.])}
        loss=ddh.loss_crop({'bounds':p},batch)
        loss.backward();self.assertEqual(float(loss),0.)

    def test_window_reject(self):
        out={'scores':torch.tensor([.99,.98]),'labels':torch.tensor([1,1]),
            'boxes':torch.tensor([[0.,0.,80.,80.],[90.,0.,170.,80.]])}
        box,reason=ddh.select_window(out,200,100)
        self.assertIsNone(box);self.assertIsNotNone(reason)

    def test_group_audit_and_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rng=np.random.default_rng(123);rows=[]
            for i in range(20):
                path=root/f'{i}.png';Image.fromarray(rng.integers(0,255,(64,64,3),dtype=np.uint8)).save(path)
                rows.append({'id':str(i),'path':path.name,'window':[0,0,64,64],
                    'crop':[5,60],'safe_box':[4,10,60,50],'quality':1,'source_group':'seq' if i<2 else ''})
            manifest=root/'m.jsonl';ddh.write_rows(manifest,rows)
            args=SimpleNamespace(manifest=str(manifest),root=d,output=str(root/'g.jsonl'),report=str(root/'pairs.json'),hash_distance=4,correlation=.98)
            ddh.audit(args);grouped=ddh.read_rows(args.output)
            self.assertEqual(grouped[0]['leak_group'],grouped[1]['leak_group'])
            ddh.split(SimpleNamespace(manifest=args.output,output=str(root/'s.jsonl'),reviewed=True,seed=42,holdout_style=''))
            splitrows=ddh.read_rows(root/'s.jsonl');self.assertTrue(ddh.validate(splitrows,d,True))
            splitrows[1]['split']='test' if splitrows[0]['split']!='test' else 'train'
            with self.assertRaises(ValueError):ddh.validate(splitrows,d,True)
            self.assertTrue(ddh.CropData(rows,d)[0]['safe_valid'])

    def test_exact_duplicate_different_groups(self):
        with tempfile.TemporaryDirectory() as d:
            Image.new('RGB',(64,64),'gray').save(Path(d)/'x.png')
            common={'path':'x.png','window':[0,0,64,64],'quality':None}
            rows=[dict(common,id='a',leak_group='a',split='train'),dict(common,id='b',leak_group='b',split='test')]
            with self.assertRaises(ValueError):ddh.validate(rows,d,True)

    def test_gate_default_manual(self):
        p={'box':[0,0,100,100],'std':[.001,.001],'quality_logit':10.,'embedding':[0.,1.]}
        self.assertIn('calibration_insufficient',ddh.gate(p,'unknown',{},{}))
        cal={'enabled':True,'styles':['layout1'],'u_threshold':.01,'temperature':1.,'quality_threshold':.9,
            'feature_bank':[[0.,1.]],'distance_threshold':.2}
        meta={'quality_trained':True,'safe_trained':True}
        self.assertEqual(ddh.gate(p,'layout1',cal,meta),[])
        self.assertIn('unvalidated_style',ddh.gate(p,'unknown',cal,meta))

    def test_calibration_policy_audit(self):
        # Fixed synthetic predictions isolate policy bookkeeping from model accuracy.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=[]
            for i in range(66):
                s='train' if i<2 else 'val' if i<6 else 'calib'
                q=0 if i in (3,5) else 1
                rows.append({'id':str(i),'path':'unused.png','window':[0,0,100,100],
                    'safe_box':[10,30,90,70],'quality':q,'style_group':'layout1','leak_group':str(i),'split':s})
            ddh.write_rows(root/'m.jsonl',rows)
            (root/'a.pt').write_bytes(b'checkpoint-a');(root/'b.pt').write_bytes(b'checkpoint-b')
            args=SimpleNamespace(manifest=str(root/'m.jsonl'),root=d,window_checkpoint=str(root/'a.pt'),
                crop_checkpoint=str(root/'b.pt'),device='cpu',seed=42,samples=2,margin=.02,min_groups=30,
                max_risk=.05,output=str(root/'cal.json'))
            def fake_pipeline(r,*unused,**kwargs):
                return {'id':r['id'],'status':'predicted','box':[0,0,100,100],'bounds':[.2,.8],
                    'std':[.001,.001],'quality_logit':8. if r['quality']==1 else -8.,'embedding':[1.,0.]}
            meta={'quality_trained':True,'safe_trained':True}
            with patch.object(ddh,'validate',return_value=True),patch.object(ddh,'get_models',return_value=(None,None,meta)),patch.object(ddh,'pipeline',side_effect=fake_pipeline),patch('builtins.print'):
                ddh.calibrate(args)
                cal=json.loads(Path(args.output).read_text());self.assertTrue(cal['enabled'])
                self.assertEqual(cal['audit']['groups'],60)
                self.assertEqual(cal['audit']['failed_groups'],0)
                rows[-1]['safe_box']=None;ddh.write_rows(root/'m.jsonl',rows)
                ddh.calibrate(args)
                cal=json.loads(Path(args.output).read_text());self.assertFalse(cal['enabled'])
                self.assertEqual(cal['audit']['unknown_label_eligible_images'],1)


if __name__=='__main__':unittest.main()
