"""DDH crop research core v2.1 —— 在不改动原交付 ddh.py 的前提下做两处改进：

  A. 非对称容差损失（--safe-mode derived）：safe_box 不再人工标注，
     而是由裁剪标注推导：容差 δ 以内不罚、往内切重罚、往外扩轻罚。
     这正是"宁可宽、不可窄"的直接编码，省掉全部 safe_box 人工标注。
  B. 步骤 A 的选点指标与部署门控对齐（--select-window）：原实现在验证集上
     只看 top-1 框的 1-IoU，而部署时要求"分数≥阈值且恰好一个框"，
     两者不一致会选出"平均 IoU 好但大量图被拒绝"的检查点。

用法与 ddh.py 完全一致；被改动的只有损失与选点，其余（数据、增强、推理、
校准、评估）逐行相同，便于与旧结果对比。原文件 ddh.py 保持不变。

No patient identities, no diagnostic claims.

Coordinates are ALWAYS in original-image pixels in the input JSONL.
Window and safe_box are xyxy, crop is [top,bottom], end coordinates exclusive.
No EXIF autorotation: label the exact decoded image used here.
Run python ddh.py --help. All inference results require human confirmation.
"""
import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import roi_align, box_iou
from torchvision.transforms.functional import to_tensor


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_rows(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty data or duplicate image IDs')
    return rows


def write_rows(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def image_for(row, root):
    return Image.open(Path(root)/row['path']).convert('RGB')


def validate(rows, root, splits=False):
    group_splits = defaultdict(set)
    for r in rows:
        im = image_for(r, root); w,h = im.size
        x0,y0,x1,y1 = r['window']
        if not all(float(v).is_integer() for v in r['window']):
            raise ValueError('Window coordinates must be integers: '+r['id'])
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError('Invalid window: '+r['id'])
        if r.get('quality') not in (None, 0, 1): raise ValueError('quality must be 0,1,null')
        if r.get('crop') is not None:
            t,b = r['crop']
            if not (y0 <= t < b <= y1): raise ValueError('Invalid crop: '+r['id'])
        if r.get('safe_box') is not None:
            a,t,c,b = r['safe_box']
            if not (x0 <= a < c <= x1 and y0 <= t < b <= y1):
                raise ValueError('safe_box outside true window: '+r['id'])
            if r.get('crop') is not None and not (r['crop'][0] <= t and r['crop'][1] >= b):
                raise ValueError('Expert crop cuts required structures: '+r['id'])
        if splits:
            if not r.get('leak_group') or r.get('split') not in ('train','val','calib','test'):
                raise ValueError('Run audit and reviewed split first')
            group_splits[r['leak_group']].add(r['split'])
    if any(len(s)>1 for s in group_splits.values()):
        raise ValueError('Leakage group crosses splits')
    # Exact decoded images cannot cross splits, even with erroneous group IDs.
    seen = {}
    if splits:
        for r in rows:
            im = image_for(r, root)
            key = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
            if key in seen and seen[key] != r['split']: raise ValueError('Exact-image split leakage')
            seen[key] = r['split']
    return True


def audit(args):
    rows = read_rows(args.manifest); validate(rows,args.root)
    n = len(rows); parent = list(range(n))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j): parent[find(j)] = find(i)
    vectors=[]; bits=[]; exact={}; sources={}; pairs=[]
    for i,r in enumerate(rows):
        im = image_for(r,args.root)
        digest=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        if digest in exact:
            union(i,exact[digest]); pairs.append({'a':r['id'],'b':rows[exact[digest]]['id'],'kind':'exact'})
        else: exact[digest]=i
        # Matching anatomy rather than the surrounding UI reduces layout-only matches.
        win=im.crop(tuple(r['window'])).convert('L')
        a=np.array(win.resize((32,32)),dtype=np.float32).ravel(); a-=a.mean()
        a/=max(float(np.linalg.norm(a)),1e-8); vectors.append(a)
        small=np.array(win.resize((9,8)),dtype=np.int16)
        bits.append((small[:,1:]>small[:,:-1]).ravel())
        src=r.get('source_group')
        if src:
            if src in sources: union(i,sources[src])
            else: sources[src]=i
    vectors=np.stack(vectors); bits=np.stack(bits)
    for i in range(n):
        distances=np.count_nonzero(bits[i+1:] != bits[i],axis=1)
        for k in np.where(distances<=args.hash_distance)[0]:
            j=i+1+int(k); corr=float(vectors[i]@vectors[j])
            if corr>=args.correlation:
                union(i,j); pairs.append({'a':rows[i]['id'],'b':rows[j]['id'],
                    'kind':'near_candidate','dhash_distance':int(distances[k]),'correlation':corr})
    roots={v:f'G{k:05d}' for k,v in enumerate(sorted({find(i) for i in range(n)}))}
    for i,r in enumerate(rows): r['leak_group']=roots[find(i)]; r['split']=''
    write_rows(args.output,rows)
    counts=Counter(r['leak_group'] for r in rows)
    write_json(args.report,{'warning':'Similarity groups are NOT patient identities. Review candidate pairs and source groups before splitting.',
        'n_images':n,'n_groups':len(counts),'group_sizes':dict(counts),'pairs':pairs,
        'hash_distance':args.hash_distance,'correlation':args.correlation})


def split(args):
    if not args.reviewed: raise ValueError('Review groups first, then pass --reviewed; this is NOT patient-level splitting')
    rows=read_rows(args.manifest); groups=defaultdict(list)
    for r in rows:
        if not r.get('leak_group'): raise ValueError('Missing leakage group')
        groups[r['leak_group']].append(r)
    keys=sorted(groups); random.Random(args.seed).shuffle(keys)
    hold={k for k in keys if args.holdout_style and any(r.get('style_group')==args.holdout_style for r in groups[k])}
    remain=[k for k in keys if k not in hold]
    if len(remain)<10: raise ValueError('At least 10 non-held-out groups required for a meaningful four-way split')
    if args.holdout_style and not hold: raise ValueError('Requested held-out style not found')
    n=len(remain); a=int(.65*n); b=a+max(1,int(.15*n)); c=b+max(1,int(.10*n))
    for i,k in enumerate(remain):
        s='train' if i<a else 'val' if i<b else 'calib' if i<c else 'test'
        for r in groups[k]:r['split']=s
    for k in hold:
        for r in groups[k]:r['split']='test'
    write_rows(args.output,rows)
    print(json.dumps({'images':dict(Counter(r['split'] for r in rows)),
        'protocol':'reviewed_similarity_group_not_patient','holdout_style':args.holdout_style},ensure_ascii=False))


def letterbox(im,size=384,augment=False):
    im=im.convert('L').convert('RGB')
    if augment:
        im=ImageEnhance.Brightness(im).enhance(random.uniform(.9,1.1))
        im=ImageEnhance.Contrast(im).enhance(random.uniform(.9,1.1))
        if random.random()<.5: im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w,h=im.size; s=min(size/w,size/h); nw=max(1,round(w*s)); nh=max(1,round(h*s))
    px=(size-nw)//2; py=(size-nh)//2
    canvas=Image.new('RGB',(size,size)); canvas.paste(im.resize((nw,nh),Image.Resampling.BILINEAR),(px,py))
    x=to_tensor(canvas)
    if augment: x=(x+torch.randn_like(x)*.01*x).clamp(0,1)
    x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    return x,torch.tensor([px,py,px+nw,py+nh],dtype=torch.float32)


class CropData(Dataset):
    def __init__(self,rows,root,train=False,size=384):
        self.rows=rows;self.root=root;self.train=train;self.size=size
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; im=image_for(r,self.root); x0,y0,x1,y1=r['window']; h=y1-y0
        x,roi=letterbox(im.crop((x0,y0,x1,y1)),self.size,self.train)
        crop=r.get('crop'); safe=r.get('safe_box'); q=r.get('quality')
        valid=crop is not None and q!=0
        bounds=[(crop[0]-y0)/h,(crop[1]-y0)/h] if valid else [0.,1.]
        sb=[(safe[1]-y0)/h,(safe[3]-y0)/h] if safe else [0.,1.]
        return {'image':x,'roi':roi,'bounds':torch.tensor(bounds,dtype=torch.float32),
            'safe':torch.tensor(sb,dtype=torch.float32),'crop_valid':torch.tensor(valid),
            'safe_valid':torch.tensor(safe is not None and q!=0),
            'quality':torch.tensor(-1. if q is None else float(q)),'id':r['id']}


def ordered(z):
    # epsilon remains positive even if softmax underflows.
    p=(1-3e-4)*z.float().softmax(-1)+1e-4
    return torch.stack((p[...,0],p[...,0]+p[...,1]),-1)


class CropNet(nn.Module):
    def __init__(self,pretrained=True,baseline=False):
        super().__init__();self.baseline=baseline
        self.backbone=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc=nn.Identity()
        self.project=nn.Conv2d(256,64,1)
        self.row_conv=nn.Sequential(nn.Conv1d(128,64,3,padding=1),nn.ReLU(),nn.Conv1d(64,32,3,padding=1),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(512 if baseline else 32*24,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,3))
        self.safe_head=nn.Sequential(nn.Linear(32*24,64),nn.ReLU(),nn.Linear(64,3))
        self.quality_head=nn.Linear(512,1)
    def train(self,mode=True):
        super().train(mode)
        # Freeze ImageNet BN running statistics for small correlated mini-batches.
        for m in self.backbone.modules():
            if isinstance(m,nn.BatchNorm2d):m.eval()
        return self
    def forward(self,x,roi):
        b=self.backbone
        x=b.maxpool(b.relu(b.bn1(b.conv1(x))));x=b.layer1(x);x=b.layer2(x);c3=b.layer3(x)
        c4=b.layer4(c3);embed=b.avgpool(c4).flatten(1)
        boxes=torch.cat((torch.arange(len(x),device=x.device)[:,None],roi),1)
        aligned=roi_align(self.project(c3),boxes,output_size=(24,16),spatial_scale=1/16,aligned=True)
        rows=torch.cat((aligned.mean(-1),aligned.amax(-1)),1)
        rowvec=self.row_conv(rows).flatten(1)
        return {'bounds':ordered(self.head(embed if self.baseline else rowvec)),
            'safe':ordered(self.safe_head(rowvec)),'quality':self.quality_head(embed).squeeze(1),
            'embedding':F.normalize(embed,dim=1)}


def iou1(a,b):
    inter=(torch.minimum(a[:,1],b[:,1])-torch.maximum(a[:,0],b[:,0])).clamp_min(0)
    return inter/((a[:,1]-a[:,0])+(b[:,1]-b[:,0])-inter).clamp_min(1e-6)


def loss_crop(out,batch,baseline=False,delta=0.04,w_in=2.0,w_out=0.25,safe_mode='derived'):
    """delta 为容差（占窗口高度的比例，0.02 ≈ 12px @608px 窗口）。

    derived 模式：safe 边界 = 裁剪标注 ∓ delta，不依赖人工 safe_box：
      预测比 (标注+delta) 还紧 → 算"裁进必保范围"（重罚 w_in）
      预测比 (标注-delta) 还宽 → 算"多留了"（轻罚 w_out）
    labeled 模式：沿用原实现，用人工 safe_box。
    """
    zero=out['bounds'].sum()*0; total=zero; m=batch['crop_valid']
    if m.any():
        total=F.smooth_l1_loss(out['bounds'][m],batch['bounds'][m],beta=.02)+.25*(1-iou1(out['bounds'][m],batch['bounds'][m])).mean()
    if not baseline:
        if safe_mode=='derived':
            if m.any():
                pred=out['bounds'][m];lab=batch['bounds'][m]
                # 上边界：pred 更大=裁得更紧；下边界：pred 更小=裁得更紧
                tight=F.relu(pred[:,0]-(lab[:,0]+delta))+F.relu((lab[:,1]-delta)-pred[:,1])
                loose=F.relu((lab[:,0]-delta)-pred[:,0])+F.relu(pred[:,1]-(lab[:,1]+delta))
                total=total+w_in*tight.mean()+w_out*loose.mean()
            s=batch['safe_valid']
            if s.any():
                pred=out['bounds'][s];lab=batch['bounds'][s]
                total=total+.25*F.smooth_l1_loss(out['safe'][s],lab,beta=.02)
        else:
            s=batch['safe_valid']
            if s.any():
                pred=out['bounds'][s];safe=batch['safe'][s]
                # safe_box already includes clinically agreed margin; do not silently add a second margin.
                containment=(F.relu(pred[:,0]-safe[:,0])+F.relu(safe[:,1]-pred[:,1])).mean()
                total=total+2*containment+.25*F.smooth_l1_loss(out['safe'][s],safe,beta=.02)
        q=batch['quality']>=0
        if q.any():total=total+.25*F.binary_cross_entropy_with_logits(out['quality'][q],batch['quality'][q])
    return total


def window_model(pretrained=True,size=640):
    weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    m=fasterrcnn_mobilenet_v3_large_fpn(weights=weights,weights_backbone=None,
        min_size=size,max_size=int(size*1.5),box_detections_per_img=5)
    m.roi_heads.box_predictor=FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features,2)
    return m


class WindowData(Dataset):
    def __init__(self,rows,root,train=False):self.rows=rows;self.root=root;self.train=train
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];im=image_for(r,self.root);box=torch.tensor([r['window']],dtype=torch.float32)
        if self.train:
            im=ImageEnhance.Brightness(im).enhance(random.uniform(.85,1.15))
            # Horizontal mirroring changes labels consistently.
            if random.random()<.5:
                im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT);old=box.clone()
                box[:,0]=im.width-old[:,2];box[:,2]=im.width-old[:,0]
        return to_tensor(im),{'boxes':box,'labels':torch.ones(1,dtype=torch.int64)}


def collate_detection(batch):return tuple(zip(*batch))
def moved(batch,device):return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def select_window(out,w,h,threshold=.8):
    keep=(out['scores']>=threshold)&(out['labels']==1)
    boxes=out['boxes'][keep];scores=out['scores'][keep]
    if len(boxes)!=1:return None,'missing_or_multiple_windows'
    b=boxes[0].detach().cpu().tolist()
    b=[max(0,math.floor(b[0])),max(0,math.floor(b[1])),min(w,math.ceil(b[2])),min(h,math.ceil(b[3]))]
    if b[2]-b[0]<32 or b[3]-b[1]<32:return None,'window_too_small'
    return b,None


def training(args):
    seed_all(args.seed);rows=read_rows(args.manifest);validate(rows,args.root,True)
    tr=[r for r in rows if r['split']=='train'];va=[r for r in rows if r['split']=='val']
    if not tr or not va:raise ValueError('Need train and val rows')
    dev=torch.device(args.device); is_b=args.stage=='crop'
    model=(CropNet(not args.no_pretrained,args.baseline) if is_b else window_model(not args.no_pretrained,args.size)).to(dev)
    if is_b:
        ds=CropData(tr,args.root,True,args.size);vds=CropData(va,args.root,False,args.size)
        # Limit contribution of large similarity groups, without pretending they are patients.
        counts=Counter(r['leak_group'] for r in tr)
        sampler=WeightedRandomSampler([1/counts[r['leak_group']] for r in tr],len(tr),replacement=True)
        dl=DataLoader(ds,batch_size=args.batch,sampler=sampler,num_workers=0)
        vl=DataLoader(vds,batch_size=args.batch,shuffle=False)
        head=[p for n,p in model.named_parameters() if not n.startswith('backbone.')]
        opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':3e-5},{'params':head,'lr':3e-4}],weight_decay=1e-4)
    else:
        dl=DataLoader(WindowData(tr,args.root,True),batch_size=args.batch,shuffle=True,collate_fn=collate_detection)
        vl=DataLoader(WindowData(va,args.root),batch_size=args.batch,collate_fn=collate_detection)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs)
    best=float('inf');bad=0;history=[]
    for epoch in range(args.epochs):
        model.train();train_loss=0
        for batch in dl:
            opt.zero_grad(set_to_none=True)
            if is_b:
                b=moved(batch,dev);out=model(b['image'],b['roi'])
                loss=loss_crop(out,b,args.baseline,delta=args.delta,w_in=args.w_in,
                               w_out=args.w_out,safe_mode=args.safe_mode)
            else:
                images,targets=batch
                loss=sum(model([x.to(dev) for x in images],[moved(t,dev) for t in targets]).values())
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();train_loss+=float(loss.detach())
        model.eval();values=[]
        with torch.no_grad():
            for batch in vl:
                if is_b:
                    b=moved(batch,dev);out=model(b['image'],b['roi'])
                    values.append((float(loss_crop(out,b,args.baseline,delta=args.delta,w_in=args.w_in,
                                                   w_out=args.w_out,safe_mode=args.safe_mode)),len(b['image'])))
                else:
                    images,targets=batch;outs=model([x.to(dev) for x in images])
                    for o,t in zip(outs,targets):
                        # 与部署门控一致：要求"分数≥阈值且恰好一个框"，否则算 0 IoU；
                        # 漏检/多检都计入分母，不消失。
                        keep=(o['scores']>=getattr(args,'window_threshold',0.8))&(o['labels']==1)
                        ok=bool(int(keep.sum())==1)
                        iou=float(box_iou(o['boxes'][keep].cpu(),t['boxes'])[0,0]) if ok else 0
                        values.append((1-iou,1))
        score=sum(v*n for v,n in values)/sum(n for _,n in values)
        sched.step();rec={'epoch':epoch+1,'train_loss':train_loss/max(1,len(dl)),'val_objective':score};history.append(rec);print(json.dumps(rec))
        if score<best:
            best=score;bad=0;Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            torch.save({'state':model.state_dict(),'stage':args.stage,'size':args.size,'baseline':args.baseline,
                'quality_trained':is_b and not args.baseline and {r.get('quality') for r in tr}>={0,1},
                'safe_trained':is_b and not args.baseline and any(r.get('safe_box') and r.get('quality')!=0 for r in tr),
                'seed':args.seed,'epoch':epoch+1,'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()},args.output)
        else:bad+=1
        if bad>=args.patience:break
    write_json(args.output+'.history.json',history)


def load_checkpoint(path,device):
    state=torch.load(path,map_location=device,weights_only=True)
    model=(CropNet(False,state['baseline']) if state['stage']=='crop' else window_model(False,state['size'])).to(device)
    model.load_state_dict(state['state']);model.eval();return model,state


@torch.no_grad()
def mc_crop(model,x,roi,samples=16):
    if samples<2:raise ValueError('At least 2 stochastic draws required')
    model.eval();det=model(x,roi);bounds=[]
    for m in model.head.modules():
        if isinstance(m,nn.Dropout):m.train()
    for _ in range(samples):bounds.append(model(x,roi)['bounds'])
    p=torch.stack(bounds);model.eval()
    return {'bounds':p.mean(0),'std':p.std(0,unbiased=False),'quality':det['quality'],
        'embedding':det['embedding'],'safe':det['safe']}


def raw_crop(box,bounds):
    x0,y0,x1,y1=box;t,b=bounds
    if not (0<=t<b<=1):raise ValueError('Invalid predicted interval')
    # Outward rounding avoids discarding a boundary pixel.
    return [int(x0),max(int(y0),math.floor(y0+t*(y1-y0))),int(x1),min(int(y1),math.ceil(y0+b*(y1-y0)))]


def contains(c,s):return c[0]<=s[0] and c[1]<=s[1] and c[2]>=s[2] and c[3]>=s[3]


@torch.no_grad()
def pipeline(row,root,wm,bm,bmeta,device,window_threshold=.8,oracle=False,samples=16):
    im=image_for(row,root)
    if oracle:box=row['window'];reason=None
    else:box,reason=select_window(wm([to_tensor(im).to(device)])[0],*im.size,threshold=window_threshold)
    if reason:return {'id':row['id'],'status':reason,'box':None}
    x,roi=letterbox(im.crop(tuple(box)),bmeta['size'])
    o=mc_crop(bm,x[None].to(device),roi[None].to(device),samples)
    return {'id':row['id'],'status':'predicted','box':box,'bounds':o['bounds'][0].cpu().tolist(),
        'std':o['std'][0].cpu().tolist(),'quality_logit':float(o['quality'][0]),
        'embedding':o['embedding'][0].cpu().tolist(),'safe_prediction':o['safe'][0].cpu().tolist()}


def cp_upper(k,n,confidence=.95):
    """Exact one-sided Clopper-Pearson binomial upper bound; no scipy dependency."""
    if n==0 or k>=n:return 1.
    lo,hi=0.,1.;alpha=1-confidence
    for _ in range(64):
        p=(lo+hi)/2
        terms=[math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)+i*math.log(max(p,1e-300))+(n-i)*math.log(max(1-p,1e-300)) for i in range(k+1)]
        mx=max(terms);cdf=math.exp(mx)*sum(math.exp(t-mx) for t in terms)
        if cdf>alpha:lo=p
        else:hi=p
    return hi


def fit_temperature(logits,labels):
    if set(labels)!={0.,1.}:return 1.,False
    z=torch.tensor(logits);y=torch.tensor(labels);best=(float('inf'),1.)
    for t in np.geomspace(.25,4,61):
        loss=float(F.binary_cross_entropy_with_logits(z/float(t),y))
        if loss<best[0]:best=(loss,float(t))
    return best[1],True


def gate(pred,style,cal,meta):
    reasons=[]
    if pred['box'] is None:return [pred['status']]
    if not meta.get('quality_trained'):reasons.append('quality_head_untrained')
    if not meta.get('safe_trained'):reasons.append('safety_head_untrained')
    if not cal.get('enabled',False):reasons.append('calibration_insufficient')
    if not style or style=='unknown' or style not in cal.get('styles',[]):reasons.append('unvalidated_style')
    if max(pred['std'])>cal.get('u_threshold',0):reasons.append('high_disagreement')
    q=1/(1+math.exp(-max(-50,min(50,pred['quality_logit']/cal.get('temperature',1)))))
    if q<cal.get('quality_threshold',1):reasons.append('quality_low')
    bank=np.asarray(cal.get('feature_bank',[]),dtype=np.float32)
    if len(bank):
        dist=float(np.min(np.linalg.norm(bank-np.array(pred['embedding']),axis=1)))
        if dist>cal['distance_threshold']:reasons.append('feature_outlier')
    else:reasons.append('no_reference_bank')
    return reasons


def get_models(args):
    wm,wa=load_checkpoint(args.window_checkpoint,args.device);bm,ba=load_checkpoint(args.crop_checkpoint,args.device)
    if wa['stage']!='window' or ba['stage']!='crop':raise ValueError('Checkpoints exchanged')
    return wm,bm,ba


def calibrate(args):
    if not (0<=args.margin<.5 and 0<args.max_risk<1):raise ValueError('Invalid margin or risk')
    seed_all(args.seed);rows=read_rows(args.manifest);validate(rows,args.root,True);wm,bm,meta=get_models(args)
    # Model selection is on val; the final fixed policy is evaluated on calib.
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val'];held=[r for r in rows if r['split']=='calib']
    if not val or not held:raise ValueError('Need separate val and calib sets')
    bank=[]
    for r in train:
        p=pipeline(r,args.root,wm,bm,meta,args.device,oracle=True,samples=2)
        bank.append(p['embedding'])
    bank=np.asarray(bank,dtype=np.float32)
    vp=[pipeline(r,args.root,wm,bm,meta,args.device,samples=args.samples) for r in val]
    good=[(r,p) for r,p in zip(val,vp) if p['box'] is not None]
    labelled=[(r,p) for r,p in good if r.get('quality') in (0,1)]
    temp,temp_ok=fit_temperature([p['quality_logit'] for r,p in labelled],[float(r['quality']) for r,p in labelled])
    positive=[(r,p) for r,p in good if r.get('quality')==1 and r.get('safe_box')]
    if not positive:raise ValueError('Validation requires positive quality and expert safe_box labels')
    u=float(np.quantile([max(p['std']) for r,p in positive],.95))
    d=float(np.quantile([np.min(np.linalg.norm(bank-np.array(p['embedding']),axis=1)) for r,p in positive],.95))
    styles=sorted({r.get('style_group') for r in train if r.get('style_group') not in (None,'','unknown')})
    cal={'enabled':True,'temperature':temp,'temperature_fitted':temp_ok,'u_threshold':u,
        'distance_threshold':d,'quality_threshold':.9,'margin':args.margin,
        'feature_bank':bank.tolist(),'styles':styles,'mc_samples':args.samples,'seed':args.seed,
        'window_threshold':.8,'protocol':'similarity_group_not_patient',
        'checkpoint_hashes':[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (args.window_checkpoint,args.crop_checkpoint)]}
    # Independent audit of this already-fixed policy, not threshold optimization.
    group_errors=defaultdict(list);unknown=0;all_count=0;eligible=0
    for r in held:
        p=pipeline(r,args.root,wm,bm,meta,args.device,samples=args.samples);all_count+=1
        if gate(p,r.get('style_group'),cal,meta):continue
        eligible+=1
        if r.get('safe_box') is None or r.get('quality') is None:unknown+=1;continue
        bounds=[max(0,p['bounds'][0]-args.margin),min(1,p['bounds'][1]+args.margin)]
        fail=(r['quality']!=1) or not contains(raw_crop(p['box'],bounds),r['safe_box'])
        group_errors[r['leak_group']].append(fail)
    n=len(group_errors);k=sum(any(v) for v in group_errors.values());upper=cp_upper(k,n)
    cal.update({'enabled':bool(temp_ok and meta['quality_trained'] and meta['safe_trained'] and unknown==0 and n>=args.min_groups and upper<=args.max_risk),
        'audit':{'groups':n,'failed_groups':k,'unknown_label_eligible_images':unknown,'eligible_images':eligible,'all_images':all_count,
                 'one_sided_95_upper':upper,'max_risk':args.max_risk,
                 'warning':'Binomial bound is nominal only: true patient independence cannot be verified.'}})
    write_json(args.output,cal);print(json.dumps({k:v for k,v in cal.items() if k!='feature_bank'},ensure_ascii=False,indent=2))


def policy_checked(args):
    c=json.loads(Path(args.calibration).read_text()) if args.calibration else {}
    if c:
        actual=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (args.window_checkpoint,args.crop_checkpoint)]
        if actual!=c['checkpoint_hashes']:raise ValueError('Calibration is for different checkpoints; recalibrate')
    return c


def evaluate(args):
    seed_all(args.seed);rows=read_rows(args.manifest);validate(rows,args.root,True);wm,bm,meta=get_models(args);cal=policy_checked(args)
    rows=[r for r in rows if r['split']==args.split]
    if not rows:raise ValueError('Empty requested split')
    if args.oracle_window and cal:raise ValueError('Oracle-window evaluation must not reuse end-to-end calibration')
    results=[]
    for r in rows:
        p=pipeline(r,args.root,wm,bm,meta,args.device,oracle=args.oracle_window,samples=cal.get('mc_samples',args.samples))
        reasons=gate(p,r.get('style_group'),cal,meta)
        item={'id':r['id'],'group':r['leak_group'],'style':r.get('style_group') or 'unknown','graf':r.get('graf_type') or 'unknown',
            'quality':str(r.get('quality')) if r.get('quality') is not None else 'unknown',
            'status':p['status'],'eligible':not reasons,'reasons':reasons}
        if r.get('human_modified') in (0,1) and r.get('human_modified') is not None:
            item['human_modified']=bool(r['human_modified'])
        if p['box'] is not None:
            margin=cal.get('margin',0);bounds=[max(0,p['bounds'][0]-margin),min(1,p['bounds'][1]+margin)]
            crop=raw_crop(p['box'],bounds);item['crop']=crop
            if r.get('crop') and r.get('quality')!=0:
                t,b=r['crop'];h=r['window'][3]-r['window'][1];et=abs(crop[1]-t);eb=abs(crop[3]-b)
                inter=max(0,min(crop[3],b)-max(crop[1],t));union=crop[3]-crop[1]+b-t-inter
                item.update(top_error_px=et,bottom_error_px=eb,top_error_norm=et/h,bottom_error_norm=eb/h,
                    vertical_iou=inter/max(union,1e-8),tolerance_pass=max(et,eb)/h<=args.tolerance)
                target=torch.tensor([[r['window'][0],t,r['window'][2],b]],dtype=torch.float32)
                item['full_box_iou']=float(box_iou(torch.tensor([crop],dtype=torch.float32),target)[0,0])
            if r.get('safe_box'):
                item['safe']=contains(crop,r['safe_box']);item['critical_cut']=not item['safe']
                item['window_contains_safe']=contains(p['box'],r['safe_box'])
        else:
            # Stage-A failure is a pipeline failure, not a removed sample.
            if r.get('safe_box'):item['safe']=False
            if r.get('crop') and r.get('quality')!=0:item['tolerance_pass']=False
        results.append(item)
    def summary(items):
        out={'images':len(items),'groups':len({r['group'] for r in items}),
            'window_failure_rate':sum(r['status']!='predicted' for r in items)/len(items),
            'eligibility_rate':sum(r['eligible'] for r in items)/len(items)}
        for k in ['top_error_px','bottom_error_px','top_error_norm','bottom_error_norm','vertical_iou','full_box_iou','tolerance_pass','safe','critical_cut','window_contains_safe','human_modified']:
            vals=[r[k] for r in items if k in r];out[k]=float(np.mean(vals)) if vals else None;out[k+'_n']=len(vals)
        accepted=[r for r in items if r['eligible']]; labelled=[r for r in accepted if 'safe' in r and r['quality']!='unknown']
        out['eligible_safe_rate']=sum(r['safe'] and r['quality']=='1' for r in labelled)/len(labelled) if labelled else None
        out['eligible_unknown_safety']=len(accepted)-len(labelled)
        out['edit_proxy_rate']=1-out['tolerance_pass'] if out['tolerance_pass'] is not None else None
        return out
    groups={}
    for key in ('style','graf','quality'):
        groups[key]={v:summary([r for r in results if r[key]==v]) for v in sorted({r[key] for r in results})}
    write_json(args.output,{'protocol':'NOT patient-independent; similarity-group exploratory evaluation',
        'oracle_window':args.oracle_window,'summary':summary(results),'subgroups':groups,'rows':results})
    print(json.dumps(summary(results),indent=2))


def predict(args):
    seed_all(args.seed);wm,bm,meta=get_models(args);cal=policy_checked(args)
    row={'id':Path(args.image).name,'path':args.image}
    p=pipeline(row,'.',wm,bm,meta,args.device,samples=cal.get('mc_samples',args.samples))
    reasons=gate(p,args.style,cal,meta);result={k:v for k,v in p.items() if k!='embedding'}
    result.update(review_required=True,eligible_for_quick_confirmation=not reasons,reasons=reasons)
    if p['box'] is not None:
        margin=cal.get('margin',0);bounds=[max(0,p['bounds'][0]-margin),min(1,p['bounds'][1]+margin)]
        result['crop_box']=raw_crop(p['box'],bounds)
        if args.preview:
            if Path(args.preview).resolve()==Path(args.image).resolve():raise ValueError('Never overwrite original')
            Image.open(args.image).crop(result['crop_box']).save(args.preview)
            result['preview_is_unapproved_candidate']=True
    write_json(args.output,result);print(json.dumps(result,ensure_ascii=False,indent=2))


def parser():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit');a.add_argument('--manifest',required=True);a.add_argument('--root',default='.')
    a.add_argument('--output',default='grouped.jsonl');a.add_argument('--report',default='pairs.json');a.add_argument('--hash-distance',type=int,default=4);a.add_argument('--correlation',type=float,default=.98)
    a=sub.add_parser('split');a.add_argument('--manifest',required=True);a.add_argument('--output',default='split.jsonl');a.add_argument('--reviewed',action='store_true');a.add_argument('--seed',type=int,default=42);a.add_argument('--holdout-style',default='')
    a=sub.add_parser('train');a.add_argument('--stage',choices=['window','crop'],required=True);a.add_argument('--manifest',required=True);a.add_argument('--root',default='.')
    a.add_argument('--output',required=True);a.add_argument('--size',type=int,default=384);a.add_argument('--batch',type=int,default=8);a.add_argument('--epochs',type=int,default=80)
    a.add_argument('--patience',type=int,default=12);a.add_argument('--seed',type=int,default=42);a.add_argument('--no-pretrained',action='store_true');a.add_argument('--baseline',action='store_true')
    a.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a.add_argument('--safe-mode',choices=['derived','labeled'],default='derived',
                   help='derived=由裁剪标注推导安全边界（不需要人工 safe_box）；labeled=沿用原实现')
    a.add_argument('--delta',type=float,default=.04,
                   help='容差，占窗口高度比例。默认 0.04≈24px：实测同一张图两个标注者的'
                        '分歧就有 ±25~37px，容差小于它等于惩罚人工噪声；可用 --delta 0.02 收紧')
    a.add_argument('--w-in',type=float,default=2.,help='往内裁（裁到必保范围）的损失权重')
    a.add_argument('--w-out',type=float,default=.25,help='往外扩（多留）的损失权重')
    a.add_argument('--window-threshold',type=float,default=.8,help='步骤A选点时用的分数门限（与部署一致）')
    for name in ('calibrate','evaluate','predict'):
        a=sub.add_parser(name);a.add_argument('--window-checkpoint',required=True);a.add_argument('--crop-checkpoint',required=True)
        a.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a.add_argument('--seed',type=int,default=42);a.add_argument('--samples',type=int,default=16);a.add_argument('--output',required=True)
        if name!='predict':a.add_argument('--manifest',required=True);a.add_argument('--root',default='.')
        if name=='calibrate':
            a.add_argument('--margin',type=float,default=.02);a.add_argument('--min-groups',type=int,default=30);a.add_argument('--max-risk',type=float,default=.05)
        else:a.add_argument('--calibration',default='')
        if name=='evaluate':
            a.add_argument('--split',default='test',choices=['val','calib','test']);a.add_argument('--oracle-window',action='store_true');a.add_argument('--tolerance',type=float,default=.02)
        if name=='predict':a.add_argument('--image',required=True);a.add_argument('--style',default='unknown');a.add_argument('--preview',default='')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    {'audit':audit,'split':split,'train':training,'calibrate':calibrate,'evaluate':evaluate,'predict':predict}[args.command](args)
