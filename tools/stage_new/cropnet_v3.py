"""cropnet_v3 —— DDH 裁剪模型 v3：可切换 backbone（ResNet18 / ResNet50）× 可切换多尺度融合。

设计约束（与 ddh_v2.py 的 CropNet 保持"同条件对照"）：
  - 输入、预处理、增强、letterbox、loss、优化器、调度、选点口径**完全复用 ddh_v2**，本文件只替换网络。
  - 输出头结构、行卷积、roi_align 参数与 ddh_v2.CropNet 一致（24×16，spatial_scale=1/16）。
  - 原文件 ddh.py / ddh_v2.py / tools/studio_predictor.py **一行不改**（README 警告三处 CropNet 必须同步）。

四种配置（唯一变量是 backbone 与融合方式）：
  r18_l3      backbone=resnet18, fusion='l3'       —— 等价于现有 CropNet（复现基线）
  r50_l3      backbone=resnet50, fusion='l3'
  r18_l123    backbone=resnet18, fusion='l1l2l3'
  r50_l123    backbone=resnet50, fusion='l1l2l3'

多尺度融合：把 layer1/2/3 的特征图 1×1 卷积统一到 64 通道 → 全部双线性上采样到 layer3 的分辨率
（H/16 × W/16）→ 拼接 → 1×1 卷积融合回 64 通道 → 再按 ddh_v2 的做法做 roi_align。
这样"融合"只增加特征通路，不改变后续 head 的任何维度，消融干净。
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    resnet18, resnet50, ResNet18_Weights, ResNet50_Weights)
from torchvision.ops import roi_align

# 各 backbone 的 layer 输出通道数（torchvision 定义，非估算）
BACKBONES = {
    'resnet18': {'ctor': resnet18, 'weights': ResNet18_Weights.DEFAULT,
                 'ch': {'l1': 64, 'l2': 128, 'l3': 256, 'l4': 512}},
    'resnet50': {'ctor': resnet50, 'weights': ResNet50_Weights.DEFAULT,
                 'ch': {'l1': 256, 'l2': 512, 'l3': 1024, 'l4': 2048}},
}

CONFIGS = {
    'r18_l3': {'backbone': 'resnet18', 'fusion': 'l3'},
    'r50_l3': {'backbone': 'resnet50', 'fusion': 'l3'},
    'r18_l123': {'backbone': 'resnet18', 'fusion': 'l1l2l3'},
    'r50_l123': {'backbone': 'resnet50', 'fusion': 'l1l2l3'},
    # ---- 层级选择的验证实验（见 docs/层级选择的理论依据）----
    # 仅用 Layer4：验证"深层分辨率不足导致定位精度受限"（量化下界 16 px）
    'r18_l4': {'backbone': 'resnet18', 'fusion': 'l4'},
    # Layer1+2+3+4 全融合：验证"再往下取无法获得更多有用信息"
    'r18_l1234': {'backbone': 'resnet18', 'fusion': 'l1l2l3l4'},
}


def _ordered(z):
    # 与 ddh_v2.ordered 逐行一致：eps 常在 softmax 下溢时仍为正
    p = (1 - 3e-4) * z.float().softmax(-1) + 1e-4
    return torch.stack((p[..., 0], p[..., 0] + p[..., 1]), -1)


class CropNetV3(nn.Module):
    def __init__(self, config='r18_l3', pretrained=True, num_classes=3):
        super().__init__()
        if config not in CONFIGS:
            raise ValueError(f'unknown config {config}; choose from {list(CONFIGS)}')
        self.config = config
        self.backbone_name = CONFIGS[config]['backbone']
        self.fusion = CONFIGS[config]['fusion']

        spec = BACKBONES[self.backbone_name]
        w = spec['weights'] if pretrained else None
        self.backbone = spec['ctor'](weights=w)
        self.backbone.fc = nn.Identity()
        ch = spec['ch']

        # ---- 行卷积头（与 ddh_v2.CropNet 同维度）----
        self.project = nn.Conv2d(ch['l3'], 64, 1)
        self.row_conv = nn.Sequential(nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(),
                                      nn.Conv1d(64, 32, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(32 * 24, 128), nn.ReLU(), nn.Dropout(.2),
                                  nn.Linear(128, num_classes))
        self.safe_head = nn.Sequential(nn.Linear(32 * 24, 64), nn.ReLU(),
                                       nn.Linear(64, num_classes))
        self.quality_head = nn.Linear(ch['l4'], 1)

        # ---- 多尺度融合通路（按 fusion 配置构建）----
        self.ms_proj = nn.ModuleDict()
        if self.fusion in ('l4',):
            self.ms_proj['l4'] = nn.Conv2d(ch['l4'], 64, 1)
        elif self.fusion in ('l1l2l3',):
            for k in ('l1', 'l2', 'l3'):
                self.ms_proj[k] = nn.Conv2d(ch[k], 64, 1)
            self.ms_fuse = nn.Sequential(nn.Conv2d(64 * 3, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        elif self.fusion in ('l1l2l3l4',):
            for k in ('l1', 'l2', 'l3', 'l4'):
                self.ms_proj[k] = nn.Conv2d(ch[k], 64, 1)
            self.ms_fuse = nn.Sequential(nn.Conv2d(64 * 4, 64, 1), nn.BatchNorm2d(64), nn.ReLU())

    def train(self, mode=True):
        super().train(mode)
        # 与 ddh_v2 一致：冻结 ImageNet BN 统计量，避免小 batch 相关样本污染
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x, roi):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        c1 = b.layer1(x)
        c2 = b.layer2(c1)
        c3 = b.layer3(c2)
        c4 = b.layer4(c3)
        embed = b.avgpool(c4).flatten(1)

        if self.fusion == 'l3':
            feat = self.project(c3)
        elif self.fusion == 'l4':
            # 仅用 Layer4：上采样回 Layer3 的分辨率只是为了走通同一套 head，
            # **不会增加信息**（12×12 的信息量在上采样后仍是 12×12）。
            size = c3.shape[-2:]
            feat = nn.functional.interpolate(self.ms_proj['l4'](c4), size=size,
                                             mode='bilinear', align_corners=False)
        elif self.fusion == 'l1l2l3':
            # 统一到 layer3 的空间分辨率再拼接
            size = c3.shape[-2:]
            f1 = nn.functional.interpolate(self.ms_proj['l1'](c1), size=size, mode='bilinear', align_corners=False)
            f2 = nn.functional.interpolate(self.ms_proj['l2'](c2), size=size, mode='bilinear', align_corners=False)
            f3 = self.ms_proj['l3'](c3)
            feat = self.ms_fuse(torch.cat((f1, f2, f3), 1))
        elif self.fusion == 'l1l2l3l4':
            size = c3.shape[-2:]
            f1 = nn.functional.interpolate(self.ms_proj['l1'](c1), size=size, mode='bilinear', align_corners=False)
            f2 = nn.functional.interpolate(self.ms_proj['l2'](c2), size=size, mode='bilinear', align_corners=False)
            f3 = self.ms_proj['l3'](c3)
            f4 = nn.functional.interpolate(self.ms_proj['l4'](c4), size=size, mode='bilinear', align_corners=False)
            feat = self.ms_fuse(torch.cat((f1, f2, f3, f4), 1))
        else:
            raise ValueError(f'unsupported fusion {self.fusion}')

        boxes = torch.cat((torch.arange(len(x), device=x.device)[:, None], roi), 1)
        aligned = roi_align(feat, boxes, output_size=(24, 16), spatial_scale=1 / 16, aligned=True)
        rows = torch.cat((aligned.mean(-1), aligned.amax(-1)), 1)
        rowvec = self.row_conv(rows).flatten(1)
        return {'bounds': _ordered(self.head(rowvec)),
                'safe': _ordered(self.safe_head(rowvec)),
                'quality': self.quality_head(embed).squeeze(1),
                'embedding': nn.functional.normalize(embed, dim=1)}


def build(config, pretrained=True):
    return CropNetV3(config, pretrained)


def param_breakdown(model):
    """参数量分解，用于成本比较。"""
    def n(mod):
        return sum(p.numel() for p in mod.parameters())
    total = n(model)
    backbone = n(model.backbone)
    return {'total': total, 'backbone': backbone, 'head': total - backbone,
            'backbone_name': model.backbone_name, 'fusion': model.fusion}
