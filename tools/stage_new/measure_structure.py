"""A 项：实测 ResNet18/ResNet50 各层级输出形状、通道数、感受野、相对输入尺寸。

不靠论文抄数：全部用 hook 在实际模型上跑一次 forward 读出来。
输出 runs/20261103_resnet18_vs_50/reports/resnet_structure_raw.json
"""
import json, sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools' / 'stage_new'))

IN = 384  # 项目实际输入尺寸


def conv_layers(module):
    """按执行顺序返回主干支路的 [(name, conv)]。

    修正记录：此前版本用 module.named_modules() 递归收集全部 Conv2d，会把残差支路的
    downsample.0（1×1、stride=2）也算作主干卷积。该卷积 kernel=1 不扩大感受野，
    却把 jump 再乘一次 2，导致 layer2/3/4 的感受野虚高 4 倍
    （旧值 131/483/1891，正确值 99/211/435）。此处显式排除 downsample.*。
    """
    return [(n, m) for n, m in module.named_modules()
            if isinstance(m, nn.Conv2d) and 'downsample' not in n]


def receptive_field(stages):
    """stages: [(layer_name, kernel_list)] 按顺序，2D conv 的 (k, stride) 列表。
    返回恰好覆盖该层时的理论感受野（像素），含 maxpool。
    用 r_out = r_in + (k-1)*jump_in ; jump_out = jump_in*stride。
    """
    r, j = 1, 1
    kernel_list = []
    for name, mods in stages:
        for m in mods:
            if isinstance(m, (nn.Conv2d, nn.MaxPool2d)):
                k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
                s = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride)
                k, s = k[0], s[0]
                r = r + (k - 1) * j
                j = j * s
                kernel_list.append((name, k, s))
    return r, j


def stage_convs(module):
    return [m for _, m in conv_layers(module)]


def measure(model, tag):
    shapes = {}
    h = {}
    def mk(name):
        def fn(mod, inp, out):
            shapes[name] = {'shape': list(out.shape), 'channels': out.shape[1],
                            'h': out.shape[2], 'w': out.shape[3]}
        return fn
    targets = {
        'conv1': model.conv1, 'maxpool': model.maxpool,
        'layer1': model.layer1, 'layer2': model.layer2,
        'layer3': model.layer3, 'layer4': model.layer4,
    }
    hs = [t.register_forward_hook(mk(k)) for k, t in targets.items()]
    with torch.no_grad():
        model(torch.zeros(1, 3, IN, IN))
    for x in hs:
        x.remove()

    # 感受野：逐步累加
    b = model
    stages = [('conv1', [b.conv1]), ('maxpool', [b.maxpool]),
              ('layer1', stage_convs(b.layer1)), ('layer2', stage_convs(b.layer2)),
              ('layer3', stage_convs(b.layer3)), ('layer4', stage_convs(b.layer4))]
    rf = {}
    r, j = 1, 1
    eff = b.conv1.stride[0]  # 第一层 stride 2
    cum_stages = []
    for i, (name, mods) in enumerate(stages):
        cum_stages.append((name, mods))
        r, j = receptive_field(cum_stages)
        rf[name] = {'rf_px': int(r), 'jump_px': int(j),
                    'stride_vs_input': int(j),
                    'rf_norm': round(r / IN, 4)}

    # 参数/FLOPs（只算 conv，近似）
    per_stage_params = {}
    for name, mod in targets.items():
        per_stage_params[name] = sum(p.numel() for p in mod.parameters())

    total = sum(p.numel() for p in model.parameters())
    return {'backbone': tag, 'input': [IN, IN], 'stages': shapes, 'receptive_field': rf,
            'params_total': total, 'params_per_stage': per_stage_params,
            'params_fc': sum(p.numel() for p in model.fc.parameters())}


def main():
    out = {}
    for tag, ctor in [('resnet18', resnet18), ('resnet50', resnet50)]:
        m = ctor(weights=None)
        out[tag] = measure(m, tag)
    # 关键说明：大分辨率下的等效
    p = Path(__file__).resolve().parents[2] / 'runs' / '20261103_resnet18_vs_50' / 'reports'
    p.mkdir(parents=True, exist_ok=True)
    (p / 'resnet_structure_raw.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
