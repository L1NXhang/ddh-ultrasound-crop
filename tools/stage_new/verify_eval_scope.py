"""核实评估口径差异：完整 pipeline（含窗口检测） vs oracle window（只用标注窗口）。

背景：项目文档记载"新口径模型上边界误差中位 17.5 px"，
而本次实验 r18_l3 在 oracle window 口径下是 3.29 px（中位）。
必须先搞清楚这两个数字的差别来自哪里，才能写进报告。

用法：
  .venv/Scripts/python.exe tools/stage_new/verify_eval_scope.py
"""
import json, sys, math
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'tools' / 'stage_new'))

import ddh_v2
from ddh_v2 import letterbox, mc_crop, raw_crop, select_window, load_checkpoint, image_for

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = PROJ / 'data'
MAX_DELTA = 0.04


def eval_full_pipeline(crop_ckpt, window_ckpt, rows, root, device):
    """完整两阶段：predict window -> crop -> bounds。返回逐图误差。"""
    wm, wa = load_checkpoint(str(window_ckpt), device)
    bm, ba = load_checkpoint(str(crop_ckpt), device)
    out = []
    for r in rows:
        im = image_for(r, str(root))
        det = wm([to_tensor(im).to(device)])[0]
        box, reason = select_window(det, *im.size, threshold=.8)
        if reason:
            out.append({'id': r['id'], 'status': reason})
            continue
        x, roi = letterbox(im.crop(tuple(box)), ba['size'])
        o = mc_crop(bm, x[None].to(device), roi[None].to(device), 4)
        pred = o['bounds'][0].cpu().tolist()
        t, b = r['crop']
        y0, y1 = r['window'][1], r['window'][3]
        h = y1 - y0
        pred_top_px = box[1] + pred[0] * (box[3] - box[1])
        pred_bot_px = box[1] + pred[1] * (box[3] - box[1])
        out.append({'id': r['id'], 'status': 'ok',
                    'top_err_px': abs(pred_top_px - t),
                    'bot_err_px': abs(pred_bot_px - b),
                    'window_box': box, 'gt_window': r['window']})
    return out


def eval_oracle_window(crop_ckpt, rows, root, device):
    """oracle：直接用标注窗口裁剪，只评估裁剪头。"""
    bm, ba = load_checkpoint(str(crop_ckpt), device)
    out = []
    for r in rows:
        im = image_for(r, str(root))
        x0, y0, x1, y1 = r['window']
        x, roi = letterbox(im.crop((x0, y0, x1, y1)), ba['size'])
        o = mc_crop(bm, x[None].to(device), roi[None].to(device), 4)
        pred = o['bounds'][0].cpu().tolist()
        t, b = r['crop']
        h = y1 - y0
        out.append({'id': r['id'], 'status': 'ok',
                    'top_err_px': abs(y0 + pred[0] * h - t),
                    'bot_err_px': abs(y0 + pred[1] * h - b)})
    return out


def summ(name, recs):
    ok = [r for r in recs if r['status'] == 'ok']
    fail = len(recs) - len(ok)
    if not ok:
        print(f'{name}: 全部失败'); return {}
    tops = sorted(r['top_err_px'] for r in ok)
    bots = sorted(r['bot_err_px'] for r in ok)
    med = lambda v: v[len(v) // 2]
    res = {'n_total': len(recs), 'n_ok': len(ok), 'n_fail': fail,
           'top_median': round(med(tops), 2), 'top_mean': round(sum(tops) / len(tops), 2),
           'bot_median': round(med(bots), 2), 'bot_mean': round(sum(bots) / len(bots), 2)}
    print(f'{name}:')
    print(f"   n={len(recs)} 成功{len(ok)} 失败{fail}")
    print(f"   上边界 中位 {res['top_median']} px / 均值 {res['top_mean']} px")
    print(f"   下边界 中位 {res['bot_median']} px / 均值 {res['bot_mean']} px")
    return res


def main():
    rows = [json.loads(l) for l in (PROJ / 'data/reports/split.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    test = [r for r in rows if r['split'] == 'test']
    device = torch.device(DEVICE)
    print(f'device={DEVICE}  test={len(test)} 张\n')

    ckpt_v2 = PROJ / 'runs_v2/crop.pt'
    ckpt_win = PROJ / 'runs/window.pt'
    result = {}

    if ckpt_v2.exists() and ckpt_win.exists():
        print('=== 原项目模型 (runs_v2/crop.pt + runs/window.pt) ===')
        result['v2_full_pipeline'] = summ('原模型 · 完整pipeline',
                                          eval_full_pipeline(ckpt_v2, ckpt_win, test, ROOT, device))
        result['v2_oracle_window'] = summ('原模型 · oracle窗口',
                                          eval_oracle_window(ckpt_v2, test, ROOT, device))
    else:
        print('缺少原项目 checkpoint，跳过')

    (PROJ / 'runs/20261103_resnet18_vs_50/reports/eval_scope_comparison.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n-> reports/eval_scope_comparison.json')


if __name__ == '__main__':
    main()
