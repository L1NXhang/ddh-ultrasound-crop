"""项目状态面板：一条命令看清所有成果在哪、卡在哪、下一步做什么。

用法：python tools/status.py
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def jsonl(p):
    p = ROOT / p
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []


def mark(ok):
    return '[完成]' if ok else '[待办]'


def main():
    print('=' * 76)
    print('DDH 裁剪项目状态面板')
    print('=' * 76)
    print(f'项目目录: {ROOT}')

    # 1 环境
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f'[完成] 环境: torch {torch.__version__} | CUDA {cuda} | '
              f'{torch.cuda.get_device_name(0) if cuda else "无 GPU"}')
    except Exception as e:
        print(f'[注意] 环境: 无法导入 torch（{e}）——先运行 .venv\\Scripts\\activate')

    # 2 原图
    imgs = list((ROOT / 'data' / 'images').rglob('*.jpg')) + list((ROOT / 'data' / 'images').rglob('*.png'))
    todo = jsonl('data/annotations.todo.jsonl')
    print(f'{mark(bool(imgs))} 原图: {len(imgs)} 张，清单骨架 {len(todo)} 行')
    print(f'     来源分布: {dict(Counter(r.get("collect_group") or "未标" for r in todo))}')

    # 3 标签覆盖（读最新清单，优先 annotations.jsonl）
    cur = 'data/annotations.jsonl' if (ROOT / 'data' / 'annotations.jsonl').exists() else 'data/annotations.todo.jsonl'
    rows = jsonl(cur)
    n = len(rows)
    have_w = sum(r.get('window') is not None for r in rows)
    have_c = sum(r.get('crop') is not None for r in rows)
    have_s = sum(r.get('safe_box') is not None for r in rows)
    have_q = sum(r.get('quality') is not None for r in rows)
    print(f'标签覆盖（来自 {cur}，共 {n} 行）:')
    print(f'  {mark(have_w == n)} window : {have_w}/{n}')
    print(f'  {mark(have_c == n)} crop   : {have_c}/{n}')
    print(f'  {mark(have_s > 0)} safe_box: {have_s}/{n}   （不能从裁切图反推，必须人工标注）')
    print(f'  {mark(have_q > 0)} quality : {have_q}/{n}   （必须人工评定）')

    # 4 裁切成品投放口
    inbox = ROOT / 'data' / 'crops_inbox'
    if inbox.exists():
        parts = {d.name: len(list(d.glob('*.jpg'))) + len(list(d.glob('*.png')))
                 for d in sorted(inbox.iterdir()) if d.is_dir()}
        print(f'crops_inbox（裁切成品投放口）: {parts}')

    # 5 可训练清单与产物
    for f in ['data/pilot.jsonl', 'data/annotations.jsonl']:
        if (ROOT / f).exists():
            print(f'可训练清单: {f}（{len(jsonl(f))} 行）')
    runs = sorted((ROOT / 'runs').glob('*')) if (ROOT / 'runs').exists() else []
    print(f'{mark(bool(runs))} 训练产物 runs/: {[p.name for p in runs] if runs else "还没有（尚未正式训练）"}')

    # 6 下一步
    print('-' * 76)
    print('下一步:')
    tpl = json.loads((ROOT / 'data' / 'window_templates.json').read_text(encoding='utf-8')) \
        if (ROOT / 'data' / 'window_templates.json').exists() else {'templates': {}}
    confirmed = [k for k, v in tpl.get('templates', {}).items() if v.get('confirmed')]
    if not confirmed:
        print('  1) 打开 data/review/window_template_corners.png 核对窗口模板，')
        print('     确认后把 data/window_templates.json 里 "confirmed" 改成 true，')
        print('     再运行 python tools/build_manifest.py --apply')
    elif (ROOT / 'data' / 'pilot.jsonl').exists() and not runs:
        print('  1) 模板已确认。现在就能用 pilot 清单跑通训练:')
        print('     python ddh.py audit --manifest data/pilot.jsonl --root data '
              '--output data/reports/grouped.jsonl --report data/reports/pairs.json')
        print('     python ddh.py split --manifest data/reports/grouped.jsonl '
              '--output data/reports/split.jsonl --reviewed --seed 42')
        print('     python ddh.py train --stage window --manifest data/reports/split.jsonl --root data '
              '--output runs/window.pt --size 640 --batch 2 --epochs 40 --patience 8 --device cuda')
        print('  2) 同时推进两件事:')
        print(f'     - 还有 {n - have_w} 张缺 window，按 data/review/template_review_queue.txt 人工补')
        print('     - 继续收集人工裁切图，放进 data/crops_inbox/<来源>/ 后运行')
        print('       python tools/match_crops.py --cropped data/crops_inbox/张 --out data/crop_from_manual.jsonl')
        print('       python tools/build_manifest.py --apply')
    if have_s == 0:
        print('  注意: 没有 safe_box/quality 时，训练与原始评估能跑，calibrate 会明确报错（设计如此）。')
    print('=' * 76)


if __name__ == '__main__':
    main()
