"""把所有已核实的建议合并成可训练清单 data/annotations.jsonl。

只合并满足下列条件的内容，其余字段保持 null（宁缺勿造）：
  window  ← window_templates.json 里 confirmed=true 的模板，且该图在模板适用列表内
  crop    ← crop_from_manual.jsonl 里 status=ok 的反推结果（人工裁剪图匹配）
  graf_type / source_group ← 清单骨架里已有的已知字段
未覆盖的图会在报告里列出，等人工补。

用法：python tools/build_manifest.py            # 预演，只报告不写文件
      python tools/build_manifest.py --apply    # 真正写出 data/annotations.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='data/annotations.todo.jsonl')
    ap.add_argument('--templates', default='data/window_templates.json')
    ap.add_argument('--prefill', default='data/window_prefill.jsonl')
    ap.add_argument('--crops', default='data/crop_from_manual.jsonl')
    ap.add_argument('--out', default='data/annotations.jsonl')
    ap.add_argument('--subset', default='',
                    help='只保留这些字段都非空的图，逗号分隔，例如 window,crop。'
                         '用于先用一小批齐备的图跑通管线（ddh.py 会因任何一行缺字段而整条失败）')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    rows = load_jsonl(args.base)
    byid = {r['id']: r for r in rows}

    # window：只有 confirmed=true 的模板才允许套用。
    # 用模板自带的 applicable_ids（由框外条带判据得出），不能用“检测框是否贴近模板”
    # ——逐图检测本身不稳，那会漏掉大量本该套用的图。
    tpl = json.loads(Path(args.templates).read_text(encoding='utf-8'))
    win_applied, win_pending = 0, []
    for wh, t in tpl.get('templates', {}).items():
        if not t.get('confirmed'):
            continue
        for rid in t.get('applicable_ids', []):
            r = byid.get(rid)
            if r is not None and r.get('window') is None:
                r['window'] = list(t['template_xyxy'])
                win_applied += 1
    for r in rows:
        if r.get('window') is None:
            win_pending.append(r['id'])

    # crop：反推成功且判为 ok 的
    crops = {x['id']: x for x in load_jsonl(args.crops)} if Path(args.crops).exists() else {}
    crop_applied, crop_bad = 0, []
    for rid, x in crops.items():
        r = byid.get(rid)
        if r is None:
            continue
        if x.get('status') == 'ok':
            r['crop'] = x['crop']
            crop_applied += 1
        else:
            crop_bad.append((rid, x.get('status')))

    if args.subset:
        need = [f.strip() for f in args.subset.split(',') if f.strip()]
        kept = [r for r in rows if all(r.get(f) is not None for f in need)]
        print(f'--subset {" ".join(need)}：保留 {len(kept)}/{len(rows)} 行；'
              f'剔除的 {len(rows) - len(kept)} 行未写进输出（原清单不动）')
        rows = kept

    have_win = sum(r.get('window') is not None for r in rows)
    have_crop = sum(r.get('crop') is not None for r in rows)
    have_safe = sum(r.get('safe_box') is not None for r in rows)
    have_q = sum(r.get('quality') is not None for r in rows)
    report = {
        '总图数': len(rows),
        'window 已有': have_win, 'window 待补': len(rows) - have_win,
        'crop 已有': have_crop, 'crop 待补': len(rows) - have_crop,
        'safe_box 已有': have_safe, 'quality 已有': have_q,
        '本次套用窗口模板': win_applied, '本次套用裁切反推': crop_applied,
        '未确认模板(需人工改成 confirmed:true)': [k for k, v in tpl.get('templates', {}).items()
                                                  if not v.get('confirmed')],
        '裁切反推但未采用': crop_bad[:5],
        '可以开始训练的前提': 'window 与 crop 齐备后才能跑步骤B；quality 与 safe_box 决定校准与安全头是否可用',
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.apply:
        if win_pending:
            print(f'警告：仍有 {len(win_pending)} 张没有 window，这些行会保持 null，'
                  f'train 会因为字段缺失而拒绝整份清单。')
        Path(args.out).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows),
                                  encoding='utf-8')
        print('已写出', args.out)
    else:
        print('（预演，未写文件；确认无误后加 --apply）')


if __name__ == '__main__':
    main()
