"""把原始图片整理进项目：复制到 data/images/<source>/，并生成清单骨架。

只写确实已知的字段（来源分组、Graf 分型、编号），
window/crop/safe_box/quality 一律留 null，等人工标注，不伪造。

用法：python tools/prepare_dataset.py --source "<原始文件夹>" --out data
"""
import argparse
import json
import shutil
from pathlib import Path

SOURCES = ['余', '吴', '张', '阳', '胡']          # 每个文件夹 = 一个来源（设备/医生）
GRAF_DIRS = {'DDH-IIab': 'IIab', 'DDH-IIc': 'IIc', 'DDH-DIIIIV': 'III/IV'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', default='data')
    ap.add_argument('--force', action='store_true', help='允许覆盖已存在的清单（会丢失已标注内容）')
    args = ap.parse_args()
    src, out = Path(args.source), Path(args.out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    rows, copied = [], 0
    for source in SOURCES:
        d = src / source
        if not d.is_dir():
            print(f'跳过不存在的来源: {source}')
            continue
        (out / 'images' / source).mkdir(parents=True, exist_ok=True)
        # 胡 目录下按 Graf 分型分子文件夹，其余目录直接放图
        groups = ([p for p in d.iterdir() if p.is_dir()] if any(p.is_dir() for p in d.iterdir())
                  else [d])
        for g in sorted(groups):
            graf = GRAF_DIRS.get(g.name)
            for img in sorted(g.glob('*.jpg')):
                dest = out / 'images' / source / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)
                    copied += 1
                rows.append({
                    'id': img.stem,
                    'path': f'images/{source}/{img.name}',
                    'window': None,          # 待人工标注（每图必需）
                    'crop': None,            # 待人工标注
                    'safe_box': None,        # 待人工标注（老师定义）
                    'quality': None,         # 待人工评定
                    'style_group': '',       # 待人工核实界面外观组
                    'source_group': '',      # 同一序列/同一批次才填，文件夹不是，留空
                    'collect_group': source,  # 已知：采集来源文件夹（用于分组报告与来源外测）
                    'leak_group': '', 'split': '',
                    'graf_type': graf,       # 仅胡目录已知，其余 null，不推断
                    'human_modified': None,
                })
    rows.sort(key=lambda r: (r['collect_group'], r['id']))
    ids = [r['id'] for r in rows]
    assert len(set(ids)) == len(ids), '编号重复，id 必须唯一'
    p = out / 'annotations.todo.jsonl'
    if p.exists() and not args.force:
        raise SystemExit(f'{p} 已存在，为避免覆盖标注内容而中止。确需重建请加 --force')
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print(json.dumps({'复制图片': copied, '清单行数': len(rows), '清单': str(p),
                      '按来源': {s: sum(r['collect_group'] == s for r in rows) for s in SOURCES},
                      '有Graf分型': sum(r['graf_type'] is not None for r in rows)},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
