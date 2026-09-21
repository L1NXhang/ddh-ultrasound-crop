"""为新批次生成清单（无标签，window/crop 待补）。"""
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data' / 'images' / 'Ⅱc+Ⅱa-b'

rows = []
for p in sorted(BASE.rglob('*.jpg')):
    with Image.open(p) as im:
        w, h = im.size
    rows.append({'id': p.stem, 'path': p.relative_to(ROOT).as_posix(), 'window': None, 'crop': None,
                 'safe_box': None, 'quality': None, 'style_group': '', 'source_group': '',
                 'collect_group': p.parent.name, 'graf_type': None, 'human_modified': None,
                 'image_wh': [w, h]})
(ROOT / 'data' / 'new_batch.jsonl').write_text(
    ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
print(f'data/new_batch.jsonl：{len(rows)} 行')
