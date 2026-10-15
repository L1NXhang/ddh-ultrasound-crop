"""Prepare local blind-review packages; never import preview clicks as labels."""
import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/handout/边缘信息复核_20261013_v2'


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def assign_groups(groups, exclusive=30, common=10, reserve=40):
    if len(groups) < 2*exclusive+common+reserve:
        raise ValueError('Not enough independent candidate groups')
    hold = groups[:reserve]
    shared = [g[0] for g in groups[reserve:reserve+common]]
    start = reserve+common
    return {'A': shared+[g[0] for g in groups[start:start+exclusive]],
            'B': shared+[g[0] for g in groups[start+exclusive:start+exclusive*2]],
            'reserved': [r for g in hold for r in g], 'shared_ids': [r['id'] for r in shared]}


def deduplicate_candidates(rows):
    byid = defaultdict(list)
    for row in rows:
        byid[row['id']].append(row)
    conflicts = [r for group in byid.values() if len({r['pixel_sha256'] for r in group}) > 1 for r in group]
    conflict_ids = {r['id'] for r in conflicts}
    return [group[0] for rid, group in byid.items() if rid not in conflict_ids], conflicts


def inventory():
    paths = [ROOT/'data/new_batch_windows.jsonl', ROOT/'data/分配20261009.jsonl']
    rows = [dict(r, batch='new874' if i == 0 else 'assignment429')
            for i, path in enumerate(paths) for r in read(path)]
    candidates = []
    for r in rows:
        with Image.open(ROOT/r['path']) as im:
            im = im.convert('RGB')
            digest = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
            feature = im.crop(tuple(r['window'])) if r.get('window') else im
            a = np.array(feature.convert('L').resize((32, 32)), dtype=np.float32).ravel()
            a = a-a.mean()
            vector = a/max(float(np.linalg.norm(a)), 1e-8)
            small = np.array(feature.convert('L').resize((9, 8)), dtype=np.int16)
            bits = (small[:, 1:] > small[:, :-1]).ravel()
        candidates.append(dict(r, pixel_sha256=digest, vector=vector, bits=bits,
                               already_reviewed=r.get('crop_source') == 'human_tool'))
    clean, conflicts = deduplicate_candidates(candidates)
    legacy = {r['id']: r for r in read(ROOT/'data/annotations.jsonl')}
    filtered = []
    for row in clean:
        if row['id'] in legacy:
            legacy_path = ROOT/'data'/legacy[row['id']]['path']
            with Image.open(legacy_path) as im:
                im = im.convert('RGB')
                old_digest = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
            if row['pixel_sha256'] != old_digest:
                conflicts.append(dict(row, conflict_reason='id_reused_in_legacy_dataset',
                                      legacy_path=str(legacy_path.relative_to(ROOT))))
                continue
            row = dict(row, already_reviewed=True)
        filtered.append(row)
    return filtered, conflicts


def similarity_groups(rows):
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    def union(a, b):
        parent[find(b)] = find(a)

    exact, byid = {}, {r['id']: i for i, r in enumerate(rows)}
    for i, r in enumerate(rows):
        if r['pixel_sha256'] in exact:
            union(i, exact[r['pixel_sha256']])
        exact[r['pixel_sha256']] = i
    known = json.loads((ROOT/'data/reports/分配20261009_近重复.json').read_text(encoding='utf-8'))['within']
    for pair in known:
        if pair['a'] in byid and pair['b'] in byid:
            union(byid[pair['a']], byid[pair['b']])
    vectors = np.stack([r['vector'] for r in rows])
    bits = np.stack([r['bits'] for r in rows])
    for i in range(len(rows)):
        for j in np.flatnonzero(np.count_nonzero(bits[i+1:] != bits[i], axis=1) <= 4)+i+1:
            if float(vectors[i] @ vectors[j]) >= .98:
                union(i, int(j))
    grouped = defaultdict(list)
    for i, r in enumerate(rows):
        grouped[find(i)].append({k: v for k, v in r.items() if k not in ('vector', 'bits')})
    return list(grouped.values())


def ordered_groups(groups):
    rng = random.Random(20261013)
    buckets = defaultdict(list)
    for group in groups:
        group = sorted(group, key=lambda r: (not bool(r.get('window_flags')), r['id']))
        r = group[0]
        key = (r['batch'], tuple(r['image_wh']), bool(r.get('window_flags')))
        buckets[key].append(group)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    ordered = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                ordered.append(buckets[key].pop())
    return ordered


README = '''边缘信息复核：每人40张，预计先用30–60分钟熟悉并完成一轮。
1. 完整解压压缩包，双击「开始复核.html」，选择包内的 images 文件夹。
2. 按 W，从头框出完整成像窗口。不要为了去文字或图标而缩窄 window。
3. 选择“有/无/不确定”覆盖信息。有的话，选类型后按 E，框出每块文字、图标或标尺。
   同一块设备参数合成一个框，无需逐个字框；框完整覆盖物，不沿身体结构画框。
4. 按 C 画建议的最终干净裁剪框；没有覆盖信息可以点“沿用完整窗口”。
   这一轮先研究左右收缩，上下不强制套固定像素。此框只是建议，不等于解剖安全确认。
5. 拿不准、覆盖物深入中央、担心丢结构：勾“需要负责人复核”，写一句原因。
6. Enter存到本机浏览器缓存并下一张；完成后点“导出结果”，只回传导出的 JSONL。
   每10张导出一次备份。不要只关网页就认为已经交回。
7. 请独立完成，不与另一成员对答案。包内有10张双人重复图片，用来发现口径差异。

全过程在本机运行，不上传。原图不改动。没有模型预填框，避免照着算法接受形成循环真值。
不要求判断DDH分型、临床可用性或解剖安全；这些由负责人/老师最终确认。
'''


def package(rows, member):
    folder = OUT/f'成员{member}_试标40张'
    (folder/'images').mkdir(parents=True, exist_ok=True)
    tasks = []
    for i, row in enumerate(rows, 1):
        source = ROOT/row['path']
        name = f'{i:03d}'+source.suffix.lower()
        shutil.copy2(source, folder/'images'/name)
        tasks.append({'id': row['id'], 'filename': name, 'original_path': row['path'],
                      'image_wh': row['image_wh'], 'pixel_sha256': row['pixel_sha256']})
    payload = {'task_id': f'cleanup_pilot_20261013_member{member}', 'member': f'成员{member}', 'rows': tasks}
    html = (ROOT/'product/边缘信息复核.html').read_text(encoding='utf-8')
    html = html.replace('__TASK_JSON__', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c'))
    (folder/'开始复核.html').write_text(html, encoding='utf-8')
    (folder/'说明.txt').write_text(README, encoding='utf-8-sig')
    write_rows(folder/'任务清单.jsonl', tasks)
    archive = OUT/(folder.name+'.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in folder.rglob('*'):
            if path.is_file():
                z.write(path, path.relative_to(folder))
    return {'member': member, 'count': len(tasks), 'zip': str(archive.relative_to(ROOT)),
            'mb': round(archive.stat().st_size/1024**2, 2)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'分配审计.json').exists():
        raise ValueError('This assignment exists; choose a new version before regenerating')
    rows, conflicts = inventory()
    groups = [g for g in similarity_groups(rows) if not any(r['already_reviewed'] for r in g)]
    groups = ordered_groups(groups)
    assignment = assign_groups(groups)
    packages = [package(assignment['A'], 1), package(assignment['B'], 2)]
    write_rows(OUT/'预留复核组_暂不训练.jsonl', assignment['reserved'])
    conflict_records = [{k: v for k, v in r.items() if k not in ('vector', 'bits')} for r in conflicts]
    (OUT/'ID冲突_暂不分配.json').write_text(json.dumps(conflict_records, ensure_ascii=False, indent=2), encoding='utf-8')
    audit = {'seed': 20261013, 'candidate_images': sum(len(g) for g in groups), 'candidate_groups': len(groups),
             'quarantined_id_conflict_images': len(conflicts),
             'packages': packages, 'shared_ids': assignment['shared_ids'],
             'reserved_images': len(assignment['reserved']), 'reserved_groups': 40,
             'assigned_unique': len({r['id'] for r in assignment['A']+assignment['B']}),
             'assigned_batches': dict(Counter(r['batch'] for r in assignment['A']+assignment['B'])),
             'limitations': ['Similarity groups are candidates, not patient IDs; manual pair review still needed.',
                             'No preview session results used.', 'Reserved images not supplied to members.',
                             'Cleanup boxes are proposals; anatomy preservation requires owner/expert confirmation.']}
    (OUT/'分配审计.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
