"""Create fingerprint-traceable annotation packages without changing source labels."""
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
OUT = ROOT / 'data/handout/新设备标注_20261016'
SEED = 20261016


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def source_path(row):
    path = Path(row['path'])
    if path.is_absolute():
        return path
    return ROOT / path if path.parts[0] == 'data' else ROOT / 'data' / path


def feature(row):
    path = source_path(row)
    with Image.open(path) as opened:
        image = opened.convert('RGB')
        pixel = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        window = row.get('window')
        valid = window and 0 <= window[0] < window[2] <= image.width and 0 <= window[1] < window[3] <= image.height
        region = image.crop(window) if valid else image
        vector = np.asarray(region.convert('L').resize((32, 32)), dtype=np.float32).ravel()
        vector = vector - vector.mean()
        vector = vector / max(float(np.linalg.norm(vector)), 1e-8)
        small = np.asarray(region.convert('L').resize((9, 8)), dtype=np.int16)
        bits = (small[:, 1:] > small[:, :-1]).ravel()
        return dict(row, pixel_sha256=pixel, image_wh=list(image.size), vector=vector, bits=bits)


def group_features(rows):
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    def union(a, b):
        parent[find(b)] = find(a)

    exact = {}
    for i, row in enumerate(rows):
        if row['pixel_sha256'] in exact:
            union(i, exact[row['pixel_sha256']])
        exact[row['pixel_sha256']] = i
    vectors, bits = np.stack([r['vector'] for r in rows]), np.stack([r['bits'] for r in rows])
    pairs = []
    for i in range(len(rows)):
        close = np.flatnonzero(np.count_nonzero(bits[i + 1:] != bits[i], axis=1) <= 4) + i + 1
        for j in close:
            correlation = float(vectors[i] @ vectors[j])
            if correlation >= .98:
                union(i, int(j))
                pairs.append({'a': i, 'b': int(j), 'correlation': correlation})
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values()), pairs


def balanced(groups):
    rng = random.Random(SEED)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda g: -len(g))
    buckets = [[], [], []]
    strata = [Counter(), Counter(), Counter()]
    for group in shuffled:
        key = (str(group[0].get('collect_group')), tuple(group[0]['image_wh']))
        target = min(range(3), key=lambda i: (len(buckets[i]), strata[i][key], i))
        buckets[target] = buckets[target] + group
        strata[target][key] += len(group)
    return buckets


def portable(row):
    return {key: value for key, value in row.items() if key not in ('vector', 'bits')}


GUIDE = '''新设备补充标注 - 请独立完成

本包有 {count} 张图片。先做前10张并导出给负责人检查口径，再继续；每10-20张导出备份。

一、推荐流程：沿用此前的边缘信息标注工具
1. 完整解压本包，用 Edge/Chrome 双击“开始标注.html”，选择本包 images 文件夹。
2. 按 W 框完整成像区 window：包住真实成像边界，不为去图标而缩小窗口。
3. 明确选择成像区内有/没有/不确定覆盖信息。没有也必须明确选择。
4. 有覆盖信息时按 E，选择类型后框完整文字块、婴儿/体位图标、标尺/灰度条；可画多个框。
5. 按 C 框最终建议裁剪范围 clean_box。上下保留需要的解剖结构，左右可向内收以去覆盖物。
   没有需要收缩的内容时，可点“沿用完整窗口”。不要为了去覆盖物明显切掉所需结构。
6. 看不清/图标深入中央/担心丢结构时，勾“需要负责人复核”，写一句原因，不强行做小框。
7. Enter 保存到浏览器缓存并下一张。“导出结果 JSONL”才生成需要回传的文件。
8. 完成后只交回最后一份完整 JSONL。保留本包和备份；不要改任务清单或重命名图片。

二、桌面裁剪软件也能打开本包 images
选择 images 作为图片目录，另选一个空的结果目录，使用“预测后人工审核”模式。
软件会记录人工确认的四边和模型来源；若这样操作，请回传整个结果目录的 ZIP。
注意：v0.5桌面软件没有文字/图标的分类框选功能，仅用它裁剪不能补齐覆盖物检测标签。
为完成本轮三种标签，请优先用包内“开始标注.html”；不要求重复做两套裁剪。

这不是训练测试集划分；成员之间按相似图组分工，减少重复。负责人合并审核后另行划分训练/验证。
本包不含预留测试图片；不显示算法预填框，以免把算法结果当作人工真值。
不要求判断DDH分型或临床安全；拿不准的结构保留要求交负责人和老师裁定。
任务ID：{task_id}；成员：{member}。
'''


def package(rows, number):
    label = f'成员{number}'
    folder = OUT / f'{label}_{len(rows)}张'
    (folder / 'images').mkdir(parents=True)
    tasks = []
    for i, row in enumerate(rows, 1):
        path = source_path(row)
        filename = f'{i:04d}{path.suffix.lower()}'
        shutil.copy2(path, folder / 'images' / filename)
        tasks.append({'id': 'ND26_' + row['pixel_sha256'][:24], 'filename': filename,
                      'original_path': str(path.relative_to(ROOT)), 'image_wh': row['image_wh'],
                      'pixel_sha256': row['pixel_sha256'], 'source_id': row['id'],
                      'assignment_group': row['assignment_group']})
    task_id = f'new_device_20261016_member{number}'
    payload = {'task_id': task_id, 'member': label, 'rows': tasks}
    html = (ROOT / 'product/边缘信息复核.html').read_text(encoding='utf-8')
    html = html.replace('__TASK_JSON__', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c'))
    html = html.replace("task.member+' · 40 张 · 独立完成'", "task.member+' · '+task.rows.length+' 张 · 独立完成'")
    (folder / '开始标注.html').write_text(html, encoding='utf-8')
    (folder / '操作说明.txt').write_text(GUIDE.format(count=len(rows), task_id=task_id, member=label), encoding='utf-8-sig')
    write_rows(folder / '任务清单.jsonl', tasks)
    archive = folder.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(folder))
    with zipfile.ZipFile(archive) as output:
        if output.testzip() is not None:
            raise ValueError('Archive verification failed')
    return {'member': label, 'task_id': task_id, 'images': len(rows), 'zip': str(archive.relative_to(ROOT)),
            'zip_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'manifest': str((folder / '任务清单.jsonl').relative_to(ROOT)),
            'collect_groups': dict(Counter(str(r.get('collect_group')) for r in rows)),
            'sizes': dict(Counter(str(r['image_wh']) for r in rows))}


def load_pool():
    source = read(ROOT / 'data/new_batch_windows.jsonl')
    new = [feature(dict(row, origin='new874', source_index=i)) for i, row in enumerate(source)]
    history = read(ROOT / 'data/experiments/window_corrected_20261013/manifest.jsonl')
    old = [feature(dict(row, origin='historical')) for row in history]
    annotations = [row for p in (ROOT / 'data/cleanup_returns').glob('*.jsonl') for row in read(p)]
    annotated_hashes = {r['pixel_sha256'] for r in annotations}
    for p in (ROOT.parent / 'DDH裁剪结果').rglob('reviewed_training_candidates.jsonl'):
        annotated_hashes.update(feature(dict(row, path=row['path']))['pixel_sha256'] for row in read(p))
    reserve = read(ROOT / 'data/handout/边缘信息复核_20261013_v2/预留复核组_暂不训练.jsonl')
    reserve_hashes = {r['pixel_sha256'] for r in reserve}
    legacy = [feature(dict(row, origin='legacy_inventory')) for row in read(ROOT / 'data/annotations.jsonl')]
    by_id = defaultdict(set)
    for row in new + legacy:
        by_id[row['id']].add(row['pixel_sha256'])
    conflict_ids = {key for key, hashes in by_id.items() if len(hashes) > 1}
    return source, new, old, annotated_hashes, reserve_hashes, conflict_ids


def classify_pool(new, old, annotated_hashes, reserve_hashes, conflict_ids):
    groups, pairs = group_features(new + old)
    states, candidates, withheld = {}, [], []
    seen_pixels = set()
    for i, group in enumerate(groups):
        current = [dict(r, assignment_group=f'NDG{i:04d}') for r in group if r['origin'] == 'new874']
        if not current:
            continue
        existing = [r for r in group if r['origin'] == 'historical']
        protected = any(r.get('split') in {'val', 'calib', 'test'} for r in existing)
        reserved = any(r['pixel_sha256'] in reserve_hashes for r in group)
        trained = any(r.get('split') == 'train' for r in existing)
        unique = []
        for row in current:
            if row['id'] in conflict_ids:
                status = 'id_conflict_quarantine'
            elif protected:
                status = 'historical_holdout_or_near_candidate'
            elif reserved:
                status = 'existing_reserved_group'
            elif row['pixel_sha256'] in annotated_hashes:
                status = 'already_annotated_pending_review'
            elif trained:
                status = 'historical_train_or_near_candidate'
            elif row['pixel_sha256'] in seen_pixels:
                status = 'exact_duplicate'
            else:
                status = 'candidate'
                unique.append(row)
            states[row['source_index']] = status
            if status != 'candidate':
                withheld.append(dict(portable(row), disposition=status))
            seen_pixels.add(row['pixel_sha256'])
        if unique:
            candidates.append(unique)
    return states, candidates, withheld, pairs


def reserve_groups(candidates, states, withheld):
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    reserve_target = 100
    held_count = sum(v in {'existing_reserved_group', 'historical_holdout_or_near_candidate'} for v in states.values())
    reserved_new = []
    while candidates and held_count + len(reserved_new) < reserve_target:
        reserved_new.extend(candidates.pop())
    for row in reserved_new:
        states[row['source_index']] = 'new_reserved_group'
        withheld.append(dict(portable(row), disposition='new_reserved_group'))
    return candidates, states, withheld, reserve_target


def verify_assignment(source, states, buckets, annotated_hashes, reserve_hashes):
    assert len(states) == len(source) == 874
    assigned_hashes = [{r['pixel_sha256'] for r in bucket} for bucket in buckets]
    assigned_groups = [{r['assignment_group'] for r in bucket} for bucket in buckets]
    for i in range(3):
        assert len(assigned_hashes[i]) == len(buckets[i])
        for j in range(i):
            assert assigned_hashes[i].isdisjoint(assigned_hashes[j])
            assert assigned_groups[i].isdisjoint(assigned_groups[j])
    assert set.union(*assigned_hashes).isdisjoint(annotated_hashes | reserve_hashes)


def main():
    if OUT.exists():
        raise ValueError('Assignment already exists; inspect it rather than overwriting')
    OUT.mkdir(parents=True)
    source, new, old, annotated_hashes, reserve_hashes, conflict_ids = load_pool()
    states, candidates, withheld, pairs = classify_pool(new, old, annotated_hashes, reserve_hashes, conflict_ids)
    candidates, states, withheld, reserve_target = reserve_groups(candidates, states, withheld)
    buckets = balanced(candidates)
    for bucket in buckets:
        for row in bucket:
            states[row['source_index']] = 'assigned_for_annotation'
    packages = [package(bucket, i + 1) for i, bucket in enumerate(buckets)]
    audit = {'date': '2026-10-16', 'seed': SEED, 'source_rows': len(source), 'dispositions': dict(Counter(states.values())),
             'assigned_images': sum(map(len, buckets)), 'packages': packages, 'new_reserve_target_including_old': reserve_target,
             'similarity_candidate_pairs': len(pairs), 'near_hash_distance': 4, 'near_correlation': .98,
             'limitations': ['Similarity groups are not patient identities.', 'Pre-reserved groups are kept private.',
                            'Algorithm-prefilled review_new records are not counted as human annotations.',
                            'Training/validation splits will be made after label review; member split is only work allocation.']}
    verify_assignment(source, states, buckets, annotated_hashes, reserve_hashes)
    write_rows(OUT / '暂不分配_逐图原因.jsonl', withheld)
    write_rows(OUT / '预留评估_暂不标注不训练.jsonl', [r for r in withheld if 'reserved' in r['disposition']])
    write_rows(OUT / '全部原始记录_去向.jsonl', [dict(original_path=r['path'], source_index=r['source_index'],
               pixel_sha256=r['pixel_sha256'], disposition=states[r['source_index']]) for r in new])
    (OUT / '分配审计.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
