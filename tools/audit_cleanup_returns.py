"""Audit local member exports. All validated rows remain unapproved proposals."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from cleanup_returns import validate_record, reconcile_records, compare_pair

ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT = ROOT/'data/handout/边缘信息复核_20261013_v2'


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows), encoding='utf-8')


def registry(assignment):
    tasks = {}
    for member in (1, 2):
        rows = read(assignment/f'成员{member}_试标40张/任务清单.jsonl')
        tasks[f'cleanup_pilot_20261013_member{member}'] = {
            'reviewer': f'成员{member}', 'rows': {r['id']: r for r in rows}}
    return tasks


def load_exports(paths, tasks):
    valid, rejected, seen = [], [], defaultdict(set)
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            context = {'file': str(path), 'line': line_no}
            try:
                row = json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
                if not isinstance(row, dict):
                    raise ValueError('record_must_be_object')
                task = tasks.get(row.get('task_id'))
                if task is None or row.get('id') not in task['rows']:
                    raise ValueError('unknown_task_or_image')
                seen[row['task_id']].add(row['id'])
                result = validate_record(row, task['rows'][row['id']])
                if row.get('reviewer') != task['reviewer']:
                    result['errors'].append('reviewer_mismatch')
                if result['errors']:
                    rejected.append(dict(context, id=row['id'], task_id=row['task_id'], **result))
                else:
                    valid.append(dict(row, _return_file=str(path), _line=line_no, _source_sha256=digest))
            except (ValueError, TypeError, KeyError) as error:
                rejected.append(dict(context, errors=[str(error)]))
    return valid, rejected, seen


def verify_source_images(rows, tasks):
    errors, cache = [], {}
    for row in rows:
        ref = tasks[row['task_id']]['rows'][row['id']]
        path = ROOT/ref['original_path']
        if path not in cache:
            try:
                with Image.open(path) as im:
                    im = im.convert('RGB')
                    cache[path] = (list(im.size), hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest())
            except (OSError, ValueError) as error:
                cache[path] = str(error)
        result = cache[path]
        if not isinstance(result, tuple) or result != (ref['image_wh'], ref['pixel_sha256']):
            errors.append({'task_id': row['task_id'], 'id': row['id'], 'errors': ['source_image_unavailable_or_changed']})
    invalid = {(r['task_id'], r['id']) for r in errors}
    return [r for r in rows if (r['task_id'], r['id']) not in invalid], errors


def summaries(rows, tasks, seen):
    by_task = defaultdict(dict)
    for row in rows:
        by_task[row['task_id']][row['id']] = row
    summary = {tid: {'expected': len(task['rows']), 'received_ids': len(seen[tid]),
                     'validated_proposals': len(by_task[tid]),
                     'missing_or_invalid_ids': sorted(set(task['rows'])-set(by_task[tid]))}
               for tid, task in tasks.items()}
    return summary, by_task


def paired_comparisons(by_task, tasks, args):
    ids = sorted(tasks)
    common = set(tasks[ids[0]]['rows']) & set(tasks[ids[1]]['rows'])
    ready = common & set(by_task[ids[0]]) & set(by_task[ids[1]])
    pairs = [compare_pair(by_task[ids[0]][rid], by_task[ids[1]][rid], args.window_tolerance,
                          args.clean_tolerance, args.overlay_iou) for rid in sorted(ready)]
    return {'expected_shared': len(common), 'both_returned_and_valid': len(ready),
            'pending_shared_ids': sorted(common-ready), 'pairs': pairs}


def render_comparison(row_a, row_b, path):
    image = Image.open(ROOT/row_a['original_path']).convert('RGB')
    scale = min(1, 850/image.width)
    size = (round(image.width*scale), round(image.height*scale))
    sheet = Image.new('RGB', (size[0]*2, size[1]+36), '#111a29')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate((row_a, row_b)):
        x = i*size[0]
        sheet.paste(image.resize(size), (x, 36))
        draw.text((x+10, 8), f'Member {i+1}: cyan=window green=crop yellow=overlay', fill='white')
        boxes = [(row.get('window'), '#60cfff'), (row.get('clean_box'), '#59e4b2')]
        boxes += [(o['box'], '#ffc36b') for o in row['overlays']]
        for box, color in boxes:
            if box:
                draw.rectangle([x+box[0]*scale, 36+box[1]*scale, x+box[2]*scale, 36+box[3]*scale], outline=color, width=2)
    sheet.save(path)


def write_report(out, audit):
    counts = audit['tasks']
    lines = ['# 成员回传校验', '', '状态：'+audit['status'], '',
             '| 成员任务 | 应回传 | 收到不同ID | 校验通过的建议 |', '|---|---:|---:|---:|']
    for tid, result in counts.items():
        lines.append(f'| {tid} | {result["expected"]} | {result["received_ids"]} | {result["validated_proposals"]} |')
    shared = audit['shared']
    lines += ['', f'共同图片：应有 {shared["expected_shared"]} 张，双方有效回传 {shared["both_returned_and_valid"]} 张。',
              f'无效记录 {len(audit["rejected"])} 条；同人不同版本冲突 {len(audit["version_conflicts"])} 组。',
              f'需负责人查看的记录 {len(audit["review_queue"])} 条。', '',
              '校验通过仅代表格式、坐标与来源一致；所有记录仍是未批准建议，不自动进入训练。',
              '双人分歧阈值用于安排复核优先级，不是临床合格标准。', '',
              '将两个成员导出的最终 JSONL 放入 data/cleanup_returns，再运行 tools/audit_cleanup_returns.py。',
              '保留原始导出文件。若同时放入旧备份且坐标变化，会隔离冲突，不自动选最后一版。']
    if audit['review_queue']:
        lines += ['', '## 待复核记录', '', '| 图片ID | 任务或类型 | 原因 |', '|---|---|---|']
        for row in audit['review_queue']:
            lines.append(f'| {row["id"]} | {row.get("task_id", row.get("type", ""))} | {", ".join(row["reasons"])} |')
    if audit.get('comparisons'):
        lines += ['', '## 双人对照图', '']
        lines += [f'- [{item["id"]}]({item["filename"]})' for item in audit['comparisons']]
    (out/'报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (out/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def audit_returns(args):
    if min(args.window_tolerance, args.clean_tolerance) < 0 or not 0 < args.overlay_iou <= 1:
        raise ValueError('复核边差阈值必须非负，overlay IoU 必须位于 (0,1]')
    tasks = registry(ROOT/args.assignment)
    inbox, out = ROOT/args.inbox, ROOT/args.out
    inbox.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    paths = [ROOT/Path(p) for p in args.inputs] if args.inputs else sorted(inbox.glob('*.jsonl'))
    rows, rejected, seen = load_exports(paths, tasks)
    rows, conflicts, duplicates = reconcile_records(rows)
    rows, source_errors = verify_source_images(rows, tasks)
    task_summary, by_task = summaries(rows, tasks, seen)
    shared = paired_comparisons(by_task, tasks, args)
    queue = [dict(task_id=r['task_id'], id=r['id'], reasons=validate_record(r, tasks[r['task_id']]['rows'][r['id']])['review_reasons'])
             for r in rows if validate_record(r, tasks[r['task_id']]['rows'][r['id']])['review_reasons']]
    queue += [{'id': p['id'], 'reasons': p['reasons'], 'type': 'member_disagreement'}
              for p in shared['pairs'] if p['disagreement']]
    result = {'status': '等待成员回传' if not paths else '回传校验完成，待负责人裁定',
              'tasks': task_summary, 'shared': shared, 'rejected': rejected+source_errors,
              'version_conflicts': conflicts, 'identical_snapshot_duplicates': duplicates, 'review_queue': queue,
              'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
              'comparisons': [], 'parameters': {'window_tolerance_px': args.window_tolerance, 'clean_tolerance_px': args.clean_tolerance,
                             'overlay_iou': args.overlay_iou}, 'approved_for_training': 0}
    if paths and any(r['missing_or_invalid_ids'] for r in task_summary.values()):
        result['status'] = '部分回传已校验，仍有缺失、无效或冲突记录'
    if args.render:
        ids = sorted(tasks)
        for i, pair in enumerate(shared['pairs'], 1):
            if pair['disagreement']:
                filename = f'对照_{i:02d}.png'
                render_comparison(by_task[ids[0]][pair['id']], by_task[ids[1]][pair['id']], out/filename)
                result['comparisons'].append({'id': pair['id'], 'filename': filename})
    write_rows(out/'validated_proposals.jsonl', rows)
    write_report(out, result)
    print(json.dumps({'status': result['status'], 'validated': len(rows), 'rejected': len(result['rejected']),
                      'version_conflicts': len(conflicts), 'report': str(out/'报告.md')}, ensure_ascii=False))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--assignment', default='data/handout/边缘信息复核_20261013_v2')
    p.add_argument('--inbox', default='data/cleanup_returns')
    p.add_argument('--out', default='data/reports/cleanup_returns_20261014')
    p.add_argument('--inputs', nargs='*')
    p.add_argument('--window-tolerance', type=int, default=8)
    p.add_argument('--clean-tolerance', type=int, default=16)
    p.add_argument('--overlay-iou', type=float, default=.5)
    p.add_argument('--render', action='store_true')
    return p


if __name__ == '__main__':
    cli = parser()
    try:
        audit_returns(cli.parse_args())
    except (OSError, ValueError) as error:
        cli.error(str(error))
