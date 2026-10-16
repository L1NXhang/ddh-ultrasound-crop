"""Validate the three new-device task returns without approving training labels."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from audit_cleanup_returns import load_exports, verify_source_images, summaries, write_report, write_rows
from cleanup_returns import reconcile_records, validate_record

ROOT = Path(__file__).resolve().parents[1]


def registry(assignment):
    audit = json.loads((assignment / '分配审计.json').read_text(encoding='utf-8'))
    tasks = {}
    for package in audit['packages']:
        path = ROOT / package['manifest']
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(rows) != package['images'] or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Task manifest count or identities changed')
        tasks[package['task_id']] = {'reviewer': package['member'], 'rows': {r['id']: r for r in rows}}
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assignment', default='data/handout/新设备标注_20261016')
    parser.add_argument('--inbox', default='data/new_device_returns/20261016')
    parser.add_argument('--out', default='data/reports/new_device_returns_20261016')
    args = parser.parse_args()
    tasks = registry(ROOT / args.assignment)
    inbox, out = ROOT / args.inbox, ROOT / args.out
    inbox.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(inbox.glob('*.jsonl'))
    rows, rejected, seen = load_exports(paths, tasks)
    rows, conflicts, duplicates = reconcile_records(rows)
    rows, source_errors = verify_source_images(rows, tasks)
    counts, _ = summaries(rows, tasks, seen)
    queue = [{'task_id': row['task_id'], 'id': row['id'],
              'reasons': validate_record(row, tasks[row['task_id']]['rows'][row['id']])['review_reasons']}
             for row in rows if validate_record(row, tasks[row['task_id']]['rows'][row['id']])['review_reasons']]
    result = {'status': '等待成员回传' if not paths else '已校验，待负责人复核，不自动批准训练',
              'tasks': counts, 'shared': {'expected_shared': 0, 'both_returned_and_valid': 0, 'pairs': []},
              'rejected': rejected + source_errors, 'version_conflicts': conflicts,
              'identical_snapshot_duplicates': duplicates, 'review_queue': queue, 'comparisons': [],
              'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
              'approved_for_training': 0}
    write_rows(out / 'validated_proposals.jsonl', rows)
    write_report(out, result)
    print(json.dumps({'status': result['status'], 'valid': len(rows), 'invalid': len(result['rejected']),
                      'conflicts': len(conflicts), 'report': str(out / '报告.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
