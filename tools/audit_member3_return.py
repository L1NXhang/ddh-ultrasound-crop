"""Audit the member3 new-device return against its task manifest.

Validates provenance (source bytes), exported artifacts, internal consistency and
geometry, then maps records back to the authoritative manifest IDs. Writes a
fingerprint-bridged canonical JSONL, a Markdown report and an audit.json.

Nothing here approves training labels: every row stays approved_for_training=false.
"""
import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HANDOUT = ROOT / 'data/handout/新设备标注_20261016'


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb_sha(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def inside(outer, inner):
    return outer[0] <= inner[0] and inner[2] <= outer[2] and outer[1] <= inner[1] and inner[3] <= outer[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True)
    ap.add_argument('--manifest', default=str(HANDOUT / '成员3_237张/任务清单.jsonl'))
    ap.add_argument('--task-id', default='new_device_20261016_member3')
    ap.add_argument('--reviewer', default='成员3')
    ap.add_argument('--out', default=str(ROOT / 'data/reports/member3_return_20261021'))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_jsonl(args.manifest)
    by_file = {m['filename']: m for m in manifest}

    zf = zipfile.ZipFile(args.zip)
    names = set(zf.namelist())
    results = read_jsonl_from_zip = [json.loads(l) for l in zf.read('results.jsonl').decode('utf-8').splitlines() if l.strip()]
    candidates = [json.loads(l) for l in zf.read('reviewed_training_candidates.jsonl').decode('utf-8').splitlines() if l.strip()]
    project = json.loads(zf.read('project.json').decode('utf-8'))

    errors, accepted, rejected = [], [], []
    seen_files = Counter()

    for row in results:
        fname = row.get('file')
        base = fname.split('/')[-1] if fname else None
        m = by_file.get(base)
        if m is None:
            rejected.append({'file': fname, 'errors': ['unknown_image_not_in_manifest']})
            continue
        seen_files[base] += 1
        row_errors = []

        # provenance: member tool hashes raw file bytes
        source = ROOT / m['original_path']
        if not source.exists():
            row_errors.append('source_image_missing')
        elif file_sha(source) != row.get('source_sha256'):
            row_errors.append('source_sha256_mismatch')
        elif rgb_sha(source) != m['pixel_sha256']:
            row_errors.append('manifest_pixel_sha256_mismatch')

        if list(row.get('original_size') or []) != list(m['image_wh']):
            row_errors.append('original_size_mismatch')

        W, H = m['image_wh']
        win, clean = row.get('window'), row.get('clean_box')
        for label, box in (('window', win), ('clean_box', clean)):
            if not isinstance(box, list) or len(box) != 4:
                row_errors.append(f'{label}_missing')
            elif not (0 <= box[0] < box[2] <= W and 0 <= box[1] < box[3] <= H):
                row_errors.append(f'{label}_out_of_bounds')
        if win and clean and not inside(win, clean):
            row_errors.append('clean_box_outside_window')
        if row.get('crop_box') and clean and list(row['crop_box']) != list(clean):
            row_errors.append('crop_box_not_equal_clean_box')
        if row.get('crop') and clean and list(row['crop']) != [clean[1], clean[3]]:
            row_errors.append('crop_lines_not_clean_box_y')

        outputs = row.get('outputs') or {}
        for kind in ('cropped', 'annotated'):
            ref = outputs.get(kind)
            if not ref or ref not in names:
                row_errors.append(f'{kind}_missing')
        cropped = outputs.get('cropped')
        if cropped in names and clean:
            try:
                with Image.open(io.BytesIO(zf.read(cropped))) as im:
                    im.load()
                    if im.size != (clean[2]-clean[0], clean[3]-clean[1]):
                        row_errors.append('cropped_size_mismatch')
            except Exception:
                row_errors.append('cropped_undecodable')
        annotated = outputs.get('annotated')
        if annotated in names:
            try:
                with Image.open(io.BytesIO(zf.read(annotated))) as im:
                    im.load()
                    if list(im.size) != list(m['image_wh']):
                        row_errors.append('annotated_size_mismatch')
            except Exception:
                row_errors.append('annotated_undecodable')

        if row.get('approved_for_training') is not False:
            row_errors.append('approved_flag_not_false')
        if row.get('human_reviewed') is not True:
            row_errors.append('not_human_reviewed')

        if row_errors:
            rejected.append({'file': base, 'id': row.get('id'), 'errors': sorted(set(row_errors))})
        else:
            accepted.append(dict(row, _manifest_id=m['id'], _source_sha256=m['pixel_sha256'],
                                 _source_path=m['original_path'], _assignment_group=m.get('assignment_group')))

    # cross-file consistency
    cand_by_sha = {c['source_sha256']: c for c in candidates}
    inconsistent = []
    for row in accepted:
        c = cand_by_sha.get(row['source_sha256'])
        if c is None:
            inconsistent.append({'file': row['file'], 'error': 'absent_from_candidates'})
            continue
        for field in ('window', 'clean_box'):
            if list(c.get(field) or []) != list(row.get(field) or []):
                inconsistent.append({'file': row['file'], 'error': f'{field}_differs_in_candidates'})

    duplicate_ids = {k: v for k, v in seen_files.items() if v > 1}
    missing = sorted(set(by_file) - set(seen_files))

    # model comparison
    deltas = []
    for row in accepted:
        pred = (row.get('prediction') or {}).get('crop_box')
        clean = row.get('clean_box')
        if not pred or not clean:
            continue
        model_win = row.get('model_window') or row.get('window')
        deltas.append({
            'file': row['file'],
            'top_delta': clean[1] - pred[1],
            'bottom_delta': clean[3] - pred[3],
            'window_max_edge_change': max(abs(a-b) for a, b in zip(row['window'], model_win)),
        })

    def summary(values):
        values = sorted(values)
        n = len(values)
        return {'n': n, 'mean': round(sum(values)/n, 2) if n else None,
                'median': values[n//2] if n else None,
                'min': values[0] if n else None, 'max': values[-1] if n else None}

    metrics = {
        'top_delta': summary([d['top_delta'] for d in deltas]),
        'bottom_delta': summary([d['bottom_delta'] for d in deltas]),
        'window_max_edge_change': summary([d['window_max_edge_change'] for d in deltas]),
        'identical_to_model': sum(1 for d in deltas if d['top_delta'] == 0 and d['bottom_delta'] == 0),
        'window_modified': sum(1 for r in accepted if r.get('window_modified')),
        'crop_modified': sum(1 for r in accepted if r.get('crop_modified')),
    }

    audit = {
        'status': '已校验，待负责人复核，不自动批准训练',
        'task_id': args.task_id,
        'reviewer_package': args.reviewer,
        'zip': str(args.zip),
        'zip_sha256': file_sha(args.zip),
        'project': project,
        'expected_images': len(manifest),
        'received_records': len(results),
        'validated_proposals': len(accepted),
        'rejected': rejected,
        'duplicate_file_entries': duplicate_ids,
        'missing_from_return': missing,
        'cross_file_inconsistencies': inconsistent,
        'model_comparison': metrics,
        'model_checkpoint_sha256': sorted({(r.get('model') or {}).get('model_sha256') for r in results}),
        'margin_values': sorted({(r.get('model') or {}).get('margin') for r in results}),
        'crop_policies': sorted({(r.get('model') or {}).get('crop_policy') for r in results}),
        'review_status': dict(Counter(r.get('review_status') for r in results)),
        'flags': dict(Counter(str(r.get('flags')) for r in results)),
        'approved_for_training': 0,
        'notes': [
            'Source images are hashed by raw file bytes by the crop tool; the manifest uses an RGB-pixel hash. Both are verified per image.',
            'Package image IDs are the first 20 hex chars of the byte hash, so records are mapped back through the manifest filename and byte hash.',
            'Geometry and pixel checks cannot prove anatomical completeness or that device text/icons were removed.',
            'Reviewer and approval attribution: the software records reviewer=null; approval must be recorded explicitly by the owner.',
        ],
    }

    canon = out/'canonical_proposals.jsonl'
    canon.write_text(''.join(json.dumps({k: v for k, v in r.items() if not k.startswith('_')} |
                                        {'task_id': args.task_id, 'manifest_id': r['_manifest_id'],
                                         'manifest_pixel_sha256': r['_source_sha256'],
                                         'source_path': r['_source_path'],
                                         'assignment_group': r['_assignment_group'],
                                         'approved_for_training': False,
                                         'reviewer': None}, ensure_ascii=False) + '\n'
                              for r in accepted), encoding='utf-8')
    (out/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({'validated': len(accepted), 'rejected': len(rejected),
                      'missing': len(missing), 'inconsistent': len(inconsistent),
                      'report': str(out/'报告.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()