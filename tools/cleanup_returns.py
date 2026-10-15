"""Validate member proposals and compare independent annotations, without approval."""
import json
from collections import defaultdict

from crop_geometry import validate_window

SCHEMA = 'ddh_edge_cleanup_review/v1'
KINDS = {'device_text', 'body_marker', 'scale', 'other'}


def contains(a, b):
    return a[0] <= b[0] < b[2] <= a[2] and a[1] <= b[1] < b[3] <= a[3]


def intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def iou(a, b):
    area = lambda box: (box[2]-box[0])*(box[3]-box[1])
    common = intersection(a, b)
    return common/max(1, area(a)+area(b)-common)


def checked_box(value, size, name, errors, optional=False):
    if optional and value is None:
        return None
    try:
        return validate_window(value, size)
    except (ValueError, TypeError):
        errors.append(name+'_invalid')
        return None


def check_metadata(row, reference, errors):
    for key in ('id', 'filename', 'original_path', 'pixel_sha256', 'image_wh'):
        if row.get(key) != reference.get(key):
            errors.append(key+'_mismatch')
    if row.get('original_size') != reference['image_wh']:
        errors.append('original_size_mismatch')
    for key, value in [('schema', SCHEMA), ('coordinate_space', 'original_pixels'),
                       ('window_source', 'human_drawn')]:
        if row.get(key) != value:
            errors.append(key+'_invalid')
    for key in ('approved_for_training', 'anatomy_verified'):
        if row.get(key) is not False:
            errors.append(key+'_must_be_false')
    if type(row.get('needs_review')) is not bool:
        errors.append('needs_review_must_be_boolean')
    if not isinstance(row.get('note'), str):
        errors.append('note_must_be_text')


def check_overlays(row, size, errors):
    items = row.get('overlays')
    if not isinstance(items, list):
        errors.append('overlays_must_be_list')
        return []
    valid = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('kind'), str) or item['kind'] not in KINDS:
            errors.append('overlay_kind_invalid')
            continue
        box = checked_box(item.get('box'), size, 'overlay_box', errors)
        if box:
            valid.append({'kind': item['kind'], 'box': box})
    return valid


def check_status(row, overlays, errors, reasons):
    status = row.get('overlay_status')
    uncertain = row.get('needs_review') is True
    if not isinstance(status, str) or status not in {'present', 'absent', 'uncertain'}:
        errors.append('overlay_status_invalid')
    if status == 'absent' and overlays:
        errors.append('absent_with_overlay_boxes')
    if status == 'present' and not overlays and not uncertain:
        errors.append('present_without_overlay_boxes')
    if status == 'uncertain' and not uncertain:
        errors.append('uncertain_status_requires_review')
    if uncertain:
        reasons.append('annotator_uncertain')
        if not isinstance(row.get('note'), str) or not row['note'].strip():
            errors.append('review_note_required')


def validate_record(row, reference):
    errors, reasons = [], []
    if not isinstance(row, dict):
        return {'errors': ['record_must_be_object'], 'review_reasons': []}
    check_metadata(row, reference, errors)
    uncertain = row.get('needs_review') is True
    size = reference['image_wh']
    window = checked_box(row.get('window'), size, 'window', errors, uncertain)
    clean = checked_box(row.get('clean_box'), size, 'clean_box', errors, uncertain)
    overlays = check_overlays(row, size, errors)
    check_status(row, overlays, errors, reasons)
    if clean and (window is None or not contains(window, clean)):
        errors.append('clean_box_outside_window')
    if clean and any(intersection(clean, o['box']) > 0 for o in overlays):
        (reasons if uncertain else errors).append('clean_box_contains_overlay')
    if window and any(intersection(window, o['box']) == 0 for o in overlays):
        reasons.append('overlay_outside_window')
    return {'errors': sorted(set(errors)), 'review_reasons': sorted(set(reasons))}


def annotation_signature(row):
    excluded = {'reviewed_at', '_return_file', '_line', '_source_sha256'}
    return json.dumps({k: v for k, v in row.items() if k not in excluded}, sort_keys=True, ensure_ascii=False)


def reconcile_records(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['task_id'], row['id'])].append(row)
    accepted, conflicts, duplicates = [], [], 0
    for key, group in sorted(grouped.items()):
        signatures = {annotation_signature(row) for row in group}
        if len(signatures) > 1:
            conflicts.append({'task_id': key[0], 'id': key[1], 'versions': group})
        else:
            accepted.append(group[0])
            duplicates += len(group)-1
    return accepted, conflicts, duplicates


def match_overlays(a, b, threshold=.5):
    edges = {i: sorted([j for j, other in enumerate(b) if item['kind'] == other['kind'] and
                       iou(item['box'], other['box']) >= threshold],
                      key=lambda j: -iou(item['box'], b[j]['box'])) for i, item in enumerate(a)}
    matched = {}

    def augment(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in matched or augment(matched[j], visited):
                matched[j] = i
                return True
        return False

    for i in range(len(a)):
        augment(i, set())
    return [{'a': i, 'b': j, 'iou': iou(a[i]['box'], b[j]['box'])} for j, i in matched.items()]


def edge_difference(a, b):
    return max(abs(x-y) for x, y in zip(a, b)) if a and b else None


def compare_pair(a, b, window_tolerance=8, clean_tolerance=16, overlay_iou=.5):
    window = edge_difference(a.get('window'), b.get('window'))
    clean = edge_difference(a.get('clean_box'), b.get('clean_box'))
    matches = match_overlays(a['overlays'], b['overlays'], overlay_iou)
    reasons = []
    if window is None or window > window_tolerance:
        reasons.append('window_difference')
    if clean is None or clean > clean_tolerance:
        reasons.append('clean_box_difference')
    if a['overlay_status'] != b['overlay_status']:
        reasons.append('overlay_status_difference')
    if len(matches) != len(a['overlays']) or len(matches) != len(b['overlays']):
        reasons.append('overlay_boxes_difference')
    return {'id': a['id'], 'window_max_edge_difference': window, 'clean_max_edge_difference': clean,
            'overlay_count_a': len(a['overlays']), 'overlay_count_b': len(b['overlays']),
            'overlay_matched': len(matches), 'overlay_matches': matches,
            'disagreement': bool(reasons), 'reasons': reasons}
