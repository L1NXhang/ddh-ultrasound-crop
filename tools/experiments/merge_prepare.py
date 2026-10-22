"""Prepare the merged training set: old + member2 + member3 + member1(clean 214).

Key decisions (all recorded in protocol.json):
  - member1 keeps its 214 clean rows; the 24 needs_review rows are EXCLUDED
    (owner instruction 2026-10-22: "有问题的图片就不用").
  - member2 (237) and member3 (237) are fully usable; owner confirmed the
    2% margin on the old model did not corrupt the human final boxes.
  - Labels are all "final_crop_box"; inference/training use margin=0.
  - Splits are assigned by similarity leak_group so near-duplicate frames
    never straddle train/test.
  - The 100 pre-reserved images are used ONLY for leak detection, never trained.
"""
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_new_device_round import read, feature, group_features, portable, write_rows

SEED = 20261022
HANDOUT = ROOT / 'data/handout/新设备标注_20261016'


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def member1_rows():
    vals = read(ROOT / 'data/reports/member1_return_20261017/validated_proposals.jsonl')
    ok = [r for r in vals if not r['needs_review']]
    held = [dict(r, exclusion='annotator_uncertain_owner_dropped') for r in vals if r['needs_review']]
    assert len(vals) == 238 and len(ok) == 214 and len(held) == 24, (len(vals), len(ok), len(held))
    out = []
    for row in ok:
        clean = row['clean_box']
        out.append(feature(dict(row, path=row['original_path'], crop=[clean[1], clean[3]],
                                cohort='new', annotation_author='成员1', origin='member1',
                                human_reviewed=True, label_semantics='final_crop_box',
                                annotation_source='edge_cleanup/human_drawn', quality=None, safe_box=None)))
    return out, held


def member2_rows(manifest_by_sha):
    res = [json.loads(l) for l in Path(r"D:\ddhmodel\DDH裁剪结果\成员2_237张_e15a0b8a00\results.jsonl")
           .read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    out = []
    for r in res:
        m = manifest_by_sha[r['source_sha256']]
        cb = r['clean_box']
        rec = dict(id=m['id'], path=m['original_path'], window=list(r['window']),
                   crop=[cb[1], cb[3]], clean_box=cb, cohort='new', annotation_author='张林',
                   origin='member2', human_reviewed=True, label_semantics='final_crop_box',
                   annotation_source='crop_studio/human', quality=None, safe_box=None,
                   source_sha256=r['source_sha256'], assignment_group=m.get('assignment_group'),
                   model_crop_box=r.get('model_crop_box'), return_margin=0.02)
        out.append(feature(rec))
    assert len(out) == 237, len(out)
    return out


def member3_rows(zip_path, manifest_by_sha):
    import zipfile
    zf = zipfile.ZipFile(zip_path)
    res = [json.loads(l) for l in zf.read('results.jsonl').decode('utf-8').splitlines() if l.strip()]
    out = []
    for r in res:
        m = manifest_by_sha[r['source_sha256']]
        cb = r['clean_box']
        rec = dict(id=m['id'], path=m['original_path'], window=list(r['window']),
                   crop=[cb[1], cb[3]], clean_box=cb, cohort='new', annotation_author='成员3',
                   origin='member3', human_reviewed=True, label_semantics='final_crop_box',
                   annotation_source='crop_studio/human', quality=None, safe_box=None,
                   source_sha256=r['source_sha256'], assignment_group=m.get('assignment_group'),
                   model_crop_box=r.get('model_crop_box'), return_margin=0.0)
        out.append(feature(rec))
    assert len(out) == 237, len(out)
    return out


def main():
    if (OUT / 'protocol.json').exists():
        raise ValueError('Frozen protocol already exists in ' + str(OUT))

    # manifests -> sha lookup for member2/3 (their tools hash raw bytes)
    def manifest_sha_map(member, count):
        rows = read(HANDOUT / f'成员{member}_{count}张/任务清单.jsonl')
        return {file_sha(ROOT / r['original_path']): r for r in rows}

    m2 = member2_rows(manifest_sha_map(2, 237))
    m3 = member3_rows(ROOT / 'data/new_device_returns/20261021_member3/成员3_回传_20261021.zip',
                      manifest_sha_map(3, 237))
    m1, held1 = member1_rows()

    previous_raw = read(ROOT / 'data/experiments/finetune_zhanglin_20261016/split.jsonl')
    already = {str(r['id']) for r in previous_raw}

    # member2's 237 rows are ALREADY part of the previous split (frozen 153/35/49).
    # Re-adding them would duplicate images and let them match their own group.
    m2 = [r for r in m2 if str(r['id']) not in already]
    already_reported = 237 - len(m2)

    previous = [feature(dict(r, origin='previous')) for r in previous_raw]
    reserved = [feature(dict(r, origin='reserved'))
                for r in read(HANDOUT / '预留评估_暂不标注不训练.jsonl')]

    # ids must be globally unique across sources
    def key(r):
        return str(r['id'])
    all_new = m1 + m2 + m3
    assert len({key(r) for r in all_new}) == len(all_new), 'duplicate ids across member returns'

    # feature() carries numpy vectors; grouping needs them, serialisation must not
    groups, pairs = group_features(all_new + previous + reserved)
    usable, excluded_similarity = [], []
    for index, group in enumerate(groups):
        current = [r for r in group if r['origin'] in ('member1', 'member2', 'member3')]
        if not current:
            continue
        prev_in_group = any(r['origin'] == 'previous' for r in group)
        if prev_in_group:
            # an image already in the previous split cannot be re-split; keep the
            # previous record authoritative and hold these out of the new labels.
            excluded_similarity.extend(dict(portable(r), exclusion='similarity_to_previous_split')
                                       for r in current)
        else:
            usable.append([dict(portable(r), leak_group=f'NEW_{index:04d}') for r in current])

    random.Random(SEED).shuffle(usable)
    a, b = int(len(usable) * .65), int(len(usable) * .80)
    added = [dict(r, split='train' if i < a else 'val' if i < b else 'test',
                  approved_for_training=False, dataset_usage='user_authorized_20261022')
             for i, group in enumerate(usable) for r in group]

    merged = [portable(r) for r in previous] + added
    assert len({r['id'] for r in merged}) == len(merged)
    assert not ({r['id'] for r in added} & {r['id'] for r in previous})

    # carry prior splits for the old cohort, new splits for the new cohort
    for r in merged:
        if r.get('cohort') == 'old' and not r.get('split'):
            raise ValueError('old row missing split')
        r.pop('vector', None)
        r.pop('bits', None)

    write_rows(OUT / 'split.jsonl', merged)
    write_rows(OUT / 'added.jsonl', [portable(r) for r in added])
    write_rows(OUT / 'excluded.jsonl', [portable(r) for r in held1 + excluded_similarity])

    counts = collections.Counter(r['cohort'] + '/' + r['split'] for r in merged)
    by_author = collections.Counter(r.get('annotation_author', 'legacy') + '/' + r['split']
                                    for r in added)
    protocol = {
        'seed': SEED, 'split_seed': SEED, 'date': '2026-10-22',
        'user_authorized_training': True,
        'label_semantics': 'final_crop_box', 'crop_policy': 'final_bounds/v1', 'margin': 0.0,
        'parent_checkpoint': 'runs_v2/crop.pt',
        'parent_sha256': hashlib.sha256((ROOT / 'runs_v2/crop.pt').read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256((OUT / 'split.jsonl').read_bytes()).hexdigest(),
        'sources': {
            'previous_split': 'data/experiments/finetune_zhanglin_20261016/split.jsonl',
            'member1': 'data/reports/member1_return_20261017/validated_proposals.jsonl (214 of 238; 24 needs_review dropped by owner)',
            'member2': 'DDH裁剪结果/成员2_237张_e15a0b8a00/results.jsonl',
            'member3': 'data/new_device_returns/20261021_member3/成员3_回传_20261021.zip',
        },
        'added_counts': {k: v for k, v in sorted(by_author.items())},
        'counts': {k: v for k, v in sorted(counts.items())},
        'excluded': {'member1_needs_review': len(held1),
                     'member2_already_in_previous_split': already_reported,
                     'similarity_to_previous_split': len(excluded_similarity)},
        'similarity_pairs': len(pairs),
        'previous_splits_preserved': True,
        'reserved100_used_for_training_or_prediction': False,
        'selection': 'Equal old/new validation loss; max 30 epochs, patience 6, same loss and LR as previous experiment',
        'evaluation': 'Frozen margin=0; reference and production-rule windows; identical seeds',
        'deployment_gate': 'No automatic deployment; check both edge errors, tails, and old-device regression',
        'limitations': [
            'Similarity groups do not establish patient independence.',
            'Member2 return was produced with the old model (margin=0.02); member3 with margin=0. Human final boxes are used as-is for both.',
            'Member1 return came from an independent annotation tool; no model prediction recorded.',
            '24 member1 needs_review images excluded per owner instruction.',
            'No overlay detection or left/right training.',
        ],
    }
    (OUT / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'counts': protocol['counts'], 'added': protocol['added_counts'],
                      'excluded': protocol['excluded'], 'rows': len(merged)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()