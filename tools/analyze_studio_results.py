"""Audit original pixels, provenance and four-edge feedback without changing annotations."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from crop_geometry import validate_window
from studio_project import file_hash


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def contained(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Record path leaves selected directory')
    return path


def audit_record(record, source, output, number):
    raw = contained(source, record['file'])
    if file_hash(raw) != record['source_sha256']:
        raise ValueError(f'R{number:03}: original file hash mismatch')
    with Image.open(raw) as image:
        image = image.convert('RGB')
        if list(image.size) != record['original_size']:
            raise ValueError(f'R{number:03}: original dimensions mismatch')
        window = validate_window(record['window'], image.size)
        clean = validate_window(record['clean_box'], image.size)
        if not (window[0] <= clean[0] < clean[2] <= window[2] and window[1] <= clean[1] < clean[3] <= window[3]):
            raise ValueError(f'R{number:03}: crop leaves imaging window')
        cropped = contained(output, record['outputs']['cropped'])
        with Image.open(cropped) as saved:
            if saved.convert('RGB').tobytes() != image.crop(clean).tobytes() or saved.size != (clean[2]-clean[0], clean[3]-clean[1]):
                raise ValueError(f'R{number:03}: exported pixels differ from original crop')
        pixel_hash = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
    latest = record.get('latest_prediction') or record['prediction']
    if record.get('human_reviewed') and latest.get('window') != window:
        raise ValueError(f'R{number:03}: confirmed window differs from latest prediction window')
    prediction = latest['crop_box']
    return {'number': number, 'id': record['id'], 'file': record['file'], 'pixel_sha256': pixel_hash,
            'window': window, 'clean_box': clean,
            'trim_inside_window': [clean[0]-window[0], clean[1]-window[1], window[2]-clean[2], window[3]-clean[3]],
            'delta_from_latest_prediction': [clean[i]-prediction[i] for i in range(4)],
            'window_modified': record.get('window_modified', False),
            'human_reviewed': record.get('human_reviewed', False),
            'approved_for_training': record.get('approved_for_training', False),
            'source_verified': True, 'export_pixels_verified': True}


def sheets(records, output, report):
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 15)
    for start in range(0, len(records), 12):
        canvas = Image.new('RGB', (1280, 840), '#faf9f5')
        draw = ImageDraw.Draw(canvas)
        for offset, record in enumerate(records[start:start+12]):
            x, y = offset % 4 * 320, offset // 4 * 280
            with Image.open(contained(output, record['outputs']['cropped'])) as image:
                image = image.convert('RGB')
                image.thumbnail((306, 244))
                canvas.paste(image, (x+(320-image.width)//2, y+28+(244-image.height)//2))
            draw.text((x+10, y+5), f'R{start+offset+1:03}  {record["clean_box"][2]} right', font=font, fill='#282720')
        canvas.save(report/f'cropped_sheet_{start//12+1:02}.png')
    selected = [r for r in records if r.get('window_modified') or r['clean_box'][2] > 830]
    for number, record in enumerate(selected, 1):
        with Image.open(record['path']) as image:
            image = image.convert('RGB')
            draw = ImageDraw.Draw(image)
            draw.rectangle(record['window'], outline='#df8066', width=4)
            draw.rectangle(record['clean_box'], outline='#62c591', width=4)
            image.save(report/f'edge_case_{number:02}_{record["id"]}.png')


def analyze(output, report):
    meta = json.loads((output/'project.json').read_text(encoding='utf-8'))
    source = Path(meta['source']).resolve()
    input_hash = file_hash(output/'results.jsonl')
    records = rows(output/'results.jsonl')
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate image IDs in latest results')
    checked = [audit_record(record, source, output, i+1) for i, record in enumerate(records)]
    candidates = rows(output/'reviewed_training_candidates.jsonl')
    by_hash = {r['source_sha256']: r for r in records}
    for candidate in candidates:
        record = by_hash[candidate['source_sha256']]
        if candidate['window'] != record['window'] or candidate['clean_box'] != record['clean_box'] or candidate['approved_for_training']:
            raise ValueError('Training candidate differs from confirmed result or is already approved')
    trims = np.array([row['trim_inside_window'] for row in checked])
    deltas = np.array([row['delta_from_latest_prediction'] for row in checked])
    summary = {'project_images': meta['images'], 'results': len(records), 'candidates': len(candidates),
               'input_results_sha256': input_hash,
               'human_reviewed': sum(bool(r.get('human_reviewed')) for r in records),
               'modified_any': sum(bool(r.get('modified')) for r in records),
               'window_modified': sum(bool(r.get('window_modified')) for r in records),
               'source_hashes_and_export_pixels_verified': len(checked),
               'decoded_unique_images': len({r['pixel_sha256'] for r in checked}),
               'approved_for_training': sum(bool(r.get('approved_for_training')) for r in records),
               'geometry_is': 'final crop rectangles, not typed icon/text bounding boxes',
               'edge_changes_vs_latest_prediction': {name: int((deltas[:,i] != 0).sum()) for i, name in enumerate(('left','top','right','bottom'))},
               'right_edge': {'trimmed_images': int((trims[:,2] > 0).sum()),
                              'median_final_x': float(np.median([r['clean_box'][2] for r in checked])),
                              'median_trim_when_trimmed': float(np.median(trims[trims[:,2] > 0,2]))},
               'flags': dict(collections.Counter(flag for r in records for flag in r.get('flags', [])))}
    report.mkdir(parents=True, exist_ok=True)
    (report/'audit.json').write_text(json.dumps({'summary': summary, 'records': checked}, ensure_ascii=False, indent=2), encoding='utf-8')
    sheets(records, output, report)
    lines = ['# 余组裁剪反馈分析（2026-10-15）', '',
             f'本次读取 {len(records)} 张最新结果和 {len(candidates)} 条训练候选，未改写原始图片、审核记录或模型权重。', '',
             f'- 人工确认：{summary["human_reviewed"]} 张；调整裁剪范围：{summary["modified_any"]} 张；修改完整窗口：{summary["window_modified"]} 张。',
             f'- 原图 SHA256、坐标范围、导出尺寸和原图裁剪像素一致性：{len(checked)} 张通过。',
             f'- 去除完全相同解码图片后：{summary["decoded_unique_images"]} 张。',
             '- 所有候选保持未批准训练。', '', '## 四边变化', '',
             '| 相对最近模型建议 | 调整张数 |', '|---|---:|']
    lines += [f'| {label} | {summary["edge_changes_vs_latest_prediction"][key]} |' for label, key in [('左边','left'),('上边','top'),('右边','right'),('下边','bottom')]]
    lines += ['', f'相对完整成像窗口，{summary["right_edge"]["trimmed_images"]} 张收紧了右边界；收紧的中位数为 {summary["right_edge"]["median_trim_when_trimmed"]:.0f} 像素，最终右边界的中位数为 x={summary["right_edge"]["median_final_x"]:.0f}。', '',
              f'{summary["modified_any"]} 张调整不能直接理解为模型错误率：现有模型只预测上下边界，这次审核还增加了移除成像区内图标和文字的左右裁剪要求。上下边界也允许在保留结构的前提下偏松或偏紧。', '',
              '## 这些数据能训练什么', '',
              '当前 JSON 保存的是完整成像窗口与最终保留矩形，适合训练四边 clean_box，或学习某类版式下的左右裁剪建议。它没有给出婴儿图标、文字各自的目标框和类别，不能直接当作图标检测器的监督标签。', '',
              '这 85 张来自同一任务目录，版式相近。后续需要与成员标注合并复核、检查近重复和来源分组，并保留独立设备/来源测试集。完全相同图片去重并不足以证明无患者或连续帧泄漏。', '',
              '## 复核入口', '',
              'cropped_sheet_*.png 展示全部成品，R001 等编号对应 audit.json 中的 number。edge_case_*.png 展示修改过窗口或右边界超过 830 的情况；图标是否落在最终框内需结合其垂直位置检查，不能只用固定 x 坐标判定。', '',
              '像素校验确认导出忠实于所选矩形，不代表已经完成解剖结构保留或所有设备文字的语义复核。']
    (report/'分析报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.project.resolve(), args.report.resolve()), ensure_ascii=False))
