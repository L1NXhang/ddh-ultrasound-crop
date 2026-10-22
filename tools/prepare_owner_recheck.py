"""Package uncertain human annotations for owner review without approving them."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def select_records(rows):
    selected = sorted((r for r in rows if r.get('needs_review') is True), key=lambda r: r['filename'])
    if not selected or len({r['id'] for r in selected}) != len(selected):
        raise ValueError('Empty review task or duplicate IDs')
    if any('/' in r['filename'] or '\\' in r['filename'] or r['filename'] in ('.', '..') for r in selected):
        raise ValueError('Unsafe image filename')
    return selected


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Review template changed: ' + old[:60])
    return text.replace(old, new, 1)


OVERLAY_EDITOR = '''
function renderOverlayEditor(){
 const panel=$('existingOverlays');panel.replaceChildren();
 for(const [i,item] of draft.overlays.entries()){
  const label=document.createElement('div');label.className='small';label.textContent='已有覆盖物 '+(i+1);
  const select=document.createElement('select');select.dataset.overlayIndex=String(i);
  for(const [value,title] of [['device_text','设备信息 / 文字'],['body_marker','体位示意图 / 婴儿形图标'],['scale','标尺 / 灰度条'],['other','其他覆盖物']]){
   const option=document.createElement('option');option.value=value;option.textContent=title;select.append(option);
  }
  select.value=item.kind;select.onchange=()=>{draft={...draft,overlays:draft.overlays.map((o,j)=>j===i?{...o,kind:select.value}:o)};draw()};
  panel.append(label,select);
 }
}
'''


def render_html(template, task):
    html = replace_once(template, '__TASK_JSON__', json.dumps(task, ensure_ascii=False).replace('<', '\\u003c'))
    html = replace_once(html, "'ddh-cleanup-v1:'+task.task_id", "'owner-recheck-v1:'+task.task_id+':'+task.source_sha256")
    html = replace_once(html, ':fresh();syncControls()', ':JSON.parse(JSON.stringify(task.prefill[row.id]));syncControls()')
    html = replace_once(html, "$('note').value=draft.note}", "$('note').value=draft.note;renderOverlayEditor()}")
    html = replace_once(html, 'function fresh()', OVERLAY_EDITOR + '\nfunction fresh()')
    html = replace_once(html, '<button id="undo"', '<div id="existingOverlays" style="margin-top:12px"></div><button id="undo"')
    html = replace_once(html, 'overlays:draft.overlays.slice(0,-1)};draw()', 'overlays:draft.overlays.slice(0,-1)};renderOverlayEditor();draw()')
    html = replace_once(html, "progress();setMode('window');", "progress();setMode('clean');")
    html = replace_once(html, 'reviewed_at:new Date().toISOString()', "reviewed_at:new Date().toISOString(),review_author:'张林',review_stage:'owner_recheck',prior_reviewed_at:task.prefill[row.id].reviewed_at,review_parent_source_sha256:task.source_sha256")
    html = replace_once(html, "link.download='结果_'+task.task_id+'.jsonl'", "link.download='结果_'+task.task_id+'_负责人复核24张.jsonl'")
    html = replace_once(html, "'40 张任务图片已就绪'", "task.rows.length+' 张复核图片已就绪'")
    html = replace_once(html, "task.member+' · 40 张 · 独立完成'", "'张林复核 · '+task.rows.length+' 张 · 原标注：'+task.member")
    html = html.replace('本机离线 · 独立试标', '本机离线 · 负责人复核')
    html = html.replace('保存并下一张 · Enter', '复核并保存下一张 · Enter')
    html = html.replace('从一张原图开始', '复核成员的 24 张疑问图')
    html = html.replace('不预填算法框，请按自己的判断独立标注。', '已带入成员原框；仅你按保存的图片计入复核进度。')
    return html


GUIDE = '''负责人复核 · 24张

1. 完整解压，双击“开始复核.html”，用Edge或Chrome打开。
2. 在页面选择本包 images 文件夹，成员原来的框会自动显示。复核进度从0/24开始。
3. 蓝框是完整成像区；绿框是最终裁剪框；黄框是覆盖物。原框正确就保留，不需重画。
4. 在“已有覆盖物”下拉框直接改类型。体位/婴儿示意图选“体位示意图 / 婴儿形图标”。
5. 默认C裁剪模式，拖出新绿框可替换原框。W修改完整成像区，E新增覆盖物。
   修改蓝框会清空绿框，需要重新画绿框。撤销按钮删除最后一个覆盖物，保留其他框。
6. 如果覆盖物与最终框仍有相交，要检查实际图像；不要为了去图标强行切掉需要的结构。
   确认问题已解决后取消“需要负责人复核”。拿不准就保留勾选，把备注“1”改成具体原因。
7. 每张按Enter或“复核并保存下一张”，保存后自动下一张。无需扣除或补上2%。
8. 每隔几张导出备份。完成时导出结果JSONL，只交最后一份完整24条的文件。

页面不训练模型，不修改原始标注，不自动批准训练。只导出你重新保存过的图片，未保存的不导出。
未解决的疑问也可以保存回传，但会继续排除训练。途中离开前先导出；浏览器缓存不是正式备份。
此任务沿用原成员任务和图片ID，另记review_author=张林及原文件指纹，用于明确合并修订。
原238条回传保持原样，不把新旧文件混放后自动择新。将24条复核结果交回，由负责人核对合并。
'''


def build(source, destination):
    rows = [json.loads(s) for s in source.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    selected = select_records(rows)
    if len(selected) != 24 or len({r['task_id'] for r in selected}) != 1:
        raise ValueError('Expected 24 records from one task')
    if destination.exists() or destination.with_suffix('.zip').exists():
        raise ValueError('Review package already exists; do not overwrite')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    tasks = [{k: r[k] for k in ['id', 'filename', 'original_path', 'image_wh', 'pixel_sha256', 'source_id', 'assignment_group']} for r in selected]
    for r in selected:
        from PIL import Image
        with Image.open(ROOT / r['original_path']) as image:
            im = image.convert('RGB')
            if hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest() != r['pixel_sha256']:
                raise ValueError('Source image fingerprint changed')
    task = {'task_id': selected[0]['task_id'], 'member': selected[0]['reviewer'], 'rows': tasks,
            'prefill': {r['id']: r for r in selected}, 'source_sha256': source_hash}
    html = render_html((ROOT / 'product/边缘信息复核.html').read_text(encoding='utf-8'), task)
    (destination / 'images').mkdir(parents=True)
    for r in selected:
        shutil.copy2(ROOT / r['original_path'], destination / 'images' / r['filename'])
    (destination / '开始复核.html').write_text(html, encoding='utf-8')
    (destination / '操作说明.txt').write_text(GUIDE, encoding='utf-8-sig')
    (destination / '原24条_仅溯源勿作为复核结果.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected), encoding='utf-8')
    archive = destination.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(destination))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP validation failed')
    return {'images': len(selected), 'folder': str(destination), 'zip': str(archive), 'source_sha256': source_hash}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.out), ensure_ascii=True))
