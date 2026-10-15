"""批量裁剪工具（本地单机版）：选一个文件夹，逐张确认，输出两种结果。

流程：窗口由算法逐图检测（不套模板）→ 裁剪线由模型预测 → 人工按 Enter 接受或拖线微调
输出：<out>/cropped/<原名>.png    按原图像素裁好的图（不缩放）
      <out>/annotated/<原名>.png  原图 + 红框(窗口) + 绿线(上边界) + 蓝线(下边界)
      <out>/results.jsonl         每张的窗口/裁剪框/是否人工微调（可追溯，也是训练数据）

用法：
  python tools/crop_app.py --folder "D:/待裁图" --out "D:/裁剪结果"
  python tools/crop_app.py --folder "D:/待裁图" --out "D:/裁剪结果" --dry-run   # 只看不写
然后浏览器打开 http://127.0.0.1:8770
"""
import argparse
import json
import mimetypes
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from crop_geometry import validate_window, validate_crop, modification_details

IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.bmp')

STATE = {'rows': [], 'folder': None, 'out': None, 'done': {}, 'lock': threading.Lock(),
         'models': None, 'dry': False, 'session': {}, 'session_done': set(),
         'predictions': {}, 'latest_predictions': {}, 'model_lock': threading.Lock()}


def load_done():
    p = STATE['out'] / 'results.jsonl'
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                STATE['done'][r['id']] = r


def append_done(rec):
    with STATE['lock']:
        updated = {**STATE['done'], rec['id']: rec}
        p = STATE['out'] / 'results.jsonl'
        p.parent.mkdir(parents=True, exist_ok=True)
        pending = p.with_suffix('.jsonl.tmp')
        pending.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in updated.values()), encoding='utf-8')
        pending.replace(p)
        STATE['done'] = updated


def load_models(args):
    import torch
    import ddh
    dev = args.device
    bm, ba = ddh.load_checkpoint(args.crop_checkpoint, dev)
    STATE['models'] = (bm, ba, dev)
    STATE['ddh'] = ddh
    STATE['device'] = dev


def predict(rid):
    """窗口=算法逐图检测；裁剪线=模型预测（相对窗口的比例）"""
    from detect_window import detect
    r = next(x for x in STATE['rows'] if x['id'] == rid)
    path = STATE['folder'] / r['file']
    d = detect(path)
    out = {'id': rid, 'file': r['file'], 'window': d['box'], 'window_flags': d['flags'],
           'window_confidence': d['confidence']}
    if not d['box']:
        out['crop'] = None
        STATE['predictions'][rid] = dict(out)
        STATE['latest_predictions'][rid] = dict(out)
        return out
    out.update(predict_crop(path, d['box']))
    STATE['predictions'][rid] = dict(out)
    STATE['latest_predictions'][rid] = dict(out)
    return out


def predict_crop(path, window):
    bm, ba, dev = STATE['models']
    ddh = STATE['ddh']
    import torch
    from PIL import Image
    im = Image.open(path).convert('RGB')
    window = validate_window(window, im.size)
    x, roi = ddh.letterbox(im.crop(tuple(window)), ba['size'])
    with STATE['model_lock'], torch.no_grad():
        o = ddh.mc_crop(bm, x[None].to(dev), roi[None].to(dev), 4)
    b = [max(0.0, float(o['bounds'][0][0]) - STATE.get('margin', 0.0)),
         min(1.0, float(o['bounds'][0][1]) + STATE.get('margin', 0.0))]
    c = ddh.raw_crop(window, b)
    return {'crop': [c[1], c[3]], 'std': [float(v) for v in o['std'][0]]}


def export(rec):
    """写出裁剪图与标注图（都按原图像素）"""
    from PIL import Image, ImageDraw
    path = STATE['folder'] / rec['file']
    im = Image.open(path).convert('RGB')
    w = validate_window(rec['window'], im.size)
    c = validate_crop(rec['crop'], w)
    box = (w[0], c[0], w[2], c[1])
    if not STATE['dry']:
        (STATE['out'] / 'cropped').mkdir(parents=True, exist_ok=True)
        (STATE['out'] / 'annotated').mkdir(parents=True, exist_ok=True)
        im.crop(box).save(STATE['out'] / 'cropped' / (Path(rec['file']).stem + '.png'))
        ann = im.copy()
        d = ImageDraw.Draw(ann)
        d.rectangle([w[0], w[1], w[2], w[3]], outline=(255, 60, 60), width=3)
        d.line([0, c[0], ann.width, c[0]], fill=(0, 255, 120), width=3)
        d.line([0, c[1], ann.width, c[1]], fill=(80, 200, 255), width=3)
        ann.save(STATE['out'] / 'annotated' / (Path(rec['file']).stem + '.png'))
    rec['crop_box'] = list(box)
    rec['exported'] = not STATE['dry']


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.split('?')[0]
        if p in ('/', '/index.html'):
            h = (Path(__file__).parent / 'webapp' / 'crop.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(h)))
            self.end_headers()
            self.wfile.write(h)
            return
        if p == '/api/queue':
            rows = [r for r in STATE['rows'] if r['id'] not in STATE['done']]
            self._json({'total': len(STATE['rows']), 'done': len(STATE['done']),
                        'remaining': len(rows), 'next': rows[0]['id'] if rows else None})
            return
        if p == '/api/stats':
            done = list(STATE['done'].values())
            mod = sum(1 for r in done if r.get('modified'))
            self._json({'done': len(done), 'total': len(STATE['rows']), 'modified': mod,
                        'modified_rate': round(mod / len(done), 3) if done else None,
                        'out': str(STATE['out']), 'dry_run': STATE['dry']})
            return
        if p.startswith('/api/image/'):
            rid = p.rsplit('/', 1)[-1]
            r = next((x for x in STATE['rows'] if x['id'] == rid), None)
            if not r:
                self._json({'error': 'unknown'}, 404)
                return
            f = STATE['folder'] / r['file']
            data = f.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', mimetypes.guess_type(str(f))[0] or 'image/jpeg')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if p.startswith('/api/predict/'):
            rid = p.rsplit('/', 1)[-1]
            try:
                STATE['session'][rid] = time.time()
                self._json(predict(rid))
            except Exception as e:
                self._json({'error': repr(e)}, 500)
            return
        self._json({'error': 'not found'}, 404)

    def do_POST(self):
        try:
            n = int(self.headers.get('Content-Length', 0))
            if not 0 <= n <= 65536:
                raise ValueError('请求过大')
            body = json.loads(self.rfile.read(n).decode('utf-8')) if n else {}
            if not isinstance(body, dict):
                raise ValueError('请求必须为 JSON 对象')
        except (ValueError, UnicodeError) as error:
            self._json({'error': str(error)}, 400)
            return
        if self.path == '/api/repredict':
            self._repredict(body)
            return
        if self.path == '/api/save':
            self._save(body)
            return
        if self.path == '/api/skip':
            rid = body.get('id')
            STATE['rows'] = [x for x in STATE['rows'] if x['id'] != rid] + \
                            [x for x in STATE['rows'] if x['id'] == rid]
            self._json({'ok': True})
            return
        self._json({'error': 'not found'}, 404)

    def _save(self, body):
        r = next((x for x in STATE['rows'] if x['id'] == body.get('id')), None)
        if not r:
            self._json({'error': 'unknown id'}, 404)
            return
        initial = STATE['predictions'].get(r['id'])
        if initial is None:
            self._json({'error': '请先加载本图的预测结果'}, 400)
            return
        latest = STATE['latest_predictions'].get(r['id'], initial)
        if body.get('window') != latest.get('window'):
            self._json({'error': '窗口已修改，请先重新预测裁剪线'}, 400)
            return
        rec = {'id': r['id'], 'file': r['file'], 'window': body.get('window'),
               'crop': body.get('crop'), 'model_window': initial.get('window'),
               'model_crop': initial.get('crop'), 'flags': initial.get('window_flags', []),
               'latest_prediction': STATE['latest_predictions'].get(r['id']),
               'seconds': round(time.time() - STATE['session'].get(r['id'], time.time()), 1),
               'time': time.strftime('%Y-%m-%d %H:%M:%S')}
        try:
            rec.update(modification_details(rec['model_window'], rec['model_crop'], rec['window'], rec['crop']))
            export(rec)
            if not STATE['dry']:
                append_done(rec)
            else:
                STATE['done'] = {**STATE['done'], rec['id']: rec}
        except (ValueError, TypeError) as e:
            self._json({'error': str(e)}, 400)
            return
        except Exception as e:
            self._json({'error': f'导出失败: {e}'}, 500)
            return
        STATE['session_done'].add(r['id'])
        self._json({'ok': True, 'done': len(STATE['done'])})
        return

    def _repredict(self, body):
        from PIL import Image
        row = next((r for r in STATE['rows'] if r['id'] == body.get('id')), None)
        if row is None:
            self._json({'error': 'unknown id'}, 404)
            return
        path = STATE['folder'] / row['file']
        try:
            with Image.open(path) as im:
                window = validate_window(body.get('window'), im.size)
            prediction = dict(predict_crop(path, window), window=window)
            STATE['latest_predictions'][row['id']] = prediction
            self._json(prediction)
        except ValueError as error:
            self._json({'error': str(error)}, 400)
        except Exception as error:
            self._json({'error': f'重新预测失败: {error}'}, 500)


def build_queue(folder):
    files = sorted([p for p in folder.rglob('*') if p.suffix.lower() in IMAGE_EXT])
    rows = []
    for f in files:
        rows.append({'id': f.stem, 'file': str(f.relative_to(folder)).replace('\\', '/')})
    STATE['rows'] = rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', required=True, help='待裁剪的图片文件夹（会递归子目录）')
    ap.add_argument('--out', default='', help='输出目录（默认 <folder>_cropped）')
    ap.add_argument('--crop-checkpoint', default='runs_v2/crop.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--margin', type=float, default=0.02, help='额外安全边距（占窗口高度）')
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--dry-run', action='store_true', help='只看不写文件')
    ap.add_argument('--open-browser', action='store_true', help='服务就绪后自动打开浏览器')
    args = ap.parse_args()
    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f'找不到文件夹: {folder}')
    STATE['folder'] = folder
    STATE['out'] = Path(args.out) if args.out else folder.parent / (folder.name + '_cropped')
    STATE['out'].mkdir(parents=True, exist_ok=True)
    STATE['dry'] = args.dry_run
    STATE['margin'] = args.margin
    cp = Path(args.crop_checkpoint)
    if not cp.is_absolute():
        cp = ROOT / cp
    args.crop_checkpoint = str(cp)
    build_queue(folder)
    load_done()
    load_models(args)
    print(f'待处理 {len(STATE["rows"])} 张，已完成 {len(STATE["done"])} 张')
    print(f'输出目录 {STATE["out"]}')
    print(f'  cropped/  裁好的图（原图像素）')
    print(f'  annotated/ 标了窗口与裁剪线的图')
    print(f'  results.jsonl 记录')
    if args.dry_run:
        print('（--dry-run：不会写任何文件）')
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'\n浏览器打开  http://127.0.0.1:{args.port}')
    if args.open_browser:
        import webbrowser
        webbrowser.open(f'http://127.0.0.1:{args.port}/')
    srv.serve_forever()


if __name__ == '__main__':
    main()
