"""辅助裁剪标注工具（human-in-the-loop）。

模型先给预测（窗口框 + 上下裁剪线），人工在浏览器里确认或拖动修改，
每次操作都记录"模型预测 vs 人工最终"的差异，用于：
  1) 提高效率（模型先给答案，人只改错的地方）；
  2) 优化模型（人工修改后的结果直接成为下一轮训练标签；
     human_modified 字段用于评估"人工修改率"）。

工具同时覆盖四件事：窗口框、上下裁剪线、safe_box、quality。

用法：
    python tools/annotate_app.py --manifest data/train_ready.jsonl --root data
    python tools/annotate_app.py --folder "data/images/Ⅱc+Ⅱa-b" --root . --new-batch
然后在浏览器打开 http://127.0.0.1:8765

输出：data/annotations_human.jsonl（可直接被 build_manifest.py 合并）
"""
import argparse
import base64
import hmac
import json
import mimetypes
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # 让 import ddh 在任意工作目录下都能工作
IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.bmp')

STATE = {
    'rows': [],          # 待处理队列
    'by_id': {},
    'root': Path('.'),
    'out': Path('data/annotations_human.jsonl'),
    'done': {},          # id -> 已保存的记录
    'models': None,
    'meta': None,
    'lock': threading.Lock(),
    't_start': time.time(),
    'session': {},       # id -> 开始时间
}


def load_done():
    p = STATE['out']
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                STATE['done'][r['id']] = r


def append_done(rec):
    p = STATE['out']
    p.parent.mkdir(parents=True, exist_ok=True)
    with STATE['lock']:
        with open(p, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        STATE['done'][rec['id']] = rec


def load_models(args):
    import ddh                     # sys.path 已在模块顶部插入项目根目录
    import torch
    device = args.device
    wm, wa = ddh.load_checkpoint(args.window_checkpoint, device)
    bm, ba = ddh.load_checkpoint(args.crop_checkpoint, device)
    STATE['models'] = (wm, bm, ba, device)
    STATE['ddh'] = ddh
    if args.calibration and Path(args.calibration).exists():
        STATE['cal'] = json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    else:
        STATE['cal'] = {}


def predict(rid):
    """跑模型；返回预测与门控理由。无窗口时 box=None。"""
    row = STATE['by_id'][rid]
    if STATE['models'] is None:                    # --no-model：纯手动标注
        return {'id': rid, 'status': 'model_disabled', 'box': None, 'bounds': None, 'std': None,
                'quality_logit': None, 'reasons': ['model_disabled'],
                'window': list(row['window']) if row.get('window') else None,
                'window_source': 'manifest' if row.get('window') else None,
                'safe_box': list(row['safe_box']) if row.get('safe_box') else None,
                'crop': list(row['crop']) if row.get('crop') else None,
                'quality': row.get('quality')}
    ddh = STATE['ddh']
    wm, bm, ba, device = STATE['models']
    p = ddh.pipeline(row, str(STATE['root']), wm, bm, ba, device, samples=8)
    reasons = ddh.gate(p, row.get('style_group'), STATE['cal'], ba)
    out = {'id': rid, 'status': p['status'], 'box': p.get('box'),
           'bounds': p.get('bounds'), 'std': p.get('std'),
           'quality_logit': p.get('quality_logit'), 'reasons': reasons}
    if p.get('box'):
        margin = STATE['cal'].get('margin', 0)
        b = [max(0, p['bounds'][0] - margin), min(1, p['bounds'][1] + margin)]
        out['crop_box'] = ddh.raw_crop(p['box'], b)
    # 没标签时也能用：窗口优先用清单里的真值，其次用模型
    if row.get('window'):
        out['window'] = list(row['window'])
        out['window_source'] = 'manifest'
    elif p.get('box'):
        out['window'] = list(p['box'])
        out['window_source'] = 'model'
    else:
        out['window'] = None
        out['window_source'] = None
    out['safe_box'] = list(row['safe_box']) if row.get('safe_box') else None
    out['crop'] = list(row['crop']) if row.get('crop') else None
    out['quality'] = row.get('quality')
    out['reference'] = STATE['reference']
    out['top_offset'] = STATE['top_offset']
    out['top_target'] = STATE['top_target']
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _authed(self):
        """HTTP Basic 认证。未配置 --auth 时视为开放（仅限本机/局域网调试）。"""
        want = STATE.get('auth')
        if not want:
            return True
        got = self.headers.get('Authorization', '')
        if got.startswith('Basic '):
            try:
                u, _, pw = base64.b64decode(got[6:]).decode('utf-8').partition(':')
                return hmac.compare_digest(f'{u}:{pw}', want)
            except Exception:
                return False
        return False

    def _deny(self):
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm="DDH annotation"')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._authed():
            self._deny(); return
        u = urlparse(self.path)
        if u.path in ('/', '/index.html'):
            html = (Path(__file__).parent / 'webapp' / 'annotate.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        if u.path == '/api/queue':
            rows = [r for r in STATE['rows'] if r['id'] not in STATE['done']]
            self._json({'total': len(STATE['rows']), 'done': len(STATE['done']),
                        'remaining': len(rows), 'next': rows[0]['id'] if rows else None})
            return
        if u.path.startswith('/api/image/'):
            rid = u.path.rsplit('/', 1)[-1]
            row = STATE['by_id'].get(rid)
            if not row:
                self._json({'error': 'unknown id'}, 404)
                return
            p = STATE['root'] / row['path']
            data = p.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', mimetypes.guess_type(str(p))[0] or 'image/jpeg')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if u.path == '/api/stats':
            self._stats()
            return
        if u.path.startswith('/api/predict/'):
            rid = u.path.rsplit('/', 1)[-1]
            try:
                STATE['session'][rid] = time.time()
                self._json(predict(rid))
            except Exception as e:
                self._json({'error': repr(e)}, 500)
            return
        self._json({'error': 'not found'}, 404)

    def _stats(self):
        done = list(STATE['done'].values())
        mod = sum(1 for d in done if d.get('human_modified'))
        secs = [d['seconds'] for d in done if d.get('seconds')]
        self._json({'done': len(done), 'total': len(STATE['rows']), 'modified': mod,
                    'modified_rate': round(mod / len(done), 3) if done else None,
                    'avg_seconds': round(sum(secs) / len(secs), 1) if secs else None})

    def do_POST(self):
        if not self._authed():
            self._deny(); return
        u = urlparse(self.path)
        n = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(n).decode('utf-8')) if n else {}
        if u.path == '/api/save':
            rid = body.get('id')
            row = STATE['by_id'].get(rid)
            if not row:
                self._json({'error': 'unknown id'}, 404)
                return
            t0 = STATE['session'].get(rid)
            rec = {k: row.get(k) for k in
                   ('id', 'path', 'collect_group', 'graf_type', 'style_group', 'source_group', 'split')}
            rec.update({
                'window': body.get('window'),
                'window_source': body.get('window_source'),        # model / manifest / human
                'crop': body.get('crop'),                          # [top, bottom]，原图像素
                'safe_box': body.get('safe_box'),
                'quality': body.get('quality'),
                'model_crop': body.get('model_crop'),
                'model_window': body.get('model_window'),
                'human_modified': bool(body.get('human_modified')),
                'review_reasons': body.get('reasons', []),
                'seconds': round(time.time() - t0, 1) if t0 else None,
                'annotated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'tool': 'annotate_app/1',
            })
            append_done(rec)
            self._json({'ok': True, 'done': len(STATE['done'])})
            return
        if u.path == '/api/skip':
            rid = body.get('id')
            STATE['rows'] = [r for r in STATE['rows'] if r['id'] != rid] + [r for r in STATE['rows'] if r['id'] == rid]
            self._json({'ok': True})
            return
        if u.path == '/api/stats':
            self._stats()
            return
        self._json({'error': 'not found'}, 404)


def build_queue(args):
    if args.folder:
        base = Path(args.folder)
        files = sorted([p for p in base.rglob('*') if p.suffix.lower() in IMAGE_EXT])
        rows = []
        for p in files:
            rel = p.relative_to(STATE['root'])
            rows.append({'id': p.stem, 'path': str(rel).replace('\\', '/'), 'window': None,
                         'crop': None, 'safe_box': None, 'quality': None,
                         'collect_group': base.name, 'style_group': '', 'source_group': ''})
    else:
        rows = [json.loads(x) for x in Path(args.manifest).read_text(encoding='utf-8').splitlines() if x.strip()]
    STATE['rows'] = rows
    STATE['by_id'] = {r['id']: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/train_ready.jsonl')
    ap.add_argument('--folder', default='', help='直接对一批图工作（如新批次），忽略 manifest')
    ap.add_argument('--root', default='data')
    ap.add_argument('--out', default='data/annotations_human.jsonl')
    ap.add_argument('--window-checkpoint', default='runs/window.pt')
    ap.add_argument('--crop-checkpoint', default='runs/crop.pt')
    ap.add_argument('--calibration', default='')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--auth', default='',
                    help='用户名:密码，开启 HTTP 基本认证。放公网必须设置！')
    ap.add_argument('--host', default='127.0.0.1',
                    help='监听地址。局域网共享给别人用时传 0.0.0.0，'
                         '别人用 http://<你的IP>:端口 打开即可（他们不需要装任何东西）')
    ap.add_argument('--no-model', action='store_true', help='只手动标注，不跑模型')
    ap.add_argument('--crop-top-offset', type=int, default=0,
                    help='预置裁剪上边界的偏移量（负=往上，含更多顶部）。'
                         '例如新口径比旧口径多保留 72px 就传 -72')
    ap.add_argument('--top-target', default='',
                    help='上边界目标区间（距窗口顶边的像素），形如 140,200：'
                         '界面会显示当前值并在区间内变绿，帮助保持口径一致')
    ap.add_argument('--redo', action='store_true',
                    help='重做模式：不跳过已完成的图，并把上次的标注预填回界面（只改要改的那一项）')
    ap.add_argument('--reference', choices=['original', 'model', 'none'], default='original',
                    help='裁剪线的初始值来源：original=清单里的原标注（重标用），'
                         'model=模型建议（新数据用），none=空白')
    args = ap.parse_args()
    def _p(x):                      # 相对路径一律相对项目根，避免依赖当前工作目录
        x = Path(x)
        return x if x.is_absolute() else (ROOT / x)
    STATE['root'] = _p(args.root)
    STATE['reference'] = args.reference
    STATE['auth'] = args.auth
    STATE['top_offset'] = args.crop_top_offset
    STATE['top_target'] = args.top_target
    STATE['out'] = _p(args.out)
    for k in ('window_checkpoint', 'crop_checkpoint', 'calibration'):
        if getattr(args, k):
            setattr(args, k, str(_p(getattr(args, k))))
    build_queue(args)
    if args.redo and STATE['out'].exists():
        prev = {json.loads(x)['id']: json.loads(x)
                for x in STATE['out'].read_text(encoding='utf-8').splitlines() if x.strip()}
        n = 0
        for r in STATE['rows']:
            pv = prev.get(r['id'])
            if pv:
                for k in ('window', 'crop', 'safe_box', 'quality'):
                    if pv.get(k) is not None:
                        r[k] = pv[k]
                n += 1
        print(f'重做模式：{n} 条已预填上次的标注（改完直接 Enter 覆盖）')
    load_done()
    if args.redo:
        STATE['done'] = {}          # 全部重新进队列
    if not args.no_model:
        load_models(args)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'队列 {len(STATE["rows"])} 张，已完成 {len(STATE["done"])} 张')
    print(f'输出 → {STATE["out"]}')
    import socket
    ips = ['127.0.0.1']
    if args.host == '0.0.0.0':
        try:
            s_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s_.connect(('8.8.8.8', 80)); ips.append(s_.getsockname()[0]); s_.close()
        except Exception:
            pass
    for ip in ips:
        print(f'浏览器打开  http://{ip}:{args.port}')
    if args.host == '0.0.0.0':
        print('（同一网络内的其他电脑用上面那个非 127 的地址打开即可，无需安装任何东西）')
        if not args.auth:
            print('⚠ 未设置 --auth，任何能访问该端口的人都能看到图片。放公网前务必加 --auth 用户名:密码')
    srv.serve_forever()


if __name__ == '__main__':
    main()
