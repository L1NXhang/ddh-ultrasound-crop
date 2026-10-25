"""Studio 训练功能：子进程训练作业管理与子进程入口。

frozen 版与源码版共用一条路径：父进程用 ``sys.executable`` 启动
``<入口> --train-job <spec.json>`` 子进程；子进程导入 ddh_v2 直接训练，
逐行打印进度，最后一行 ``TRAIN_RESULT:{...}`` 为机器可读结果。
若清单缺 split 字段，先做近重复审计并按组划分（seed 42，与项目口径一致），
再训练步骤 B（裁剪线模型）。
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

LINE_LIMIT = 400          # 保留最近 400 行日志供前端查看
SPLIT_VALUES = {'train', 'val', 'calib', 'test'}


def run_train_job(spec_file):
    """子进程入口：读取 spec → （按需）审计与划分 → 训练 → 打印结果行。"""
    from types import SimpleNamespace
    spec = json.loads(Path(spec_file).read_text(encoding='utf-8'))
    from studio_components import ensure_components, torch_available
    ensure_components(Path(__file__).resolve().parents[1])
    if not torch_available():
        raise SystemExit('推理组件未安装：无法训练，请先在「组件」中安装 CUDA 组件')
    rows = [json.loads(x) for x in Path(spec['manifest']).read_text(encoding='utf-8').splitlines() if x.strip()]
    if not rows:
        raise SystemExit('清单为空：至少需要一行带 window 与 crop 的标注')
    missing = [r for r in rows if not (r.get('window') and r.get('crop'))]
    if missing:
        raise SystemExit(f'清单里有 {len(missing)} 行缺少 window 或 crop 标注，请先完成审核再训练')
    manifest = spec['manifest']
    if not all(r.get('split') in SPLIT_VALUES for r in rows):
        print('清单缺少 split 字段：先做近重复审计，并按相似组划分 train/val（seed 42）…', flush=True)
        import ddh
        out_dir = Path(spec['output']).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        grouped, split = out_dir/'_audit_grouped.jsonl', out_dir/'_audit_split.jsonl'
        ddh.audit(SimpleNamespace(manifest=manifest, root=spec['root'], output=str(grouped),
                                  report=str(out_dir/'_audit_pairs.json'),
                                  hash_distance=4, correlation=.98))
        ddh.split(SimpleNamespace(manifest=str(grouped), output=str(split), reviewed=True,
                                  seed=42, holdout_style=''))
        manifest = str(split)
        print(f'划分完成：{manifest}', flush=True)
    import ddh_v2
    device = spec.get('device', 'auto')
    if device not in ('cpu', 'cuda'):
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'设备：{device}（auto 自动选择）', flush=True)
    ddh_v2.training(SimpleNamespace(
        seed=42, manifest=manifest, root=spec['root'], stage='crop', device=device,
        no_pretrained=bool(spec.get('no_pretrained', False)), baseline=False,
        size=int(spec.get('size', 384)), batch=int(spec.get('batch', 8)),
        epochs=int(spec.get('epochs', 30)), patience=int(spec.get('patience', 6)),
        output=spec['output'], safe_mode='derived', delta=0.04, w_in=2.0, w_out=0.25))
    print('TRAIN_RESULT:'+json.dumps({'status': 'completed', 'output': spec['output']}, ensure_ascii=False), flush=True)


class TrainingService:
    """同一时间只跑一个训练作业；日志环形缓冲；进程退出后汇总结果。"""

    def __init__(self, runtime_dir):
        self.runtime_dir = Path(runtime_dir)
        self.lock = threading.Lock()
        self.proc = None
        self.reader = None
        self.job = {'state': 'idle', 'lines': [], 'result': None, 'error': None,
                    'started': None, 'finished': None, 'spec': None}

    def start(self, spec):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                raise ValueError('已有训练在进行，请先停止或等待完成')
            for key in ('manifest', 'root', 'output'):
                if not str(spec.get(key, '')).strip():
                    raise ValueError('请填写数据清单、图片根目录与输出位置')
            if not Path(spec['manifest']).is_file():
                raise ValueError('数据清单文件不存在：'+str(spec['manifest']))
            if not Path(spec['root']).is_dir():
                raise ValueError('图片根目录不存在：'+str(spec['root']))
            Path(spec['output']).parent.mkdir(parents=True, exist_ok=True)
            self.runtime_dir.mkdir(parents=True, exist_ok=True)
            spec_file = self.runtime_dir/f'train_job_{int(time.time()*1000)}.json'
            spec_file.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
            if getattr(sys, 'frozen', False):
                # 打包版：入口就是后端 exe 自身（--train-job 由 crop_studio.main 解析）
                command = [sys.executable, '--train-job', str(spec_file)]
            else:
                command = [sys.executable, str(Path(__file__).with_name('crop_studio.py')),
                           '--train-job', str(spec_file)]
            creation = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            self.proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding='utf-8', errors='replace', cwd=str(Path(__file__).resolve().parents[1]),
                creationflags=creation)
            self.job = {'state': 'running', 'lines': [], 'result': None, 'error': None,
                        'started': time.time(), 'finished': None, 'spec': spec}
            self.reader = threading.Thread(target=self._consume, daemon=True)
            self.reader.start()
            return {'ok': True}

    def _consume(self):
        proc, tail = self.proc, []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip('\n')
            tail.append(line)
            if len(tail) > LINE_LIMIT:
                tail.pop(0)
            self.job['lines'] = tail
        code = proc.wait()
        self.job['finished'] = time.time()
        result_line = next((x[len('TRAIN_RESULT:'):] for x in tail if x.startswith('TRAIN_RESULT:')), None)
        if code == 0 and result_line:
            try:
                self.job['result'] = json.loads(result_line)
            except ValueError:
                self.job['result'] = {'status': 'completed'}
            self.job['state'] = 'completed'
        else:
            self.job['state'] = 'failed'
            self.job['error'] = next((x for x in reversed(tail) if x.strip()), f'训练进程退出（代码 {code}）')

    def status(self):
        with self.lock:
            job = dict(self.job)
            running = self.proc is not None and self.proc.poll() is None
            job['running'] = running
            job['lines'] = job['lines'][-25:]
            return job

    def stop(self):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                return {'ok': True, 'note': '没有正在进行的训练'}
            self.proc.terminate()
            return {'ok': True}
