"""等待训练结束，然后生成报告并推送到 git。由会话后台调用。

判定结束：日志出现「全部完成」，或日志 6 分钟没有更新（视为异常退出）。
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'data' / 'reports' / 'train_full.log'
PY = ROOT / '.venv' / 'Scripts' / 'python.exe'


def log_text():
    return LOG.read_text(encoding='utf-8', errors='replace') if LOG.exists() else ''


def main():
    t0 = time.time()
    last_size, last_change = -1, time.time()
    while time.time() - t0 < 3 * 3600:
        t = log_text()
        if '全部完成' in t:
            print('检测到训练完成')
            break
        if len(t) != last_size:
            last_size, last_change = len(t), time.time()
        elif time.time() - last_change > 360:
            print('日志 6 分钟无更新，判定训练已停止（可能异常退出）')
            break
        time.sleep(30)
    else:
        print('等待超时 3 小时')

    r = subprocess.run([str(PY), 'tools/make_report.py'], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-500:])
    for cmd in (['git', 'add', '-A'],
                ['git', '-c', 'user.name=张林', '-c', 'user.email=3578007488@qq.com',
                 'commit', '-m', '自动生成训练报告（含逐轮日志、端到端与 oracle 评估、分层结果、局限）'],
                ['git', 'push']):
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print(cmd[1] if cmd[0] == 'git' else cmd[0], '→', (p.stdout + p.stderr).strip()[-200:])
    print('报告与推送流程结束')


if __name__ == '__main__':
    main()
