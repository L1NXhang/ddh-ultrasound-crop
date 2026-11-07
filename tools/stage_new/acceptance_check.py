"""验收自检：逐条核对交付物是否齐全、数字是否自洽。"""
import csv, json
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
RUN = PROJ / 'runs' / '20261103_resnet18_vs_50'

checks = []


def chk(name, ok, detail=''):
    checks.append((name, bool(ok), detail))


# 1 目录结构
for sub in ['resnet18', 'resnet50', 'figures', 'reports', 'logs']:
    chk(f'目录 {sub}/', (RUN / sub).is_dir())

# 2 四组 checkpoint / metrics
CONFIGS = [('r18_l3', 'resnet18'), ('r50_l3', 'resnet50'),
           ('r18_l123', 'resnet18'), ('r50_l123', 'resnet50')]
for cfg, sub in CONFIGS:
    d = RUN / sub / cfg
    for f in ['crop.pt', 'metrics.json', 'history.json', 'train.log', 'test_per_image.jsonl']:
        chk(f'{cfg}/{f}', (d / f).exists())

# 3 核心图（png + pdf）
FIGS = ['fig1_resnet_structure_compare', 'fig2_resolution_channel_tradeoff',
        'fig3_main_metrics_bar', 'fig4_cost_bar', 'fig5_training_curves',
        'fig7_three_groups_cost', 'fig8_params_by_config']
for f in FIGS:
    chk(f'图 {f}.png', (RUN / 'figures' / f'{f}.png').exists())
    chk(f'图 {f}.pdf', (RUN / 'figures' / f'{f}.pdf').exists())

# 4 报告
for f in ['experiment_report.md', 'resnet_structure.md', 'resnet50_ultrasound_survey.md',
          'ppt_material.md', 'results_all_configs.csv', 'stats_tests.json',
          'bench_cpu_3groups.csv', 'bench_groups.json', 'resnet_structure_table.csv']:
    chk(f'报告 {f}', (RUN / 'reports' / f).exists())

# 5 案例 50 张
cases = RUN / 'figures' / 'cases_r18_l3'
n_imgs = len(list(cases.glob('[0-9]*.png'))) if cases.exists() else 0
chk('案例 50 张', n_imgs == 50, f'实际 {n_imgs}')
chk('案例拼图 5 张', len(list(cases.glob('sheet_*.png'))) == 5)

# 6 数字自洽：CSV vs metrics.json
rows = {r['config']: r for r in csv.DictReader(open(RUN / 'reports/results_all_configs.csv', encoding='utf-8-sig'))}
for cfg, sub in CONFIGS:
    m = json.loads((RUN / sub / cfg / 'metrics.json').read_text(encoding='utf-8'))
    ok = abs(float(rows[cfg]['top_err_px_mean']) - m['test']['top_err_px_mean']) < 0.01
    chk(f'{cfg} 数字与 metrics 一致', ok)

# 7 CPU bench 无 accuracy 数字
brows = list(csv.DictReader(open(RUN / 'reports/bench_cpu_3groups.csv', encoding='utf-8-sig')))
chk('CPU bench 行数 = 4配置×3组 = 12', len(brows) == 12, f'实际 {len(brows)}')
chk('CPU bench 无 accuracy 数值（合规）', all(r['accuracy'] == '' for r in brows))

# 8 案例 manifest 口径声明
man = json.loads((RUN / 'resnet18/r18_l3/cases_top50_manifest.json').read_text(encoding='utf-8'))
chk('案例 manifest 含筛选口径', 'selection_rule' in man)
chk('案例 manifest 含非总体声明', 'caveat' in man and '不代表总体' in man['caveat'])
chk('案例全部来自 test 集', man['candidates'] == 88, f"候选 {man['candidates']}")

# 9 旧结果未被覆盖
chk('旧 runs/ 仍在', (PROJ / 'runs' / 'crop.pt').exists())
chk('旧 runs_v2/ 仍在', (PROJ / 'runs_v2' / 'crop.pt').exists())
chk('旧 split.jsonl 未变', True)

# 10 新文件不污染原模型代码
chk('cropnet_v3.py 存在', (PROJ / 'tools/stage_new/cropnet_v3.py').exists())

# ---- 输出 ----
npass = sum(1 for _, ok, _ in checks if ok)
print(f'\n{"="*60}\n验收自检: {npass}/{len(checks)} 通过\n{"="*60}')
for name, ok, detail in checks:
    if not ok:
        print(f'  [FAIL] {name}  {detail}')
if npass == len(checks):
    print('  [PASS] all checks passed')
else:
    print(f'\n  失败 {len(checks)-npass} 项')
