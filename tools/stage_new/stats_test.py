"""对 18 vs 50（以及融合消融）做统计检验，避免把噪声当结论。

88 张 test 集的差异必须做检验才敢下结论：
  - 连续指标（像素误差、IoU）：配对 Wilcoxon 符号秩检验 + bootstrap 置信区间
  - 二值指标（3% 是否合格）：McNemar 检验

用法：python tools/stage_new/stats_test.py
"""
import json, sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
# 支持通过环境变量或 argv 指定运行目录（默认旧的 split.jsonl 版本）
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else (PROJ / 'runs' / '20261103_resnet18_vs_50')
if not RUN.is_absolute():
    RUN = PROJ / RUN
REP = RUN / 'reports'
REP.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ('r18_l3', 'resnet18', 'r50_l3', 'resnet50', 'ResNet18 vs ResNet50（均仅L3）'),
    ('r18_l3', 'resnet18', 'r18_l123', 'resnet18', 'ResNet18: 仅L3 vs L1+2+3融合'),
    ('r50_l3', 'resnet50', 'r50_l123', 'resnet50', 'ResNet50: 仅L3 vs L1+2+3融合'),
]


def load_per_image(cfg, sub):
    p = RUN / sub / cfg / 'test_per_image.jsonl'
    if not p.exists():
        return None
    return {json.loads(l)['id']: json.loads(l)
            for l in p.read_text(encoding='utf-8').splitlines() if l.strip()}


def wilcoxon(a, b):
    """配对 Wilcoxon 符号秩（正态近似，无 scipy 依赖）。返回 (z, p_two_sided)。"""
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    n = len(d)
    if n < 10:
        return None, None
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, float)
    ranks[order] = np.arange(1, n + 1)
    # 处理并列秩
    absd = np.abs(d)
    for v in np.unique(absd):
        idx = np.where(absd == v)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    w_plus = ranks[d > 0].sum()
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_plus - mu) / sigma
    # 双尾 p（正态近似）
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2))
    return float(z), float(p)


def mcnemar(pass_a, pass_b):
    """McNemar 精确检验（二项）。返回 (b, c, p_two_sided)。"""
    b = sum(1 for x, y in zip(pass_a, pass_b) if y and not x)  # A 不合格 B 合格
    c = sum(1 for x, y in zip(pass_a, pass_b) if x and not y)  # A 合格 B 不合格
    n = b + c
    if n == 0:
        return b, c, 1.0
    from math import comb
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return b, c, float(min(p, 1.0))


def bootstrap_ci(a, b, metric_name, n_boot=5000, seed=42):
    """配对 bootstrap：对 (差值) 做重采样，给出 95% CI。"""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi)


def main():
    out = {'note': 'test 集 88 张；配对检验。p>0.05 表示差异不能排除随机波动。', 'pairs': []}
    for cfg_a, sub_a, cfg_b, sub_b, label in PAIRS:
        A = load_per_image(cfg_a, sub_a)
        B = load_per_image(cfg_b, sub_b)
        if A is None or B is None:
            out['pairs'].append({'pair': label, 'status': '未完成',
                                 'missing': [c for c, s in [(cfg_a, A), (cfg_b, B)] if s is None]})
            continue
        ids = sorted(set(A) & set(B))
        rec = {'pair': label, 'n_paired': len(ids), 'a': cfg_a, 'b': cfg_b}
        # 连续指标
        for key, better in [('max_err_norm', 'lower'), ('vert_iou', 'higher'),
                            ('top_err_px', 'lower'), ('bot_err_px', 'lower')]:
            va = [A[i][key] for i in ids]
            vb = [B[i][key] for i in ids]
            z, p = wilcoxon(va, vb)
            mean_d, lo, hi = bootstrap_ci(va, vb, key)
            rec[key] = {
                'a_mean': round(float(np.mean(va)), 4),
                'b_mean': round(float(np.mean(vb)), 4),
                'diff_a_minus_b': round(mean_d, 4),
                'diff_95ci': [round(lo, 4), round(hi, 4)],
                'wilcoxon_z': round(z, 3) if z is not None else None,
                'p_value': round(p, 4) if p is not None else None,
                'significant_at_0.05': (p is not None and p < 0.05),
                'better': better,
            }
        # 二值：3% 合格率
        pa = [A[i]['pass_3pct'] for i in ids]
        pb = [B[i]['pass_3pct'] for i in ids]
        b, c, p = mcnemar(pa, pb)
        rec['pass_3pct'] = {
            'a_pass': int(sum(pa)), 'b_pass': int(sum(pb)), 'n': len(ids),
            'a_rate': round(sum(pa) / len(ids), 4), 'b_rate': round(sum(pb) / len(ids), 4),
            'mcnemar_b_a_fail_b_pass': b, 'mcnemar_c_a_pass_b_fail': c,
            'mcnemar_p': round(p, 4),
            'significant_at_0.05': p < 0.05,
        }
        out['pairs'].append(rec)

    (REP / 'stats_tests.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    for rec in out['pairs']:
        if rec.get('status') == '未完成':
            print(f"[未完成] {rec['pair']}"); continue
        print(f"\n=== {rec['pair']}  (n={rec['n_paired']}) ===")
        for k in ['top_err_px', 'bot_err_px', 'vert_iou']:
            d = rec[k]
            sig = '显著' if d['significant_at_0.05'] else '不显著'
            print(f"  {k:12s} {d['a_mean']:8.4f} vs {d['b_mean']:8.4f}  "
                  f"diff={d['diff_a_minus_b']:+.4f} CI={d['diff_95ci']}  p={d['p_value']}  {sig}")
        p3 = rec['pass_3pct']
        sig = '显著' if p3['significant_at_0.05'] else '不显著'
        print(f"  pass_3pct    {p3['a_rate']:.3f} vs {p3['b_rate']:.3f}  "
              f"(b={p3['mcnemar_b_a_fail_b_pass']}, c={p3['mcnemar_c_a_pass_b_fail']})  "
              f"p={p3['mcnemar_p']}  {sig}")


if __name__ == '__main__':
    main()
