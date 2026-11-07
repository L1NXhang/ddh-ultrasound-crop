"""DP / 图搜索 seam 基线：用来判断"上边界定位"到底是边缘检测问题还是语义问题。

设计目的（不是为了让 DP 赢，而是为了给瓶颈定位提供证据）
──────────────────────────────────────────────────────
经典做法（Liang et al. 2000 IEEE TMI 多尺度 DP、Wendelhag et al. 1997 Stroke）
认为边界是图像中的一条强梯度 seam，用动态规划逐列求最优路径。

本脚本做两件事：
  B1. 纯 DP seam：只用垂直梯度做代价，逐列找一条穿过窗口的 seam，取其中位行。
  B2. 理想化 DP：同上，但 seam 被允许在已知真值的 ±R 行范围内搜索
      （上界实验：即使给 DP 开"作弊"的先验，它能做到多好）。

判读方式（关键）：
  如果 B1 与 CNN 的误差在同一量级甚至更差，
  说明目标边界并不对应强梯度 seam —— 任务被"标注约定"封顶，
  继续换网络结构不会带来数量级改善。

用法：
  python tools/stage_new/dp_seam_baseline.py --manifest data/reports_v2/split.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def _load_gray(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("L"), dtype=np.float64)


def _smooth1d(a: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return a
    w = np.ones(k) / k
    return np.convolve(a, w, mode="same")


def dp_seam(grad: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """经典 seam：逐列累加梯度，允许每列移动 -1/0/+1 行，平滑项惩罚跳变。

    grad: (H, W) 越大表示越可能是边界（这里传入 |∂I/∂y|）
    lam:  平滑权重，越大越贴合水平直线
    返回每列选中的行号
    """
    H, W = grad.shape
    # cost = 负梯度（最小化代价 => 走最大梯度）
    cost = -grad / (grad.max() + 1e-9)
    acc = cost[:, 0].copy()
    back = np.zeros((H, W), dtype=np.int32)
    for x in range(1, W):
        prev = acc
        best = np.full(H, -np.inf)
        arg = np.zeros(H, dtype=np.int32)
        # d = 当前行号 - 上一列行号
        for d in (-1, 0, 1):
            cand = np.full(H, -np.inf)
            if d == 0:
                cand = prev.copy()
            elif d == -1:
                # 当前行 r 来自上一列的 r+1
                cand[: max(0, H - 1)] = prev[1:]
            else:
                # 当前行 r 来自上一列的 r-1
                cand[1:] = prev[: H - 1]
            cand = cand - lam * abs(d)
            better = cand > best
            best = np.where(better, cand, best)
            arg = np.where(better, d, arg)
        acc = best + cost[:, x]
        acc = np.where(np.isfinite(best), acc, -np.inf)
        back[:, x] = arg
    if not np.any(np.isfinite(acc)):
        return np.zeros(W, dtype=np.int32)
    y = np.zeros(W, dtype=np.int32)
    y[-1] = int(np.nanargmax(acc))
    for x in range(W - 1, 0, -1):
        nxt = y[x] + back[y[x], x]
        y[x - 1] = int(np.clip(nxt, 0, H - 1))
    return y


def run(manifest: Path, lam: float, band_r: int) -> dict:
    items = [json.loads(l) for l in manifest.open(encoding="utf-8") if l.strip()]
    items = [r for r in items if r.get("window") and r.get("crop")]

    res_b1, res_b2 = [], []
    for r in items:
        p = ROOT / r["path"]
        if not p.exists():
            p = ROOT / "data" / r["path"]
        if not p.exists():
            continue
        try:
            im = _load_gray(p)
        except Exception:
            continue
        x0, y0, x1, y1 = r["window"]
        win = im[y0:y1, x0:x1]
        if win.size == 0:
            continue
        H, W = win.shape
        gt_top = int(r["crop"][0]) - y0

        gy = np.abs(np.gradient(win, axis=0))
        # 列方向平滑：抑制斑点噪声，让 seam 更稳
        gy = np.apply_along_axis(_smooth1d, 1, gy, 5)

        # B1：自由 seam
        y = dp_seam(gy, lam=lam)
        pred_b1 = int(np.median(y))

        # B2：带宽先验（上界实验，给 DP 开作弊）
        lo = max(0, gt_top - band_r)
        hi = min(H, gt_top + band_r + 1)
        y2 = dp_seam(gy[lo:hi], lam=lam)
        pred_b2 = int(np.median(y2)) + lo

        res_b1.append(abs(pred_b1 - gt_top))
        res_b2.append(abs(pred_b2 - gt_top))

    def summ(a):
        a = np.asarray(a, dtype=float)
        return {
            "n": int(a.size),
            "mean": round(float(a.mean()), 2),
            "median": round(float(np.median(a)), 2),
            "p90": round(float(np.percentile(a, 90)), 2),
        }

    return {
        "lam": lam,
        "band_r": band_r,
        "B1_free_seam": summ(res_b1),
        "B2_band_prior": summ(res_b2),
        "note": "误差单位为窗口内像素行；H 约 608",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/reports_v2/split.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--band-r", type=int, default=60)
    ap.add_argument("--out", default="runs/20261107_diagnostics/dp_baseline.json")
    args = ap.parse_args()

    mf = ROOT / args.manifest
    if args.split:
        tmp = ROOT / "runs/20261107_diagnostics" / f"_split_{args.split}.jsonl"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with mf.open(encoding="utf-8") as f, tmp.open("w", encoding="utf-8") as g:
            for l in f:
                if l.strip() and json.loads(l).get("split") == args.split:
                    g.write(l)
        mf = tmp

    out = run(mf, args.lam, args.band_r)
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
