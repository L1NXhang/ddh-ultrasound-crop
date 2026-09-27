"""把一个标注任务打包成"发给同学"的压缩包：工具 + 清单 + 只含该任务的图片。

同学收到后：解压 → 双击 标注工具.html → 选 images 文件夹 → 选 清单.jsonl → 开始标。

用法：python tools/build_package.py --manifest data/relabel713_A.jsonl --out D:/send/同学A.zip
"""
import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / 'product' / 'DDH裁剪标注工具.html'

README = """DDH 裁剪标注任务

一、怎么开始（不用安装任何东西）
  1. 把这个压缩包完整解压到桌面（不要直接在压缩包里打开）
  2. 双击「DDH裁剪标注工具.html」（用 Chrome 或 Edge 打开）
  3. 点「开始使用」
  4. 左侧①选解压出来的 images 文件夹；②选 清单.jsonl

二、每张图怎么做（5-8 秒）
  · 绿线：裁剪的上边界，已经按统一口径预置好，不对就用鼠标拖
    —— 判断标准：切在皮下组织层下方一点，把上面那层脂肪肌肉层保留进来
  · 右侧会显示「上边距 = N px」，落在 130-170 之间就是对的（绿色）
  · 蓝线：下边界，一般不用动
  · 按 Enter 保存并进入下一张

三、常用键
  Enter  保存并下一张          A  绿线回到预置位置
  拖动   调整绿线/蓝线          ↑↓ 微调 1 像素（Shift 是 10 像素）
  1 / 0  标 quality（可选）     K  跳过这张

四、结果保存在哪
  标完点左边的「立即下载结果 jsonl」，把那个文件发回给负责人即可。
  （如果一开始点了「指定结果文件」，它会自动追加写入，更省事）
  每按一次 Enter 都会存到浏览器本地，中途关掉也不会丢。

五、注意
  · 全部在本机处理，不联网、不上传，数据不会离开你的电脑
  · 每标 20 张瞄一眼「上边距」数字，如果发现自己一直在 180 以上，停下来问负责人
  · 拖错了不用担心，按 A 就回到预置位置
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--root', default='data')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for x in (ROOT / args.manifest).read_text(encoding='utf-8').splitlines() if x.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    missing, n_img = [], 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('DDH裁剪标注工具.html', HTML.read_text(encoding='utf-8'))
        z.writestr('使用说明.txt', README)
        z.writestr('清单.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        for r in rows:
            p = ROOT / args.root / r['path']
            if p.exists():
                z.write(p, 'images/' + p.name)
                n_img += 1
            else:
                missing.append(r['id'])
    print(f'已打包 → {out}（{out.stat().st_size/1024/1024:.1f} MB）')
    print(f'  图片 {n_img}/{len(rows)} 张' + (f'，缺 {len(missing)} 张：{missing[:5]}' if missing else ''))


if __name__ == '__main__':
    main()
