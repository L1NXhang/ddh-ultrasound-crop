"""清点 crop_inbox 里的裁切压缩包，以及 images 里新加的原图压缩包。"""
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEVENZ = r'C:\Users\13882\scoop\shims\7z.exe'
SEP = chr(92)   # 反斜杠，避免转义麻烦


def rar_list(path):
    out = subprocess.run([SEVENZ, 'l', '-slt', str(path)], capture_output=True)
    text = out.stdout.decode('gbk', errors='replace')
    paths = [l[7:].strip() for l in text.splitlines() if l.startswith('Path = ')]
    return [p for p in paths[1:] if p.lower().endswith(('.jpg', '.jpeg', '.png'))]


def main():
    print('=== crop_inbox 里的裁切压缩包 ===')
    for z in ['余', '吴', '胡', '阳']:
        p = ROOT / 'data' / 'crops_inbox' / f'{z}.zip'
        if not p.exists():
            continue
        with zipfile.ZipFile(p) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
            junk = [n for n in names if '__MACOSX' in n or Path(n).name.startswith('._')]
            real = [n for n in names if n not in junk]
        tops = Counter(SEP.join(n.split('/')[:-1]) for n in real)
        print(f'\n{z}.zip: 图片 {len(names)} 个 | Mac 垃圾 {len(junk)} 个 | 有效 {len(real)} 个')
        for d, c in tops.most_common(8):
            print(f'    {d or "(根目录)"}: {c}')
        print(f'    样例: {real[0]}')

    print('\n=== 新原图 Ⅱc+Ⅱa-b.rar ===')
    imgs = rar_list(ROOT / 'data' / 'images' / 'Ⅱc+Ⅱa-b.rar')
    print(f'图片总数 {len(imgs)}')
    print('顶层目录:', dict(Counter(p.split(SEP)[0] for p in imgs)))
    for d, c in sorted(Counter(SEP.join(p.split(SEP)[:2]) for p in imgs).items()):
        print(f'    {d}: {c}')
    print('样例文件名:', imgs[0].split(SEP)[-1])


if __name__ == '__main__':
    main()
