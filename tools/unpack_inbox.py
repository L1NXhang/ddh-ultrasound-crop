"""解压裁切压缩包 / 原图压缩包，按来源归位。

- crop_inbox 里的 <来源>.zip → data/crops_inbox/<来源>/（过滤 Mac 垃圾文件，压平中间层）
- 放进 images 的原图压缩包 → data/images/<压缩包名>/（保留内部结构）

用法：python tools/unpack_inbox.py            # 预演，只报告
      python tools/unpack_inbox.py --apply    # 真正解压
"""
import argparse
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEVENZ = r'C:\Users\13882\scoop\shims\7z.exe'
SOURCES = ['余', '吴', '张', '胡', '阳']
ARCHIVE_EXT = ('.zip', '.rar', '.7z')


def junk(name):
    return '__MACOSX' in name or Path(name).name.startswith('._') or Path(name).name == '.DS_Store'


def unpack_zip(zpath, dest, apply):
    added = 0
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            if info.is_dir() or junk(info.filename):
                continue
            if not info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            target = dest / Path(info.filename).name     # 压平中间层，只保留文件名
            if target.exists():
                continue
            if apply:
                dest.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, 'wb') as out:
                    out.write(src.read())
            added += 1
    return added


def unpack_rar(rpath, dest, apply):
    if not apply:
        out = subprocess.run([SEVENZ, 'l', '-slt', str(rpath)], capture_output=True)
        text = out.stdout.decode('gbk', errors='replace')
        return sum(1 for l in text.splitlines()
                   if l.startswith('Path = ') and l.lower().endswith(('.jpg', '.jpeg', '.png')))
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([SEVENZ, 'x', str(rpath), f'-o{dest}', '-y'], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout.decode('gbk', errors='replace')[-800:])
    return sum(1 for _ in dest.rglob('*.jpg'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    inbox = ROOT / 'data' / 'crops_inbox'
    images = ROOT / 'data' / 'images'

    print('=== 裁切压缩包 → crops_inbox/<来源>/ ===')
    for z in sorted(inbox.glob('*')):
        if z.suffix.lower() not in ARCHIVE_EXT or not z.is_file():
            continue
        src = z.stem
        if src not in SOURCES:
            print(f'  跳过 {z.name}: 名字 {src} 不在来源列表 {SOURCES} 里')
            continue
        dest = inbox / src
        before = len(list(dest.glob("*.jpg"))) + len(list(dest.glob("*.png"))) if dest.exists() else 0
        n = unpack_zip(z, dest, args.apply) if z.suffix.lower() == '.zip' else unpack_rar(z, dest, args.apply)
        after = (len(list(dest.glob("*.jpg"))) + len(list(dest.glob("*.png")))) if args.apply and dest.exists() else before
        print(f'  {z.name} → crops_inbox/{src}/：{"已解压" if args.apply else "将解压"} {n} 张'
              f'{f"（目录内 {after} 张）" if args.apply else ""}')

    print('\n=== 原图压缩包 → images/<压缩包名>/ ===')
    for z in sorted(images.glob('*')):
        if z.suffix.lower() not in ARCHIVE_EXT or not z.is_file():
            continue
        dest = images / z.stem
        n = unpack_rar(z, dest, args.apply) if z.suffix.lower() in ('.rar', '.7z') else unpack_zip(z, dest, args.apply)
        print(f'  {z.name} → images/{z.stem}/：{"已解压" if args.apply else "将解压"} {n} 张')
    if not args.apply:
        print('\n（预演，未写文件；确认后加 --apply）')


if __name__ == '__main__':
    main()
