"""Build standard Windows installers from an already verified portable distribution."""
import argparse
import json
import re
import subprocess
from pathlib import Path

from studio_project import file_hash
from build_portable import clean_inventory

ROOT = Path(__file__).resolve().parents[1]
RELEASE_LIMIT = 2*1024**3


def installer_defines(source, output, variant, version, split=False):
    if variant not in {'cpu', 'auto'} or not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('Invalid installer variant or version')
    if not (source/'DDH裁剪工具.exe').is_file():
        raise ValueError('Complete portable application folder is required')
    label = 'CPU' if variant == 'cpu' else 'CPU_GPU'
    return ['/DAppSource='+str(source.resolve()), '/DInstallerOutput='+str(output.resolve()),
            '/DAppVersion='+version, '/DVariant='+label, '/DDiskParts='+('1' if split else '0')]


def build(source, output, compiler, variant, version, split=False):
    manifest = json.loads((source/'release_manifest.json').read_text(encoding='utf-8'))
    if manifest['version'] != version or manifest['variant'] != variant:
        raise ValueError('Portable release version or variant does not match installer')
    if clean_inventory(source) != manifest['files']:
        raise ValueError('Application folder contains missing, modified or unexpected files')
    for row in manifest['files']:
        file = source/row['path']
        if file.stat().st_size != row['bytes'] or file_hash(file) != row['sha256']:
            raise ValueError('Portable distribution file changed: '+row['path'])
    output.mkdir(parents=True, exist_ok=True)
    args = installer_defines(source, output, variant, version, split)
    command = [str(compiler), '/Qp', *args, str(ROOT/'desktop/installer.iss')]
    subprocess.run(command, check=True, cwd=ROOT)
    label = 'CPU' if variant == 'cpu' else 'CPU_GPU'
    files = sorted(file for file in output.glob(f'DDH_Crop_Setup_{version}_{label}_x64*')
                   if file.suffix in {'.exe', '.bin'})
    if not files or any(file.stat().st_size >= RELEASE_LIMIT for file in files):
        raise ValueError('Installer asset exceeds the GitHub Release size limit')
    if not split and (len(files) != 1 or files[0].suffix != '.exe'):
        raise ValueError('Single-file installer must contain one complete EXE')
    result = {'version': version, 'variant': variant, 'files': [
        {'name': file.name, 'bytes': file.stat().st_size, 'sha256': file_hash(file)} for file in files]}
    (output/(label+'_installer_manifest.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--variant', choices=['cpu', 'auto'], default='cpu')
    parser.add_argument('--version', default='0.5.0')
    parser.add_argument('--split', action='store_true')
    args = parser.parse_args()
    build(args.source, args.out, args.compiler, args.variant, args.version, args.split)
