"""Fail closed on private files and credential material before publishing source."""
import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

PRIVATE_ROOTS = {'data', 'runs', 'runs_v2', 'pilot', 'backup', '.venv', 'venv', 'release', 'dist', 'build'}
PRIVATE_SUFFIXES = {'.pt', '.pth', '.onnx', '.zip', '.7z', '.bin', '.exe', '.dll', '.pem', '.key',
                    '.dcm', '.dicom', '.png', '.jpg', '.jpeg', '.bmp', '.pdb'}
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')


def credential_table(text):
    column = None
    for line in text.splitlines():
        if not line.strip().startswith('|'):
            column = None
            continue
        cells = [value.strip().strip('`') for value in line.split('|')]
        headers = [i for i, value in enumerate(cells) if value.lower() in {'密码', 'password'}]
        if headers:
            column = headers[0]
            continue
        if column is None or column >= len(cells):
            continue
        value = cells[column]
        if value and not re.fullmatch(r'[-: ]+', value) and not any(marker in value for marker in ('<', '密码', '提供', 'placeholder')):
            return True
    return False


def audit_files(root, names):
    rejected = []
    for name in names:
        path = PurePosixPath(name)
        private = path.is_absolute() or '..' in path.parts or path.parts[0] in PRIVATE_ROOTS or \
            path.suffix.lower() in PRIVATE_SUFFIXES or path.name.startswith('.env') or \
            path.suffix == '.jsonl' and name != 'annotations.example.jsonl'
        if private:
            rejected.append({'file': name, 'reason': 'private data or generated binary'})
            continue
        file = root/name
        if not file.is_file():
            rejected.append({'file': name, 'reason': 'missing file'})
            continue
        if file.stat().st_size > 10*1024*1024:
            rejected.append({'file': name, 'reason': 'unexpected source file size'})
            continue
        if file.suffix.lower() in {'.md', '.py', '.js', '.html', '.css', '.json', '.yml', '.yaml', '.sh', '.txt'}:
            content = file.read_text(encoding='utf-8')
            if SECRET.search(content) or file.suffix == '.md' and credential_table(content):
                rejected.append({'file': name, 'reason': 'possible credential material'})
    return {'checked': len(names), 'rejected': rejected, 'passed': not rejected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--include-untracked', action='store_true')
    args = parser.parse_args()
    command = ['git', 'ls-files', '-z']
    names = subprocess.check_output(command, cwd=args.root).decode('utf-8').split('\0')
    if args.include_untracked:
        names += subprocess.check_output(command+['--others', '--exclude-standard'], cwd=args.root).decode('utf-8').split('\0')
    result = audit_files(args.root, sorted({name for name in names if name}))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
