"""Refresh distribution notices and checksums while retaining the tested application binaries."""
import argparse
import json
import zipfile
from pathlib import Path

from build_portable import ROOT, clean_inventory, licenses
from studio_project import file_hash


def refresh(folder, archive, build):
    release = (ROOT.parent/'release').resolve()
    if not folder.resolve().is_relative_to(release) or not archive.resolve().is_relative_to(release):
        raise ValueError('Only generated release artifacts may be refreshed')
    manifest_path = folder/'release_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    executable_files = {row['path']: row['sha256'] for row in manifest['files']
                        if row['path'].endswith(('.exe', '.dll', '.pyd', '.js', '.html', '.pt'))}
    licenses(folder/'ddh_multidevice_v2/THIRD_PARTY_LICENSES', build)
    files = clean_inventory(folder)
    for row in files:
        if row['path'] in executable_files and row['sha256'] != executable_files[row['path']]:
            raise ValueError('A tested application binary changed')
    manifest.update(files=files, file_count=len(files), bytes=sum(row['bytes'] for row in files))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    temporary = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                zipped.write(path, folder.name+'/'+path.relative_to(folder).as_posix())
    with zipfile.ZipFile(temporary) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('ZIP CRC verification failed')
    temporary.replace(archive)
    result = {'folder': str(folder), 'zip': str(archive), 'zip_bytes': archive.stat().st_size,
              'zip_sha256': file_hash(archive), 'files': len(files)+1, 'tested_binaries_unchanged': True}
    (build/'release_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    refresh(args.folder.resolve(), args.archive.resolve(), args.build.resolve())
