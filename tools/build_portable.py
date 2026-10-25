"""Build a data-free Windows CPU desktop distribution from explicit application assets."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

from build_desktop import build as build_host
from studio_project import file_hash

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT/'data/portable_build'
VERSION = '0.7.0'
NAME = f'DDH_Crop_{VERSION}_Windows_x64_CPU'
AUTO_VERSION = '0.7.0'
TRAIN_VERSION = '0.7.0'
WEB_ASSETS = ('studio.html', 'studio.css', 'studio.js', 'studio_session.js', 'desktop.js', 'review_client.js', 'collaboration.js')
PREFIXES = ('ddh_multidevice_v2/backend/', 'ddh_multidevice_v2/desktop/runtime/',
            'ddh_multidevice_v2/desktop/WebView2Runtime/', 'ddh_multidevice_v2/THIRD_PARTY_LICENSES/',
            'ddh_multidevice_v2/components/')
EXACT = {'DDH裁剪工具.exe', 'DDH裁剪工具.exe.config', '先读我.txt', 'release_manifest.json',
         'ddh_multidevice_v2/portable.json', 'ddh_multidevice_v2/model_manifest.json',
         'ddh_multidevice_v2/runs_v2/crop.pt'} | {'ddh_multidevice_v2/tools/webapp/'+name for name in WEB_ASSETS}
BOOTSTRAP_ONLY = {'ddh_multidevice_v2/components.lock.json'}
OFFLINE_ONLY = {'ddh_multidevice_v2/desktop/WebView2Runtime/'}

README = '''DDH 图像裁剪工具 · Windows {label} v{version}

1. {bootstrap_note}
2. 选择图片文件夹，再选择结果保存位置。图片可以放在电脑其他目录，不需要放入软件目录。
3. 选择“逐张审核”或“批量裁剪”。首次加载模型需要数秒。
4. 红框是完整成像区，绿框是最终裁剪范围。拖动四边调整，Enter 保存并下一张。
5. 图标或文字进入成像区时，可向内调整绿框的左右边界，保留需要的图像内容。
6. 修改红框后按 R 重新预测。Ctrl+Z 撤销，Ctrl+Shift+Z 重做，P 切换成品预览（滚轮缩放、中键拖动、对比模式）。
7. 未确认调整保存为草稿。确认后导出 PNG 与 JSON；完成后可自选是否保存 ZIP。
8. 点击右上角“协作审核”可给组员设置账号和密码；组员通过浏览器审核，记录直接存回主机。
   局域网使用 IP 和端口；外网需另行配置加密组网或 HTTPS 代理。软件不会自动修改防火墙。
9. v0.7.0：推理组件按需下载——安装包只含程序与模型，首次启动时按本机环境下载 CPU 或 CUDA
   组件（每个文件都校验官方 sha256）；离线完整包仍包含全部组件，安装后无需联网。{extra}

本包不含项目图片、样例病例或本机审核数据。{contents}
支持 Windows 10/11 64 位，系统需具备 .NET Framework 4.8（较新的 Windows 已包含）。
此版模型自动预测完整窗口和上下边界，图标/文字的左右清理仍需人工审核。

设置、日志与网页缓存位于软件内 ddh_multidevice_v2/data/studio_runtime，请解压到本人可写的本地文件夹。
原图和裁剪结果位于你自己选择的目录，建议均选择软件文件夹之外的位置。
人工审核记录可交回负责人作为训练候选；软件不会自行重新训练或替换模型。
软件包需要整个文件夹一起移动，不能只复制一个 EXE。每个平台需要独立版本，此包不适用于 macOS。
'''


def allowed_path(name):
    if '\\' in name or ':' in name:
        return False
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or path.suffix.lower() in {'.jsonl', '.dcm', '.dicom'}:
        return False
    if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}:
        return False
    return name in EXACT or any(name.startswith(prefix) for prefix in PREFIXES)


def clean_inventory(folder):
    inventory = []
    for path in sorted(folder.rglob('*')):
        if not path.is_file():
            continue
        name = path.relative_to(folder).as_posix()
        if not allowed_path(name):
            raise ValueError('Unexpected file in portable application: '+name)
        if name != 'release_manifest.json':
            inventory.append({'path': name, 'bytes': path.stat().st_size, 'sha256': file_hash(path)})
    return inventory


def freeze(python, build=BUILD, exclude_torch=False):
    command = [str(python), '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir', '--console',
               '--name', 'DDHBackend', '--distpath', str(build/'frozen'), '--workpath', str(build/'freeze_work'),
               '--specpath', str(build), '--paths', str(ROOT), '--paths', str(ROOT/'tools'),
               '--exclude-module', 'tkinter', '--exclude-module', 'scipy', '--exclude-module', 'matplotlib',
               '--exclude-module', 'pandas', '--exclude-module', 'pytest', '--exclude-module', 'IPython']
    if exclude_torch:
        # 引导版：torch/torchvision 由用户机器按需下载（components/ 目录），不进 bundle
        command += ['--exclude-module', 'torch', '--exclude-module', 'torchvision']
    command += [str(ROOT/'tools/crop_studio.py')]
    subprocess.run(command, check=True, cwd=ROOT)
    return build/'frozen/DDHBackend'


def licenses(destination, env_dir):
    packages = Path(env_dir)/'Lib/site-packages'
    for info in sorted(packages.glob('*.dist-info')):
        folder = destination/info.name
        folder.mkdir(parents=True, exist_ok=True)
        for path in info.rglob('*'):
            if path.is_file() and (path.name.upper() in {'METADATA', 'NOTICE', 'NOTICE.TXT', 'COPYING', 'COPYRIGHT'} or 'license' in path.name.lower() or 'licenses' in path.parts):
                target = folder/path.relative_to(info)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    config = (Path(env_dir)/'pyvenv.cfg').read_text(encoding='utf-8')
    base = Path(next(line.split('=', 1)[1].strip() for line in config.splitlines() if line.startswith('home =')))
    python_license = next((base/name for name in ('LICENSE.txt', 'LICENSE') if (base/name).is_file()), None)
    if python_license:
        shutil.copy2(python_license, destination/'PYTHON_LICENSE.txt')


def package(destination, reuse_frozen=False, variant='cpu', make_archive=True):
    if variant not in {'cpu', 'auto', 'train', 'bootstrap'}:
        raise ValueError('Unknown inference variant')
    # bootstrap 与 cpu 共用 CPU 构建环境，但冻结产物分开（一个含 torch 一个不含）
    build = ROOT/'data/portable_bootstrap_build' if variant == 'bootstrap' else (
        BUILD if variant == 'cpu' else ROOT/'data/portable_gpu_build')
    env_dir = ROOT/'data/portable_bootstrap_build/env' if variant == 'bootstrap' else build/'env'
    if not (env_dir/'Scripts/python.exe').is_file():
        env_dir = BUILD/'env'                    # 引导版复用 CPU 环境的 site-packages
    version = VERSION if variant in ('cpu', 'bootstrap') else AUTO_VERSION
    label = {'cpu': 'CPU', 'auto': 'CPU_GPU', 'train': 'GPU_TRAIN', 'bootstrap': 'Bootstrap'}[variant]
    destination = destination.resolve()
    if destination.exists():
        raise ValueError('Release folder already exists; choose a new output folder')
    frozen = build/'frozen/DDHBackend' if reuse_frozen else freeze(env_dir/'Scripts/python.exe', build, exclude_torch=variant == 'bootstrap')
    if not (frozen/'DDHBackend.exe').is_file():
        raise ValueError('Frozen backend missing')
    provenance = json.loads((build/'webview/provenance.json').read_text(encoding='utf-8'))
    runtime = Path(provenance['runtime_folder'])
    app = destination/'ddh_multidevice_v2'
    app.mkdir(parents=True)
    build_host(destination/'DDH裁剪工具.exe')
    shutil.copytree(frozen, app/'backend')
    shutil.copytree(ROOT/'desktop/runtime', app/'desktop/runtime')
    if variant != 'bootstrap':                     # 引导版用系统 Evergreen WebView2
        shutil.copytree(runtime, app/'desktop/WebView2Runtime')
    for name in WEB_ASSETS:
        target = app/'tools/webapp'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/'tools/webapp'/name, target)
    if variant == 'bootstrap':                     # 组件清单：安装时按它下载并校验
        shutil.copy2(ROOT/'components.lock.json', app/'components.lock.json')
    (app/'runs_v2').mkdir()
    model = subprocess.check_output([str(env_dir/'Scripts/python.exe'), str(ROOT/'tools/prepare_runtime_model.py'),
                                     str(ROOT/'runs_v2/crop.pt'), str(app/'runs_v2/crop.pt')],
                                    cwd=ROOT, encoding='utf-8')
    model_meta = json.loads(model)
    model_meta['original_checkpoint_sha256'] = file_hash(ROOT/'runs_v2/crop.pt')
    model_meta['runtime_checkpoint_sha256'] = file_hash(app/'runs_v2/crop.pt')
    (app/'model_manifest.json').write_text(json.dumps(model_meta, indent=2), encoding='utf-8')
    (app/'portable.json').write_text(json.dumps({'version': version, 'platform': 'Windows x64',
                                               'inference': variant, 'contains_project_images': False}), encoding='utf-8')
    readme_kwargs = {'version': version, 'label': label,
                     'bootstrap_note': '完整解压压缩包到本地文件夹，双击“DDH裁剪工具.exe”。不要在压缩包里面直接运行。',
                     'extra': '', 'contents': '内含现有模型、CPU 推理环境和固定版 WebView2，不需要安装 Python，不需要独立显卡，裁剪过程中不需要联网。'}
    if variant == 'bootstrap':
        readme_kwargs.update({
            'label': '引导版',
            'bootstrap_note': '完整解压到本地文件夹后双击“DDH裁剪工具.exe”。首次启动会提示下载推理组件（CPU 或 CUDA，按你的显卡选择），'
                              '组件从国内镜像下载并逐文件校验 sha256，需要联网；下载完成前无法裁剪，界面其余功能不受影响。',
            'extra': '需要联网下载推理组件（约 200 MB 起）；离线用户请使用完整离线包。',
            'contents': '内含现有模型与程序框架；深度学习组件（torch）按需下载，浏览器组件使用系统自带的 WebView2。'})
    elif variant == 'auto':
        readme_kwargs.update({
            'label': '便携 CPU/GPU 自动版',
            'contents': '内含现有模型、CPU/CUDA 推理环境和固定版 WebView2，不需要安装 Python。'
                        '有兼容的 NVIDIA CUDA 显卡时自动使用 GPU，否则使用 CPU；GPU 加载或推理失败时自动切回 CPU。AMD/Intel 显卡使用 CPU。'})
    elif variant == 'train':
        readme_kwargs.update({
            'label': '便携 GPU+训练版',
            'contents': '内含现有模型、CPU/CUDA 推理环境、模型训练组件和固定版 WebView2，不需要安装 Python。'
                        '有兼容的 NVIDIA CUDA 显卡时自动使用 GPU，否则使用 CPU。'
                        '右上角「训练」可用已审核的标注清单在本机重新训练裁剪线模型（需要 NVIDIA 显卡）；只在您主动操作时运行。',
            'extra': ''})
    readme = README.format(**readme_kwargs).replace(
        '人工审核记录可交回负责人作为训练候选；软件不会自行重新训练或替换模型。',
        '人工审核记录可交回负责人作为训练候选。'+('' if variant != 'train' else '「训练」功能只在您主动操作时运行，不会自行重新训练或替换模型。'))
    (destination/'先读我.txt').write_text(readme, encoding='utf-8-sig')
    licenses(app/'THIRD_PARTY_LICENSES', env_dir)
    files = clean_inventory(destination)
    manifest = {'schema': 'ddh_portable_release/v1', 'version': version, 'variant': variant, 'files': files,
                'file_count': len(files), 'bytes': sum(row['bytes'] for row in files),
                'project_images_included': 0, 'annotation_files_included': 0,
                'model': model_meta, 'webview': {key: value for key, value in provenance.items() if key != 'runtime_folder'}}
    (destination/'release_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if not make_archive:
        result = {'folder': str(destination), 'zip': None, 'files': len(files)+1}
        (build/'release_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    archive = destination.parent/f'DDH裁剪工具_Windows_x64_{label}_v{version}.zip'
    if archive.exists():
        raise ValueError('ZIP already exists; preserve the previous artifact first')
    print('Compressing data-free application...', flush=True)
    temporary = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                zipped.write(path, destination.name+'/'+path.relative_to(destination).as_posix())
    with zipfile.ZipFile(temporary) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('ZIP CRC verification failed')
    temporary.replace(archive)
    result = {'folder': str(destination), 'zip': str(archive), 'zip_bytes': archive.stat().st_size,
              'zip_sha256': file_hash(archive), 'files': len(files)+1}
    (build/'release_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    parser.add_argument('--variant', choices=['cpu', 'auto', 'train', 'bootstrap'], default='cpu')
    parser.add_argument('--reuse-frozen', action='store_true')
    parser.add_argument('--no-archive', action='store_true')
    args = parser.parse_args()
    name = {'cpu': NAME, 'auto': f'DDH_Crop_{AUTO_VERSION}_Windows_x64_CPU_GPU',
            'train': f'DDH_Crop_{TRAIN_VERSION}_Windows_x64_GPU_TRAIN',
            'bootstrap': f'DDH_Crop_{VERSION}_Windows_x64_Bootstrap'}[args.variant]
    package(args.out or ROOT.parent/'release'/name, args.reuse_frozen, args.variant, not args.no_archive)
