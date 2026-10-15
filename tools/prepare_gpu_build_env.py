"""Reuse installed libraries in a separate build environment without changing training/CPU setups."""
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_MODULES = {'torch', 'torchvision', 'torchgen', 'functorch'}


def gpu_asset(name):
    return name in GPU_MODULES or name.startswith(('torch-', 'torchvision-'))


if __name__ == '__main__':
    cpu = ROOT/'data/portable_build/env/Lib/site-packages'
    gpu = ROOT/'.venv/Lib/site-packages'
    destination = ROOT/'data/portable_gpu_build/env/Lib/site-packages'
    if not (destination.parents[1]/'Scripts/python.exe').is_file():
        raise ValueError('Create the isolated GPU build environment first')
    if not destination.resolve().is_relative_to((ROOT/'data/portable_gpu_build').resolve()):
        raise ValueError('Unexpected build target')
    for source, accept in ((cpu, lambda name: not gpu_asset(name)), (gpu, gpu_asset)):
        for path in source.iterdir():
            if path.name == '__pycache__' or not accept(path.name):
                continue
            target = destination/path.name
            if path.is_dir():
                shutil.copytree(path, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            else:
                shutil.copy2(path, target)
            print('Prepared '+path.name, flush=True)
    python = destination.parents[1]/'Scripts/python.exe'
    subprocess.run([str(python), '-c', "import torch,torchvision,PyInstaller; print(torch.__version__,torchvision.__version__,PyInstaller.__version__); assert torch.version.cuda=='12.1'"], check=True)
