"""Load the existing crop model lazily, keeping inference local."""
import threading
import logging
from pathlib import Path
from types import SimpleNamespace

from studio_project import file_hash
from crop_policy import DEFAULT_MARGIN, FINAL_CROP_POLICY, validate_margin


class ModelRunner:
    def __init__(self, checkpoint, device='auto', margin=DEFAULT_MARGIN):
        self.checkpoint, self.requested_device = Path(checkpoint), device
        self.margin = validate_margin(margin)
        self.device = '尚未加载'
        self.metadata = {'checkpoint': str(self.checkpoint), 'model_sha256': file_hash(self.checkpoint),
                         'window_algorithm': 'detect_window/corridor', 'margin': self.margin,
                         'crop_policy': FINAL_CROP_POLICY}
        self.lock = threading.Lock()
        self.tool = None

    def load(self):
        if self.tool is not None:
            return
        from studio_components import ensure_components, torch_available
        ensure_components(Path(__file__).resolve().parents[1])
        if not torch_available():
            raise RuntimeError('推理组件未安装：请点右上角「组件」先安装推理组件')
        import torch
        import crop_app
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if self.requested_device == 'auto' else self.requested_device
        try:
            self.load_on(crop_app, self.device)
        except RuntimeError:
            if self.requested_device != 'auto' or self.device != 'cuda':
                raise
            self.fallback(crop_app)

    def load_on(self, tool, device):
        tool.load_models(SimpleNamespace(device=device, crop_checkpoint=str(self.checkpoint)))
        tool.STATE['margin'] = self.margin
        self.tool, self.device = tool, device

    def fallback(self, tool):
        logging.warning('CUDA prediction unavailable; switching to CPU', exc_info=True)
        self.load_on(tool, 'cpu')
        self.metadata = {**self.metadata, 'device_note': 'GPU 暂不可用，已切换为 CPU'}

    def crop(self, path, box):
        try:
            return self.tool.predict_crop(path, box)
        except RuntimeError:
            if self.requested_device != 'auto' or self.device != 'cuda':
                raise
            self.fallback(self.tool)
            return self.tool.predict_crop(path, box)

    def predict(self, path, window=None):
        from detect_window import detect
        with self.lock:
            self.load()
            detected = detect(path) if window is None else {'box': window, 'flags': []}
            box = detected['box']
            result = {'window': box, 'flags': detected['flags'], 'crop_box': None}
            if box:
                prediction = self.crop(path, box)
                result.update(crop_box=[box[0], prediction['crop'][0], box[2], prediction['crop'][1]],
                              bounds_std=prediction['std'])
            return result
