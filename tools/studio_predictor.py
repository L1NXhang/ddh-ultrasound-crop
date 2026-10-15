"""Load the existing crop model lazily, keeping inference local."""
import threading
import logging
from pathlib import Path
from types import SimpleNamespace

from studio_project import file_hash


class ModelRunner:
    def __init__(self, checkpoint, device='auto', margin=.02):
        self.checkpoint, self.requested_device = Path(checkpoint), device
        self.margin = margin
        self.device = '尚未加载'
        self.metadata = {'checkpoint': str(self.checkpoint), 'model_sha256': file_hash(self.checkpoint),
                         'window_algorithm': 'detect_window/corridor', 'margin': margin}
        self.lock = threading.Lock()
        self.tool = None

    def load(self):
        if self.tool is not None:
            return
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
