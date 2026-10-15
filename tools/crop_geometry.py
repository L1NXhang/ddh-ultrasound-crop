"""Validate original-image coordinates at the local tool's API boundary."""
import math


def _coordinates(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{name} 必须包含 {length} 个原图像素坐标')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or int(v) != v for v in value):
        raise ValueError(f'{name} 必须使用有限的整数像素坐标')
    return [int(v) for v in value]


def validate_window(box, size):
    x0, y0, x1, y1 = _coordinates(box, 4, '窗口')
    width, height = size
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('窗口必须位于原图内，且左右、上下边界顺序正确')
    return [x0, y0, x1, y1]


def validate_crop(crop, window):
    top, bottom = _coordinates(crop, 2, '裁剪线')
    if not window[1] <= top < bottom <= window[3]:
        raise ValueError('裁剪线必须位于窗口内，且上边界小于下边界')
    return [top, bottom]


def modification_details(model_window, model_crop, window, crop):
    window_delta = [a-b for a, b in zip(window, model_window)] if model_window else None
    crop_delta = [a-b for a, b in zip(crop, model_crop)] if model_crop else None
    changed_window = window != model_window
    changed_crop = crop != model_crop
    return {'modified': changed_window or changed_crop,
            'window_modified': changed_window, 'crop_modified': changed_crop,
            'window_delta': window_delta, 'crop_delta': crop_delta}
