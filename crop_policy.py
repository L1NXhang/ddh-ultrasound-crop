"""Shared inference policy: model bounds describe the final crop by default."""
import math
from numbers import Real

DEFAULT_MARGIN = 0.0
FINAL_CROP_POLICY = 'final_bounds/v1'


def validate_margin(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value < .5:
        raise ValueError('额外外扩比例必须是大于等于0、小于0.5的有限数值')
    return float(value)


def apply_margin(bounds, margin=DEFAULT_MARGIN):
    margin = validate_margin(margin)
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError('模型必须给出窗口内的上下归一化边界')
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in bounds):
        raise ValueError('模型边界必须是有限数值')
    top, bottom = bounds
    if not 0 <= top < bottom <= 1:
        raise ValueError('模型边界必须满足0 <= top < bottom <= 1')
    return [max(0., float(top) - margin), min(1., float(bottom) + margin)]
