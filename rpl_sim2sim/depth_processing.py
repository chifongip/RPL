"""NumPy equivalent of Torch's antialiased bicubic downsampling (align_corners=False)."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def _weights(source, target):
    scale = source / target
    support = max(scale, 1.0)
    centers = (np.arange(target) + 0.5) * scale - 0.5
    x = np.abs((np.arange(source)[None] - centers[:, None]) / support)
    # Antialiasing uses the Keys cubic kernel with a=-0.5, as Torch/Pillow.
    values = np.where(x < 1, 1.5 * x**3 - 2.5 * x**2 + 1, np.where(x < 2, -0.5 * x**3 + 2.5 * x**2 - 4 * x + 2, 0))
    values /= values.sum(axis=1, keepdims=True)
    return values.astype(np.float32)


def resize_bicubic_antialias(image, height, width):
    return _weights(image.shape[0], height) @ image.astype(np.float32) @ _weights(image.shape[1], width).T
