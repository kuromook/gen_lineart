"""The one registered normalization, shared by both sides.

doc/prereg_external_standard_20261030.md, section "正規化". The white point is
equalized on both sides (the user's ruling of 2026-10-30): without it the
external paper (mode 149-216) falls entirely into the tool's faint band
(128-220) while the author's paper (254) does not, and the three faint axes
plus background_ratio split at AUC 1.0 on paper colour rather than on drawing.

Order matters and is fixed here: BOTH sides are read back from a JPEG that
went through the same quality-95 path, and the white point is applied AFTER
that read. The author's tiles are already in that form and are not re-encoded
(the user's ruling 6: do not redo the known cell); the external side is
written to JPEG once when its windows are cut.
"""

import numpy as np
from PIL import Image

JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0
INK_THRESHOLD = 128
INK_GATE = 0.011992  # the author pool's own minimum, section 正規化-6


def paper_level(gray):
    """The mode of the image -- what this tool calls the paper."""
    return int(np.bincount(gray.ravel(), minlength=256).argmax())


def equalize_white_point(gray):
    """v' = clip(round(v * 255 / m)), m = the paper level. Same formula both sides."""
    m = paper_level(gray)
    if m <= 0:
        return gray, m
    scaled = np.rint(gray.astype(np.float32) * (255.0 / m))
    return np.clip(scaled, 0, 255).astype(np.uint8), m


def ink_ratio(gray):
    return float((gray < INK_THRESHOLD).mean())


def load_normalized(path):
    """Read one tile (either side) and apply the registered normalization."""
    gray = np.asarray(Image.open(path).convert("L"))
    return equalize_white_point(gray)


def save_jpeg(gray, path):
    Image.fromarray(gray, mode="L").save(
        path, format="JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING
    )
