"""M9: the input A0 needs registered -- how to quantise a composited colour
page, and what area floor to use, measured on the 8 pilot previews.

A0 (registration 3.2, input corrected in 追補2 B4) is: quantise the composite
-> 8-connected components -> areas, no learning. Two things were deliberately
left unregistered until a colour raster existed:
    the quantisation method and its level
    the area floor, which 追補2 B4 requires to be a FRACTION of the page,
    because MIN_REGION_PX = 500 in pixels means a 4x different native-
    equivalent floor between a 1/4 and a 1/2 preview.

The previews ARE the only composites that exist (no native-size composite was
ever rendered). Their reduction differs per page and even within a work:
1/4, 1/2, 1/8 among these 8. Every row prints its own scale.

MEASURES ONLY. No arm is scored here; A0 is not run against any standard.
The cuts chosen from this sweep are registered in 追補5 and then frozen for
the test set.
"""
import json, sys
from pathlib import Path
import cv2, numpy as np

D = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/colour_pilot')
MANIFEST = Path('/home/sh1/deepl/lineart/inbox/colour_layers_manifest_pilot.json')
LEVELS = (2, 4, 8, 16, 32)          # levels per channel
FLOORS = (1e-5, 1e-4, 1e-3, 1e-2)   # fraction of page area


def quantise(bgr, levels):
    """Uniform quantisation to `levels` steps per channel. Stated plainly so
    the registration can name it: floor(v * levels / 256), no dithering, no
    palette fitting, no colour-space change (sRGB bytes as delivered)."""
    q = (bgr.astype(np.uint16) * levels // 256).astype(np.uint8)
    return q[..., 0].astype(np.int32) * levels * levels + \
           q[..., 1].astype(np.int32) * levels + q[..., 2].astype(np.int32)


def region_areas(label_img):
    """Areas of the 8-connected regions of EQUAL quantised colour.

    cv2.connectedComponents only labels a binary mask, and a colour page's
    areas touch, so a per-colour loop would cost one full pass per colour.
    This walks the image once with floodFill in mask-only, fixed-range mode
    (loDiff = upDiff = 0, so a region is exactly one quantised value) and
    lets the accumulated mask act as the barrier for later seeds."""
    h, w = label_img.shape
    img = label_img.astype(np.float32)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    seen = mask[1:-1, 1:-1]
    flags = 8 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (1 << 8)
    areas = []
    for y in range(h):
        row = seen[y]
        for x in range(w):
            if row[x]:
                continue
            cnt, _, _, _ = cv2.floodFill(img, mask, (x, y), 0, 0, 0, flags)
            areas.append(int(cnt))
    return np.array(areas, dtype=np.int64)


def main():
    pages = json.load(open(MANIFEST, encoding='utf-8'))['pages']
    print('page / preview size / scale, then components above each area floor\n')
    for p in pages:
        f = D / p['preview']['file']
        im = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if im is None:
            print(f'{p["page"]}: preview unreadable'); continue
        h, w = im.shape[:2]
        print(f'{p["page"]}  preview {w}x{h}  scale 1/{p["preview"]["縮小率"]:.0f}  '
              f'distinct colours (full 8-bit) {len(np.unique(quantise(im,256))):>7}')
        n_px = im.shape[0] * im.shape[1]
        print(f'    {"levels":>7} {"colours":>8} {"regions":>9} ' +
              ' '.join(f'{"n@"+format(fl,".0e"):>10}' for fl in FLOORS), flush=True)
        for lv in LEVELS:
            q = quantise(im, lv)
            ncol = len(np.unique(q))
            a = region_areas(q)
            counts = [int((a >= max(1, round(fl * n_px))).sum()) for fl in FLOORS]
            print(f'    {lv:>7} {ncol:>8} {len(a):>9} ' +
                  ' '.join(f'{c:>10}' for c in counts), flush=True)
        print(flush=True)
    print('No arm is scored in this file.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
