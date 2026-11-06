"""Wording survey for the close-up angle: face heights only (no axis).

Runs the validated detector (face_scale_probe.FaceDetector, knobs untouched)
on each whole 1024 image and prints, per wording x model, how many images
have a detected face and the median box height. Images with no detected
face are counted, never dropped silently.
"""

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_scale_probe import FaceDetector, load_gray  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/external_closeup_survey_20261106")
det = FaceDetector()
for d in sorted(root.glob("gen_*")):
    files = sorted(d.glob("*.png"))
    hs, miss = [], []
    for f in files:
        b = det(load_gray(f))
        if b:
            hs.append(float(np.median([q[3] - q[1] for q in b])))
        else:
            miss.append(f.name)
    q = " / ".join(f"{v:.0f}" for v in np.percentile(hs, [0, 50, 100])) if hs else "-"
    print(f"{d.name:32s} images {len(files)}  face {len(hs)}  height min/med/max {q}  >=600: {sum(h >= 600 for h in hs)}")
    for m in miss:
        print(f"    no face: {m}")
