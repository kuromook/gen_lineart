"""IC-L1: does the koma crop box, applied to a page raster at page coordinates,
reproduce the panel raster? Known-answer cell: page0118, the one ako6 page for
which a page-level line raster exists. The shift rows establish that the test
has resolution -- without them an agreement number proves nothing."""
import csv, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0,'/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import ink_of

KOMA = Path('/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826')
PAGE = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/part_layers/018_2018_ako6_ako6_page0118_line.jpg')
page = cv2.imread(str(PAGE), cv2.IMREAD_GRAYSCALE)
print('page raster shape (h,w):', page.shape, '  manifest canvas is 7016x6543 (w,h)')

rows = [r for r in csv.DictReader(open(KOMA/'manifest.csv'))
        if r['pid']=='018_2018_ako6__ako6__page0118']
def iou(a,b):
    u=(a|b).sum()
    return float((a&b).sum()/u) if u else None

print(f'\n{"panel":>6} {"size":>12}  ' + '  '.join(f'd={d:+d}' for d in (0,2,5,10,25)))
for r in rows:
    x0,y0,x1,y1 = (int(r[k]) for k in ('x0','y0','x1','y1'))
    g = cv2.imread(str(KOMA/'line'/r['name']), cv2.IMREAD_GRAYSCALE)
    gi = ink_of(g)
    out=[]
    for d in (0,2,5,10,25):
        crop = page[y0+d:y1+d, x0+d:x1+d]
        if crop.shape != g.shape: out.append('  shape'); continue
        out.append(f'{iou(gi, ink_of(crop)):.3f}')
    print(f'{r["panel_index"]:>6} {g.shape[1]}x{g.shape[0]:<7}  ' + '  '.join(f'{v:>6}' for v in out))
