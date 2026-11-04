"""Feasibility survey for angle B (windows that contain fill), 2026-11-04.

Counts only: how many 480 windows on each side carry fill, where fill is ink
whose distance to paper is >= 8 px (the IC5 split). No axis value is computed.
Author side: A1 subregions at native scale, stride 240, support >= 0.85 (the
tiler default), ink gate 0.011992. External side: the registered 5 windows of
the seed2 1024px images. Usage: fill_window_survey.py a1|ext
"""
import csv, glob, collections, sys
import numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS=None

from pathlib import Path
R=Path('/home/sh1/deepl/lineart'); X=Path('/home/sh1/deepl/lineart-external-standard')
T=480; GATE=0.011992
def split(gray, sup=None):
    ink=gray<128
    if sup is not None: ink&=sup
    d=cv2.distanceTransform(ink.astype(np.uint8),cv2.DIST_L2,3)
    n=ink.sum(); return n/ink.size, (ink&(d>=8)).sum()/max(n,1)
TH=(0.02,0.05,0.10,0.20)
def report(name, shares, nsrc, nsrc_hit):
    s=np.array(shares)
    print(name, 'windows', len(s), ' '.join(f'fill>={t}:{(s>=t).sum()}' for t in TH), 'sources with >=0.10:', nsrc_hit, '/', nsrc, flush=True)
side=sys.argv[1]
if side=='a1':
    for w in ['ako5ver2','fitness','gakuen','hamlabi','housei']:
        d=glob.glob(str(R/f'dataset/regions_{w}_koma_subregions_*_masked'))[0]
        sh=[]; hit=0; rows=list(csv.DictReader(open(d+'/manifest.csv')))
        for r in rows:
            g=np.asarray(Image.open(r['masked_line_path']).convert('L'))
            m=np.asarray(Image.open(R/r['valid_mask_path']).convert('L'))>127
            if m.shape!=g.shape: continue
            H,W=g.shape; h=False
            for y in range(0,max(H-T,0)+1,240):
                for x in range(0,max(W-T,0)+1,240):
                    if H<T or W<T: continue
                    sm=m[y:y+T,x:x+T]
                    if sm.mean()<0.85: continue
                    ir,fs=split(g[y:y+T,x:x+T],sm)
                    if ir<GATE: continue
                    sh.append(fs); h|=fs>=0.10
            hit+=h
        report('a1_'+w,sh,len(rows),hit)
else:
    WIN=[(272,272),(0,0),(544,544),(0,544),(544,0)]
    for k in ['gen_animagine_xl_31_seed2','gen_illustrious_xl_v01_seed2']:
        sh=[];hit=0;ps=sorted((X/'data/external_standard'/k).glob('*.png'))
        for p in ps:
            g=np.asarray(Image.open(p).convert('L')); h=False
            for y,x in WIN:
                ir,fs=split(g[y:y+T,x:x+T])
                if ir<GATE: continue
                sh.append(fs); h|=fs>=0.10
            hit+=h
        report(k,sh,len(ps),hit)
