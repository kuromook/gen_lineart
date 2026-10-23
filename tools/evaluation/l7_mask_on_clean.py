"""L7: does Track J's instrument hold up on cleanly separated data?

User's direction (2026-10-24): validate J's assets on data where line and fill
are actually separated, and measure the false-positive rate.

Ground truth here is the AUTHOR'S fill (production layers with keep4>=0.95),
not the radius-4 mask. Registered quantities, per stratum, pixel-pooled:

    precision = |mask AND truth| / |mask|      <- PRIMARY (1 - precision = FP rate)
    recall    = |mask AND truth| / |truth|
    mask_frac = |mask| / panel area
    true_frac = |truth| / panel area           (= the all_black arm's IoU)

Strata, split on L6's `leak` (how much of the true fill reached the raster),
never on anything the mask produces:
    CLEAN  leak < 0.10 and share_mixed < 0.10   (35 panels, pre-registered)
    LEAKY  leak >= 0.50                          (contrast)
The contrast is the point: if precision tracks the stratum, the mask's accuracy
is a property of whether fill leaked into its input, not of the mask.

Identity carried inside: clean_px recomputed here must equal L6's value.
Trivial arms on the true ground truth: all_black (IoU = true_frac) and blank
(region-F1), plus the connected-component count of the author's fill, against
J's "line art encloses about 1.2 regions per panel" which came from the mask.
"""
import csv, json, sys, collections, statistics as st
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0,'/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import fill_mask, ink_of
from region_metrics import regions, MIN_REGION_PX
KOMA=Path('/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826')
BASE=Path('/home/sh1/deepl/lineart/data/incoming_manuscript')
OUT=Path('/home/sh1/deepl/lineart-fill-region/results/l7_mask_on_clean'); OUT.mkdir(parents=True,exist_ok=True)

form={}
for r in csv.DictReader(open('results/l5_ako7_generation/layer_form.csv')):
    form[(r['work'],r['page'],r['name'],r['n_px'])]=float(r['keep4'])
l6={r['name']:r for r in csv.DictReader(open('results/l6_clean_subset/per_panel.csv'))}
man_rows={r['name']:r for r in csv.DictReader(open(KOMA/'manifest.csv'))}

sel=[]
for n,r in l6.items():
    sm,lk=float(r['share_mixed']),float(r['leak'])
    if sm<0.10 and lk<0.10: sel.append((n,'CLEAN',r))
    elif lk>=0.50:          sel.append((n,'LEAKY',r))
print(f'panels: CLEAN {sum(1 for s in sel if s[1]=="CLEAN")}  LEAKY {sum(1 for s in sel if s[1]=="LEAKY")}')

bypage=collections.defaultdict(list)
for n,strat,r6 in sel: bypage[(r6['work'],r6['pid'])].append((n,strat))

rows=[]; ident_ok=0
for (w,pid),items in sorted(bypage.items()):
    man=json.load(open(BASE/w/'fill'/'beta_manifest.json'))
    page=[p for p in man['pages'] if p['page']==pid][0]
    truth_full=None
    for b in page['beta']:
        im=cv2.imread(str(BASE/w/'fill'/b['file']), cv2.IMREAD_GRAYSCALE)<128
        if form.get((w,pid,b['name'],str(int(im.sum())))) is not None and \
           form[(w,pid,b['name'],str(int(im.sum())))]>=0.95:
            truth_full = im if truth_full is None else (truth_full|im)
    for n,strat in items:
        mr=man_rows[n]
        x0,y0,x1,y1=(int(mr[k]) for k in ('x0','y0','x1','y1'))
        g=cv2.imread(str(KOMA/'line'/n), cv2.IMREAD_GRAYSCALE)
        truth=np.zeros(g.shape,bool) if truth_full is None else truth_full[y0:y1,x0:x1]
        assert int(truth.sum())==int(l6[n]['clean_px']), f'identity failed on {n}'
        ident_ok+=1
        mask=fill_mask(ink_of(g))
        inter=int((mask&truth).sum())
        _,tids,_=regions(truth,MIN_REGION_PX)
        rows.append({'name':n,'work':w,'stratum':strat,'area':g.size,
                     'mask_px':int(mask.sum()),'truth_px':int(truth.sum()),'inter':inter,
                     'truth_regions':len(tids)})
print(f'identity (truth_px == L6 clean_px): {ident_ok} panels, all equal -> PASS\n')
with open(OUT/'per_panel.csv','w',newline='') as fh:
    wtr=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)

print(f'{"層":7} {"コマ":>4} {"precision":>10} {"FP率":>7} {"recall":>8} {"mask面積":>9} {"真ベタ面積":>11}')
for strat in ('CLEAN','LEAKY'):
    v=[r for r in rows if r['stratum']==strat]
    if not v: continue
    i=sum(r['inter'] for r in v); m=sum(r['mask_px'] for r in v)
    t=sum(r['truth_px'] for r in v); a=sum(r['area'] for r in v)
    print(f'{strat:7} {len(v):>4} {i/m:>10.4f} {1-i/m:>7.1%} {i/t:>8.4f} {100*m/a:>8.2f}% {100*t/a:>10.2f}%')
print()
for strat in ('CLEAN','LEAKY'):
    v=[r for r in rows if r['stratum']==strat]
    if not v: continue
    a=sum(r['area'] for r in v); t=sum(r['truth_px'] for r in v)
    reg=[r['truth_regions'] for r in v]
    print(f'{strat}: 自明な腕 all_black の IoU = 真ベタ面積比 = {t/a:.4f}')
    print(f'       作者のベタの連結成分(>=500px) 1コマあたり 中央 {st.median(reg):.1f} / 平均 {st.mean(reg):.1f} / 最大 {max(reg)}')
    print(f'       blank の region-F1 = {sum(1 for x in reg if x==0)/len(v):.4f}  (面が0のコマの割合)')
