import csv, json, itertools
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

S=Path('/tmp/claude-1000/-home-sh1-deepl-lineart-pair-signal/af198da1-466b-4398-8cb1-f231b99b5157/scratchpad/judge_pairs')
REC=Path('/home/sh1/deepl/lineart-aesthetic-judge/results/comparison_pairs_20260916')
W02=Path('/home/sh1/deepl/lineart-controlnet-sd15-refine/results/controlnet_lora_manga_consistency_w0.2_snapshot_probe_20260914')
COND=Path('/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning')
LIST=Path('/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family.txt')

# Three tone-matched candidates. The dark variants (control, w02_1000) are left
# out deliberately: their pairs are decided by paper tone, which the project can
# already compute, so they would buy 0.83-near-white preference, not judgement.
VARIANTS=[('preproc',  lambda t: COND/f'{t}.jpg', True),
          ('w02_7000', lambda t: W02/'step_7000'/f'{t}_out.png', False),
          ('w02_10580',lambda t: W02/'step_10580'/f'{t}_out.png', False)]
VNAMES=[v for v,_,_ in VARIANTS]

tiles=[Path(l.strip()).stem for l in open(LIST) if l.strip()]
tiles=[t for t in tiles if all(fn(t).exists() for _,fn,_ in VARIANTS)]
ink={}
with open(W02/'per_tile/step_7000.csv') as f:
    for r in csv.DictReader(f): ink[Path(r['sample']).stem]=float(r['gt_ink_ratio'])
tiles=[t for t in tiles if t in ink]
rng=np.random.default_rng(20260916)
chosen=[]
for bucket in np.array_split(np.array(sorted(tiles,key=lambda t:ink[t])),5):
    idx=rng.choice(len(bucket),size=16,replace=False)
    chosen += [str(bucket[i]) for i in sorted(idx)]
print(f'tiles: {len(chosen)} (16 from each GT-ink quintile, of {len(tiles)} eligible)')

(S/'img').mkdir(parents=True, exist_ok=True)
for t in chosen:
    sprite=Image.new('L',(480*len(VARIANTS),480),255)
    for i,(v,fn,inv) in enumerate(VARIANTS):
        im=Image.open(fn(t)).convert('L')
        if im.size!=(480,480): im=im.resize((480,480),Image.LANCZOS)
        if inv: im=ImageOps.invert(im)
        sprite.paste(im,(480*i,0))
    sprite.save(S/f'img/{t}.png', optimize=True)
total=sum(p.stat().st_size for p in (S/'img').glob('*.png'))
print(f'sprites: {len(chosen)} files, {total/1e6:.1f} MB')

pairs=[]
combos=list(itertools.combinations(range(len(VARIANTS)),2))
seq=[(t,a,b) for t in chosen for a,b in combos]
rng.shuffle(seq)
for i,(t,a,b) in enumerate(seq):
    l,r=(a,b) if rng.random()<0.5 else (b,a)
    pairs.append({'id':f'p{i:04d}','tile':t,'l':l,'r':r,'rep':''})
ridx=rng.choice(len(pairs),size=60,replace=False)
reps=[{'id':f'r{j:04d}','tile':pairs[k]['tile'],'l':pairs[k]['r'],'r':pairs[k]['l'],'rep':pairs[k]['id']}
      for j,k in enumerate(ridx)]
pos=sorted(rng.choice(range(len(pairs)//2,len(pairs)),size=60,replace=False))
for p,rw in zip(pos,reps): pairs.insert(p,rw)
print(f'pairs: {len(pairs)} ({len(reps)} repeats, sides swapped, placed in the back half)')

json.dump({'variants':VNAMES,'pairs':pairs}, open(S/'pairs.json','w'), separators=(',',':'))
REC.mkdir(parents=True,exist_ok=True)
with open(REC/'pairs.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['id','tile','left_variant','right_variant','repeat_of'])
    for p in pairs: w.writerow([p['id'],p['tile'],VNAMES[p['l']],VNAMES[p['r']],p['rep']])
json.dump({'variants':VNAMES,'tiles':chosen,'seed':20260916,
           'rationale':'tone-matched candidates only; dark variants excluded so pairs cannot be decided on paper tone'},
          open(REC/'design.json','w'), indent=1)
# verification montage
W=200
sheet=Image.new('RGB',(W*3,W*6),'white')
for r,t in enumerate([chosen[3],chosen[19],chosen[35],chosen[51],chosen[67],chosen[76]]):
    sp=Image.open(S/f'img/{t}.png')
    for c in range(3): sheet.paste(sp.crop((480*c,0,480*(c+1),480)).convert('RGB').resize((W,W)),(c*W,r*W))
sheet.save(REC/'staged_variants_check.png')
print('montage:', REC/'staged_variants_check.png', '| columns:', ' | '.join(VNAMES))
print('manifest:', REC/'pairs.csv')
