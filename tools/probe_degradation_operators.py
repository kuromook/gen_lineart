"""下絵らしさの落とし方の候補を1枚ずつ作って、原寸で並べる(**探りで、収集ではない**)。"""
import csv, sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.morphology import skeletonize

ROOT = Path("/home/sh1/deepl/lineart-aesthetic-judge")
GT = Path("/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line")
ROUGH = ROOT / "results/trackc_judge_20261004/stroke_projected/rough"
INK = 242
rng = np.random.default_rng(20261107)

def stats(g):
    m = g < INK
    sk = skeletonize(m); L = float(sk.sum())
    if L < 50: return dict(ink=float(m.mean()), comp=np.nan, junc=np.nan)
    lab, n = ndimage.label(m, structure=np.ones((3,3),int))
    nb = ndimage.convolve(sk.astype(np.uint8), np.ones((3,3),np.uint8), mode="constant") - 1
    return dict(ink=float(m.mean()), comp=n/L*100, junc=float(((nb>=3)&sk).sum())/L*100)

def fragment(g, n_cuts=420, r=2):
    """**断片化**: 線の上に小さな消しを無作為に置き、続いた線を切れ切れにする。"""
    out = g.copy(); m = g < INK
    ys, xs = np.nonzero(skeletonize(m))
    if len(ys) == 0: return out
    idx = rng.choice(len(ys), size=min(n_cuts, len(ys)), replace=False)
    im = Image.fromarray(out); d = ImageDraw.Draw(im)
    for i in idx:
        d.ellipse([xs[i]-r, ys[i]-r, xs[i]+r, ys[i]+r], fill=255)
    return np.asarray(im, float)

def overdraw(g, k=3, amp=4.0, light=0.45):
    """**重ね描き**: 同じ線を少しずつずらして薄く何本も重ねる(試行線)。"""
    base = (255.0 - g) / 255.0
    acc = base * 1.0
    for _ in range(k):
        dy, dx = rng.normal(0, amp, 2)
        th = rng.normal(0, 1.2)
        im = Image.fromarray((base*255).astype(np.uint8)).rotate(th, resample=Image.BILINEAR, fillcolor=0)
        sh = np.asarray(im, float)/255.0
        sh = ndimage.shift(sh, (dy, dx), order=1, mode="constant", cval=0.0)
        acc = np.maximum(acc, sh*light)
    return 255.0 - np.clip(acc, 0, 1)*255.0

def wobble(g, amp=2.2, smooth=22):
    """**ゆらぎ**: 線の通り道をゆっくり揺らす(手が決まっていない)。"""
    h, w = g.shape
    dy = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), smooth) 
    dx = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), smooth)
    dy = dy/ (np.abs(dy).max()+1e-9) * amp; dx = dx/(np.abs(dx).max()+1e-9) * amp
    yy, xx = np.mgrid[0:h, 0:w]
    return ndimage.map_coordinates(g, [yy+dy, xx+dx], order=1, mode="nearest").astype(float)

def width_jitter(g, n=36, smooth=26):
    """**太さのむら**: 場所によって線を太らせたり細らせたりする。"""
    h, w = g.shape
    f = ndimage.gaussian_filter(rng.normal(0,1,(h,w)), smooth); f = f/(np.abs(f).max()+1e-9)
    ink = (255.0-g)/255.0
    thick = ndimage.grey_dilation(ink, size=(3,3)); thin = ndimage.grey_erosion(ink, size=(3,3))
    a = np.clip((f+1)/2, 0, 1)
    out = thin*(1-a) + thick*a
    return 255.0 - np.clip(out,0,1)*255.0

tile = sys.argv[1] if len(sys.argv) > 1 else "lineart_005_006"
g = np.asarray(Image.open(GT/f"{tile}.jpg").convert("L").resize((480,480)), float)
r = np.asarray(Image.open(ROUGH/f"{tile}_out.png").convert("L"), float)
cands = [("GT(あなたが使えると答えた側)", g),
         ("ラフ = 下絵(使えないと答えた側)", r),
         ("(a) 断片化 —— 続いた線を切れ切れに", fragment(g)),
         ("(b) 重ね描き —— 試行線を薄く何本も", overdraw(g)),
         ("(c) ゆらぎ —— 通り道を揺らす", wobble(g)),
         ("(d) 太さのむら —— 太い所と細い所", width_jitter(g))]
print(f"{'':34s} {'墨の割合':>8s} {'断片/長さ':>10s} {'分岐/長さ':>10s}")
for name, x in cands:
    s = stats(x)
    print(f"{name:34s} {s['ink']:>8.4f} {s['comp']:>10.3f} {s['junc']:>10.3f}")
W = 480; pad = 6; lab = 24
cols, rowsn = 3, 2
out = Image.new("L", (cols*W + (cols+1)*pad, rowsn*(W+lab) + (rowsn+1)*pad), 255)
d = ImageDraw.Draw(out)
F = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 15)
for i, (name, x) in enumerate(cands):
    cx, cy = i % cols, i // cols
    X = pad + cx*(W+pad); Y = pad + cy*(W+lab+pad)
    d.text((X, Y+2), name, fill=0, font=F)
    out.paste(Image.fromarray(np.clip(x,0,255).astype(np.uint8)), (X, Y+lab))
p = ROOT/"results/judge_damage_ladder_20261107/operator_candidates_native.png"
out.save(p); print("\n", p, out.size)
