# -*- coding: utf-8 -*-
"""「膜の層」34 + 対照 8 の層の画と α のグラフを並べた見る用のページを作る(R3 の後、ユーザー 11-20 の依頼)。

入力: data/census/r_film_rgba_20261120/(抽出側の RGBA の層の画・頁・点検の表、追跡しない)
出力: results/r3/film_view/(縮小の画・原寸の切り出し・index.html。原稿の画素を含む。追跡しない)
採点はしない。使い方: ../lineart/venv/bin/python tools/r3_film_view.py   (PIL の在る venv)
"""
import csv, html, io, json, os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SRC = "data/census/r_film_rgba_20261120"
IMG = SRC + "/r_film_layers_rgba_20261120"
OUT = "results/r3/film_view"
LINE = "data/census/r_line/r_line_columns_v2_20261119.csv"
TW = 560      # 縮小の幅
CROP = 360    # 原寸の切り出しの辺


def alpha_map(a):
    """α 0 = 白 / α 1〜7 = 橙(薄い→濃い)/ α ≥ 8 = 青みの灰→黒。"""
    out = np.full(a.shape + (3,), 255, np.uint8)
    lo = (a >= 1) & (a <= 7)
    t = (a[lo].astype(np.float32) - 1) / 6
    out[lo] = np.stack([255 - 10 * t, 214 - 60 * t, 150 - 90 * t], -1).astype(np.uint8)
    hi = a >= 8
    t = (a[hi].astype(np.float32) - 8) / 247
    out[hi] = np.stack([150 - 140 * t, 160 - 145 * t, 200 - 170 * t], -1).astype(np.uint8)
    return out


def on_white(rgba):
    rgb = rgba[..., :3].astype(np.float32)
    al = rgba[..., 3:4].astype(np.float32) / 255
    return (rgb * al + 255 * (1 - al)).astype(np.uint8)


def best_crop(a, n=CROP):
    """α ≥ 8 の画素が最も多い n×n の窓(n/4 刻み)。"""
    h, w = a.shape
    if h <= n or w <= n:
        return 0, 0
    m = (a >= 8).astype(np.int32)
    ii = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, by, bx = -1, 0, 0
    for y in range(0, h - n + 1, n // 4):
        for x in range(0, w - n + 1, n // 4):
            s = ii[y + n, x + n] - ii[y, x + n] - ii[y + n, x] + ii[y, x]
            if s > best:
                best, by, bx = s, y, x
    return by, bx


def hist_svg(a):
    c = np.bincount(a.ravel(), minlength=256)[1:].astype(np.float64)  # α 1..255
    W, H, L, B = 520, 150, 44, 26
    ymax = np.log10(max(c.max(), 10))
    def X(v): return L + (v - 1) / 254 * (W - L - 8)
    def Y(n): return H - B - (np.log10(n) / ymax) * (H - B - 8) if n > 0 else H - B
    parts = ['<svg viewBox="0 0 %d %d" class="hist" role="img" aria-label="α の値ごとの画素の数(対数)">' % (W, H)]
    parts.append('<rect x="%.1f" y="8" width="%.1f" height="%d" class="band"/>' % (X(1), X(8) - X(1), H - B - 8))
    for e in range(0, int(ymax) + 1):
        y = Y(10 ** e)
        parts.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" class="grid"/><text x="%d" y="%.1f" class="ax" text-anchor="end">10<tspan dy="-4" font-size="8">%d</tspan></text>' % (L, W - 8, y, y, L - 4, y + 3, e))
    for v in (1, 8, 64, 128, 192, 255):
        parts.append('<text x="%.1f" y="%d" class="ax" text-anchor="middle">%d</text>' % (X(v), H - 8, v))
    pts = " ".join("%.1f,%.1f" % (X(v), Y(c[v - 1])) for v in range(1, 256))
    parts.append('<polyline points="%.1f,%d %s %.1f,%d" class="area"/>' % (X(1), H - B, pts, X(255), H - B))
    parts.append('<polyline points="%s" class="line"/>' % pts)
    parts.append('<line x1="%.1f" x2="%.1f" y1="8" y2="%d" class="cut"/>' % (X(8), X(8), H - B))
    parts.append("</svg>")
    tot = c.sum()
    return "".join(parts), c[:7].sum() / tot if tot else 0, c[254] / tot if tot else 0


def main():
    os.makedirs(OUT + "/img", exist_ok=True)
    labels = json.load(open(OUT + "/labels.json", encoding="utf-8"))
    names = {(r["page_path"], r["layer_id"]): r["layer_name"] for r in csv.DictReader(io.open(LINE, encoding="utf-8"))}
    rows = list(csv.DictReader(io.open(SRC + "/r_film_layers_rgba_20261120_check.csv", encoding="utf-8")))
    cards = []
    for r in rows:
        no = int(r["no"]); key = (r["page_path"], r["layer_id"])
        lab = labels["%s|%s" % key]
        im = np.array(Image.open("%s/%s" % (IMG, r["image_layer"])).convert("RGBA"))
        a = im[..., 3]
        svg, f17, f255 = hist_svg(a)
        h, w = a.shape; th = max(1, round(h * TW / w))
        Image.fromarray(on_white(im)).resize((TW, th), Image.LANCZOS).save("%s/img/%02d_w.jpg" % (OUT, no), quality=82)
        Image.fromarray(alpha_map(a)).resize((TW, th), Image.LANCZOS).save("%s/img/%02d_a.jpg" % (OUT, no), quality=85)
        y, x = best_crop(a)
        cw = im[y:y + CROP, x:x + CROP]
        Image.fromarray(on_white(cw)).save("%s/img/%02d_cw.png" % (OUT, no))
        Image.fromarray(alpha_map(cw[..., 3])).save("%s/img/%02d_ca.png" % (OUT, no))
        pg = Image.open("%s/%s" % (IMG, r["image_page"])).convert("RGB")
        pg.thumbnail((TW, TW * 2)); pg.save("%s/img/%02d_p.jpg" % (OUT, no), quality=80)
        cards.append(dict(no=no, group=r["group"], stage=lab["stage"], unit=lab["unit"], name=names.get(key, ""),
                          vis=r["layer_visible_in_csp"] == "1", a8=float(r["a8_frac"]), f17=f17, f255=f255,
                          cov=int(r["layer_px_alpha_gt0"]) / (int(r["canvas_w"]) * int(r["canvas_h"])),
                          svg=svg, crop="x %d〜%d / y %d〜%d" % (x, x + CROP, y, y + CROP), comment=lab["comment"]))
        print(no, r["group"], lab["stage"], "ok", flush=True)
    json.dump([{k: v for k, v in c.items() if k != "svg"} for c in cards], open(OUT + "/cards.json", "w", encoding="utf-8"), ensure_ascii=False)
    with open(OUT + "/cards_svg.json", "w", encoding="utf-8") as fh:
        json.dump({c["no"]: c["svg"] for c in cards}, fh)



def page():
    """cards.json から index.html を書く(画は img/ を相対で引く)。"""
    cards = json.load(open(OUT + "/cards.json", encoding="utf-8"))
    svgs = json.load(open(OUT + "/cards_svg.json", encoding="utf-8"))
    E = html.escape
    nf = sum(c["group"] == "膜" for c in cards)
    st = {}
    for c in cards:
        st.setdefault(c["group"], {}).setdefault(c["stage"], 0)
        st[c["group"]][c["stage"]] += 1
    summ = "".join('<tr><th>%s</th>%s</tr>' % (g, "".join("<td>%s %d</td>" % (s, n) for s, n in sorted(d.items())))
                   for g, d in st.items())
    out = []
    for c in cards:
        no = "%02d" % c["no"]
        out.append('''<article class="card" data-group="%(g)s" data-stage="%(s)s" id="l%(no)s">
<header><span class="no">%(no)s</span><span class="pill %(gc)s">%(g)s</span><span class="pill st">作者: %(s)s</span>
<span class="nm">%(unit)s ・ %(name)s</span><span class="vis">%(vis)s</span></header>
<dl class="nums"><div><dt>α ≥ 8 の割合</dt><dd>%(a8).3f</dd></div><div><dt>α 1〜7 の割合</dt><dd>%(f17).3f</dd></div>
<div><dt>α 255 の割合</dt><dd>%(f255).3f</dd></div><div><dt>キャンバスの被覆</dt><dd>%(cov).2f</dd></div></dl>
%(cm)s
<div class="hrow"><figure class="h">%(svg)s<figcaption>α の値ごとの画素の数(縦は対数)。橙の帯 = α 1〜7、縦線 = α 8</figcaption></figure></div>
<div class="grid">
<figure><img src="img/%(no)s_w.jpg" alt="層を白に重ねた縮小"><figcaption>層を白に重ねた縮小</figcaption></figure>
<figure><img src="img/%(no)s_a.jpg" alt="α の地図の縮小"><figcaption>α の地図の縮小</figcaption></figure>
<figure><img class="px" src="img/%(no)s_cw.png" alt="原寸の切り出し"><figcaption>原寸 1:1(%(crop)s)白に重ねた</figcaption></figure>
<figure><img class="px" src="img/%(no)s_ca.png" alt="原寸の α の地図"><figcaption>原寸 1:1 の α の地図</figcaption></figure>
<figure class="pg"><img src="img/%(no)s_p.jpg" alt="頁のプレビュー"><figcaption>頁(CSP 保存の縮小、赤枠 = 層の範囲)%(pn)s</figcaption></figure>
</div></article>''' % dict(no=no, g=E(c["group"]), gc="film" if c["group"] == "膜" else "ctl", s=E(c["stage"]),
                           unit=E(c["unit"]), name=E(c["name"]), vis="CSP で見える" if c["vis"] else "非表示",
                           a8=c["a8"], f17=c["f17"], f255=c["f255"], cov=c["cov"], svg=svgs[str(c["no"])],
                           crop=c["crop"], cm=('<p class="cm">作者のコメント: %s</p>' % E(c["comment"])) if c["comment"] else "",
                           pn="" if c["vis"] else "。この層は非表示なので頁には写っていない"))
    tpl = open(os.path.join(os.path.dirname(__file__), "r3_film_view_template.html"), encoding="utf-8").read()
    open(OUT + "/index.html", "w", encoding="utf-8").write(
        tpl.replace("{{N}}", str(len(cards))).replace("{{NF}}", str(nf)).replace("{{SUMM}}", summ).replace("{{CARDS}}", "\n".join(out)))


if __name__ == "__main__":
    import sys
    page() if sys.argv[1:] == ["--page"] else (main(), page())
