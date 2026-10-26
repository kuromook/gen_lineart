"""遮蔽物の層(occluder)。**手元に層名が在る頁だけ。**

**なぜ要るか**(作者の言明、2026-10-26):

> **手前側に何かがあると、その後ろ側のレイヤでは描かないで済ますことが多い。**
> **しかし、遮蔽物レイヤがあるということがわからないと、
> なぜ下のレイヤで何かが途切れているかわからないということになる。**

**下の層の線が途切れているのは、描き手が省いたからで、欠陥ではない。**
**遮蔽物が分からないと、その途切れを「欠落」と読む。**

**作者が挙げた4種**: 3D / フキダシ(独立したフキダシレイヤ)/ ホワイト(`white` という名が多い、
との言明)/ 黒消し。

**この表が答えない**:
- **全数の傾向。** 層名が手元に在るのは納品の manifest が覆う頁だけ(ごく一部)
- **その層が何を隠しているか。** 名前だけでは分からない —— **重なり順が要る**
  (`parent_id` / `order_in_parent` / `ancestors`。依頼8でまとめて頼む)
- **語彙が網羅的か。** 名前の候補は**作者が選ぶ**(規律6)。
  **`white` は作者自身が「多いと思う」と言っている** —— 推測である

**層名は作者の作業名なので `census/` には出さない。** 公開するのは**種別ごとの件数**だけ。
"""
import collections
import glob
import json
import os
import re

ROOT_DEFAULT = "/home/sh1/deepl/lineart/data/incoming_manuscript"

# 遮蔽物の候補。**名前の規則は作者に未確認**(`white` は作者の「多いと思う」から)
CLASSES = (
    ("3d_kind", "3D(種別で分かる。名前ではない)", None, "3d"),
    ("white", "ホワイト(白で隠す)", r"white|ホワイト|ﾎﾜｲﾄ", None),
    ("balloon", "フキダシ(独立した層)", r"フキダシ|ふきだし|吹き出し|バルーン|balloon", None),
    ("black_erase", "黒消し", r"黒消|黒ケシ|くろけし|黒けし", None),
)


def load_named_layers(root: str, drops):
    """納品の manifest から、**層名が在る頁**の層を集める。

    **落としたものを理由別に数える**(規律9)—— manifest の形が揃っていないので、
    名前の無い層・頁の鍵が無いものを黙って捨てない。
    """
    pages = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            drops.drop(f"manifest が読めない: {type(e).__name__}", path)
            continue
        if not isinstance(d, dict):
            drops.drop("manifest の最上位が dict でない", path)
            continue
        if isinstance(d.get("pages"), list):
            for p in d["pages"]:
                if isinstance(p, dict) and isinstance(p.get("layers"), list):
                    key = p.get("page") or p.get("source")
                    if not key:
                        drops.drop("頁の鍵(`page`/`source`)が無い", path)
                        continue
                    pages[key] += p["layers"]
                else:
                    drops.flag("`layers` を持たない頁の行", path)
        elif isinstance(d.get("layers"), list):
            pages[d.get("page") or os.path.basename(path)] += d["layers"]
        else:
            drops.flag("層の表を持たない manifest(別の用途)", path)
    return pages


def classify(pages, drops):
    """頁ごとに、どの遮蔽物の種別が在るか。**名前は返さない**(件数だけ)。"""
    per_class = collections.Counter()
    per_class_pages = collections.defaultdict(set)
    n_layers = n_named = 0
    pats = {k: (re.compile(p, re.I) if p else None) for k, _, p, _ in CLASSES}
    kinds = {k: v for k, _, _, v in CLASSES}
    for page, layers in pages.items():
        for L in layers:
            n_layers += 1
            name = (L.get("name") or "").strip()
            if name:
                n_named += 1
            else:
                drops.flag("層名が空(種別でしか見られない)")
            kind = (L.get("v_layer_kind") or "").strip().lower()
            for key, _, _, want_kind in CLASSES:
                hit = False
                if want_kind is not None:
                    hit = kind == want_kind
                elif pats[key] is not None and name:
                    hit = bool(pats[key].search(name))
                if hit:
                    per_class[key] += 1
                    per_class_pages[key].add(page)
    rows = []
    for key, label, pat, want_kind in CLASSES:
        rows.append([key, label,
                     "名前の規則" if want_kind is None else "種別",
                     per_class[key], len(per_class_pages[key])])
    return rows, len(pages), n_layers, n_named
